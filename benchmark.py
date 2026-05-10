#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
SHOS — BENCHMARK & MESURES SYSTÈME
=============================================================================
Objectif : produire les chiffres réels pour le Chapitre 4 du mémoire.
           FPS détection, latence E2E, CPU, RAM, température Pi, MQTT.

Plateforme cible : Raspberry Pi 4B (Linux/Raspberry Pi OS)
Compatible      : Windows et macOS (dégradé — pas de Picamera2, pas de vcgencmd)

Auteur          : SHOS Project
Date            : 2026-05
Version         : 1.1.0  ← Corrections : paho-mqtt API v2, Picamera2 natif,
                           fix timing mode full, DeprecationWarning supprimé

Usage :
    python3 benchmark.py                    # Benchmark complet (60s)
    python3 benchmark.py --mode fps         # FPS uniquement
    python3 benchmark.py --mode mqtt        # Latence MQTT uniquement
    python3 benchmark.py --mode system      # CPU/RAM/Temp uniquement
    python3 benchmark.py --duration 120     # Durée personnalisée (secondes)
    python3 benchmark.py --output mon_test  # Préfixe de fichier CSV

Sorties :
    results/benchmark_fps_YYYYMMDD_HHMMSS.csv
    results/benchmark_system_YYYYMMDD_HHMMSS.csv
    results/benchmark_mqtt_YYYYMMDD_HHMMSS.csv
    results/benchmark_summary_YYYYMMDD_HHMMSS.txt  ← Coller dans le mémoire
=============================================================================
"""

import os
import sys
import csv
import time
import json
import queue
import logging
import argparse
import platform
import threading
import statistics
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Logging structuré (niveaux INFO/WARNING/ERROR) ─────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SHOS.Benchmark")


# =============================================================================
# 1. DÉTECTION DE PLATEFORME
# =============================================================================

IS_RASPBERRY_PI = (
    platform.system() == "Linux"
    and Path("/proc/device-tree/model").exists()
)

def get_pi_model() -> str:
    """Retourne le modèle exact du Pi (pour le mémoire)."""
    try:
        with open("/proc/device-tree/model", "r") as f:
            return f.read().strip().replace("\x00", "")
    except Exception:
        return "Non-Raspberry Pi"


# =============================================================================
# 2. IMPORTS CONDITIONNELS (compatibilité cross-platform)
# =============================================================================

# psutil — monitoring CPU/RAM (toutes plateformes)
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    logger.warning("psutil non installé : CPU/RAM non mesurables. "
                   "Installer : pip install psutil")

# paho-mqtt — tests latence MQTT
# Détection API v1 vs v2 (paho >= 2.0 exige CallbackAPIVersion)
try:
    import paho.mqtt.client as mqtt
    try:
        from paho.mqtt.client import CallbackAPIVersion
        MQTT_API_VERSION = 2
    except ImportError:
        MQTT_API_VERSION = 1   # paho < 2.0
    HAS_MQTT = True
except ImportError:
    HAS_MQTT = False
    MQTT_API_VERSION = 0
    logger.warning("paho-mqtt non installé. Installer : pip install paho-mqtt")

# OpenCV — mesures FPS sur flux vidéo
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    logger.warning("opencv non installé. Installer : pip install opencv-python")

# Picamera2 — accès natif libcamera sur Raspberry Pi (préféré à v4l2)
# Sur Pi, Picamera2 fonctionne là où /dev/video0 échoue avec OpenCV+v4l2
try:
    from picamera2 import Picamera2
    HAS_PICAMERA2 = True
    logger.info("Picamera2 disponible — mode caméra natif libcamera activé")
except ImportError:
    HAS_PICAMERA2 = False

# MediaPipe — détection mains (pour mesure FPS réelle)
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    logger.warning("mediapipe non installé. Installer : pip install mediapipe")


# =============================================================================
# 3. MODULE : TEMPÉRATURE RASPBERRY PI
# =============================================================================

class TemperatureMonitor:
    """
    Lecture température CPU via vcgencmd (Pi uniquement) ou thermal zone.
    # ASVS V16.2 : logging des événements système critiques (surchauffe)
    """

    @staticmethod
    def get_cpu_temp() -> Optional[float]:
        """
        Retourne la température CPU en °C.
        Méthode 1 : vcgencmd (Raspberry Pi officiel)
        Méthode 2 : /sys/class/thermal (Linux générique)
        Méthode 3 : None si non disponible (Windows/macOS)
        """
        # Méthode 1 — vcgencmd (Raspberry Pi)
        if IS_RASPBERRY_PI:
            try:
                import subprocess
                result = subprocess.run(
                    ["vcgencmd", "measure_temp"],
                    capture_output=True, text=True, timeout=2
                )
                # Format : "temp=47.8'C"
                raw = result.stdout.strip()
                temp = float(raw.replace("temp=", "").replace("'C", ""))
                return round(temp, 1)
            except Exception:
                pass

            # Méthode 2 — thermal zone (Linux)
            try:
                with open("/sys/class/thermal/thermal_zone0/temp") as f:
                    return round(int(f.read().strip()) / 1000, 1)
            except Exception:
                pass

        # Méthode 3 — psutil (partiel, pas toujours dispo)
        if HAS_PSUTIL:
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    for key in ["cpu_thermal", "coretemp", "acpitz"]:
                        if key in temps and temps[key]:
                            return round(temps[key][0].current, 1)
            except Exception:
                pass

        return None


# =============================================================================
# 4. MODULE : MESURES SYSTÈME (CPU, RAM, TEMP)
# =============================================================================

class SystemMonitor:
    """
    Collecte en continu : CPU%, RAM%, température.
    Tourne dans un thread séparé pour ne pas bloquer les autres mesures.
    # ASVS V15.4 : safe concurrency — accès aux listes protégé par Lock
    """

    def __init__(self, interval_s: float = 1.0):
        self.interval = interval_s
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()  # ASVS V15.4 : mutex partagé

        # Stockage des mesures
        self._cpu_samples: list[float] = []
        self._ram_samples: list[float] = []
        self._temp_samples: list[float] = []
        self._timestamps: list[float] = []

        self.temp_monitor = TemperatureMonitor()

    def start(self) -> None:
        """Démarre la collecte en arrière-plan."""
        if not HAS_PSUTIL:
            logger.error("psutil requis pour SystemMonitor.")
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._collect_loop,
            name="SystemMonitor",
            daemon=True  # Thread démon : s'arrête proprement si le main meurt
        )
        self._thread.start()
        logger.info("SystemMonitor démarré (intervalle=%.1fs)", self.interval)

    def stop(self) -> None:
        """Arrête la collecte proprement."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        logger.info("SystemMonitor arrêté — %d échantillons collectés",
                    len(self._cpu_samples))

    def _collect_loop(self) -> None:
        """Boucle interne de collecte (thread séparé)."""
        # Premier appel cpu_percent retourne toujours 0.0 — on l'ignore
        psutil.cpu_percent(interval=None)

        while self._running:
            t_start = time.monotonic()

            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent
            temp = self.temp_monitor.get_cpu_temp()

            with self._lock:  # ASVS V15.4 : accès thread-safe
                self._cpu_samples.append(cpu)
                self._ram_samples.append(ram)
                self._timestamps.append(time.time())
                if temp is not None:
                    self._temp_samples.append(temp)

                # Alerte surchauffe (important pour le Pi)
                if temp and temp > 80.0:
                    logger.warning("⚠ TEMPÉRATURE CRITIQUE : %.1f°C", temp)

            elapsed = time.monotonic() - t_start
            sleep_time = max(0, self.interval - elapsed)
            time.sleep(sleep_time)

    def get_stats(self) -> dict:
        """Retourne les statistiques calculées (thread-safe)."""
        with self._lock:
            cpu = list(self._cpu_samples)
            ram = list(self._ram_samples)
            temp = list(self._temp_samples)

        def stats_dict(data: list, unit: str) -> dict:
            if not data:
                return {"min": None, "max": None, "mean": None,
                        "stdev": None, "unit": unit}
            return {
                "min": round(min(data), 2),
                "max": round(max(data), 2),
                "mean": round(statistics.mean(data), 2),
                "stdev": round(statistics.stdev(data), 2) if len(data) > 1 else 0.0,
                "unit": unit,
                "samples": len(data),
            }

        return {
            "cpu_percent": stats_dict(cpu, "%"),
            "ram_percent": stats_dict(ram, "%"),
            "temperature_c": stats_dict(temp, "°C"),
        }

    def get_raw_samples(self) -> list[dict]:
        """Retourne toutes les lignes pour export CSV."""
        with self._lock:
            rows = []
            for i, ts in enumerate(self._timestamps):
                row = {
                    "timestamp": datetime.fromtimestamp(ts).isoformat(),
                    "cpu_percent": self._cpu_samples[i] if i < len(self._cpu_samples) else None,
                    "ram_percent": self._ram_samples[i] if i < len(self._ram_samples) else None,
                    "temp_c": self._temp_samples[i] if i < len(self._temp_samples) else None,
                }
                rows.append(row)
        return rows


# =============================================================================
# 5. MODULE : MESURES FPS — OBSERVATEUR MQTT (sans conflit caméra)
# =============================================================================

class VisionMQTTObserver:
    """
    Mesure le FPS et la latence E2E de vision_service via MQTT.

    POURQUOI CETTE APPROCHE ?
    ─────────────────────────
    Sur Raspberry Pi, la caméra CSI (Picamera2/libcamera) est une ressource
    exclusive : un seul processus peut l'ouvrir à la fois. Quand vision_service
    tourne déjà (ce qui est le cas en production), toute tentative d'ouvrir
    /dev/video0 depuis le benchmark échoue avec "camera index out of range".

    SOLUTION : ne pas toucher la caméra. vision_service publie déjà sur MQTT
    à chaque frame traitée. On s'y abonne et on compte les messages/seconde.
    La latence E2E = timestamp embarqué par vision_service dans le payload
    MQTT − temps de réception ici. C'est la vraie latence de bout en bout
    (capture → traitement MediaPipe → publish MQTT → réception benchmark).

    TOPICS ATTENDUS (vision_service.py) :
        shos/camera/vision  ← payload JSON avec champ "timestamp" (time.time())
                               et optionnellement "fps", "gesture", "hands"

    Si vision_service ne publie pas encore ces champs, le module mesure
    uniquement le débit (messages/seconde) et log un avertissement.

    Métriques produites :
        fps_observe     : messages MQTT reçus par seconde (= FPS réel pipeline)
        latency_e2e_ms  : latence capture→réception (si timestamp dans payload)
        frames_received : total messages reçus pendant la durée
        topic_observed  : topic MQTT écouté
    """

    def __init__(
        self,
        broker: str = "localhost",
        port: int = 1883,
        topic: str = "shos/camera/vision",
    ):
        if not HAS_MQTT:
            raise RuntimeError(
                "paho-mqtt requis. Installer : pip install paho-mqtt"
            )
        self.broker = broker
        self.port = port
        self.topic = topic

        # ASVS V15.4 : accès concurrent protégé par Lock
        self._lock = threading.Lock()
        self._frame_rx_times: list[float] = []   # time.monotonic() à chaque réception
        self._e2e_latencies_ms: list[float] = []  # latence E2E calculée
        self._frames_with_ts: int = 0             # frames avec timestamp valide
        self._frames_without_ts: int = 0          # frames sans timestamp (fallback)

    def run(self, duration_s: int = 60) -> dict:
        """
        S'abonne au topic vision et observe pendant `duration_s` secondes.
        Retourne les métriques calculées.
        """
        logger.info("VisionMQTTObserver — écoute %s pendant %ds",
                    self.topic, duration_s)

        # Création du client MQTT (API v2 ou v1 selon version installée)
        if MQTT_API_VERSION == 2:
            client = mqtt.Client(
                callback_api_version=CallbackAPIVersion.VERSION2,
                client_id="shos_benchmark_vision_obs",
            )
        else:
            client = mqtt.Client(client_id="shos_benchmark_vision_obs")

        # ── Callbacks ──────────────────────────────────────────────────────
        def on_connect(client, userdata, flags, rc, *extra):
            rc_code = rc if isinstance(rc, int) else rc.value
            if rc_code == 0:
                client.subscribe(self.topic, qos=0)
                logger.info("Abonné à %s (broker=%s:%d)",
                            self.topic, self.broker, self.port)
            else:
                logger.error("Connexion MQTT refusée (rc=%d) — "
                             "vision_service tourne-t-il ?", rc_code)

        def on_message(client, userdata, msg):
            t_receive = time.monotonic()
            t_wall = time.time()   # Pour latence E2E absolue

            with self._lock:
                self._frame_rx_times.append(t_receive)

                # Tentative de lecture du timestamp vision_service
                try:
                    payload = json.loads(msg.payload.decode("utf-8"))
                    ts_sent = payload.get("timestamp")   # time.time() côté service

                    if ts_sent and isinstance(ts_sent, (int, float)):
                        # Latence E2E absolue (vision_service → broker → ici)
                        latency_ms = (t_wall - ts_sent) * 1000
                        # On filtre les valeurs aberrantes (horloge désync > 5s)
                        if 0 < latency_ms < 5000:
                            self._e2e_latencies_ms.append(round(latency_ms, 3))
                            self._frames_with_ts += 1
                        else:
                            self._frames_without_ts += 1
                    else:
                        self._frames_without_ts += 1

                except (json.JSONDecodeError, UnicodeDecodeError):
                    # Payload non-JSON (ex: MJPEG brut) — on compte juste le débit
                    self._frames_without_ts += 1

        def on_disconnect(client, userdata, *args):
            logger.warning("MQTT déconnecté pendant l'observation.")

        client.on_connect = on_connect
        client.on_message = on_message
        client.on_disconnect = on_disconnect

        # ── Connexion et observation ────────────────────────────────────────
        try:
            client.connect(self.broker, self.port, keepalive=60)
            client.loop_start()

            t_start = time.monotonic()
            t_last_log = t_start

            logger.info("Observation démarrée — attente des frames de vision_service...")

            while time.monotonic() - t_start < duration_s:
                time.sleep(1.0)

                with self._lock:
                    n = len(self._frame_rx_times)
                elapsed = time.monotonic() - t_start

                # Log progression toutes les 10s
                if time.monotonic() - t_last_log >= 10.0:
                    fps_now = n / max(elapsed, 0.001)
                    logger.info("%.0fs écoulées — %.1f FPS observés (%d frames)",
                                elapsed, fps_now, n)
                    t_last_log = time.monotonic()

                # Avertissement si aucune frame reçue après 5s
                if elapsed > 5.0 and n == 0:
                    logger.warning(
                        "⚠ Aucune frame reçue sur %s après 5s. "
                        "Vérifier que vision_service publie sur ce topic.",
                        self.topic
                    )
                    # On arrête tôt pour ne pas bloquer
                    break

        except ConnectionRefusedError:
            logger.error("Broker MQTT inaccessible (%s:%d).", self.broker, self.port)
            return {"error": "broker_unreachable"}
        finally:
            client.loop_stop()
            client.disconnect()

        return self._compute_vision_stats(duration_s)

    def _compute_vision_stats(self, duration_s: int) -> dict:
        """Calcule les métriques FPS et latence E2E depuis les observations."""
        with self._lock:
            rx_times = list(self._frame_rx_times)
            e2e = list(self._e2e_latencies_ms)
            n_with = self._frames_with_ts
            n_without = self._frames_without_ts

        total_frames = len(rx_times)

        if total_frames == 0:
            return {
                "error": "no_frames_received",
                "topic": self.topic,
                "hint": (
                    "vision_service ne publie pas sur ce topic, ou le service "
                    "n'est pas démarré. Vérifier : mosquitto_sub -t '#' -v"
                ),
            }

        # FPS observé = débit réel du pipeline complet
        fps_observed = round(total_frames / duration_s, 2)

        # Inter-frame jitter (stabilité du flux)
        if len(rx_times) > 1:
            intervals_ms = [
                (rx_times[i+1] - rx_times[i]) * 1000
                for i in range(len(rx_times) - 1)
            ]
            jitter_ms = round(statistics.stdev(intervals_ms), 2)
            interval_mean_ms = round(statistics.mean(intervals_ms), 2)
        else:
            jitter_ms = None
            interval_mean_ms = None

        # Latence E2E (uniquement si timestamp présent dans payload)
        e2e_stats = {}
        if e2e:
            sorted_e2e = sorted(e2e)
            e2e_stats = {
                "mean_ms": round(statistics.mean(e2e), 2),
                "min_ms": round(min(e2e), 2),
                "max_ms": round(max(e2e), 2),
                "median_ms": round(statistics.median(e2e), 2),
                "p95_ms": round(sorted_e2e[int(len(sorted_e2e) * 0.95)], 2),
                "stdev_ms": round(statistics.stdev(e2e), 2) if len(e2e) > 1 else 0.0,
                "samples": len(e2e),
            }
        elif n_without > 0:
            logger.warning(
                "%d frames reçues sans timestamp — ajouter "
                "'timestamp': time.time() dans vision_service.py pour "
                "mesurer la latence E2E.",
                n_without
            )

        return {
            "fps_observed": fps_observed,
            "frames_received": total_frames,
            "frames_with_timestamp": n_with,
            "frames_without_timestamp": n_without,
            "duration_s": duration_s,
            "interval_mean_ms": interval_mean_ms,
            "jitter_ms": jitter_ms,
            "latency_e2e": e2e_stats,
            "topic_observed": self.topic,
        }


# =============================================================================
# 6. MODULE : MESURES LATENCE MQTT
# =============================================================================

class MQTTLatencyBenchmark:
    """
    Mesure la latence round-trip MQTT réelle (publish → broker → receive).

    PROTOCOLE CORRIGÉ v1.2 :
    ────────────────────────
    Le client s'abonne ET publie sur le même topic (TOPIC_ECHO).
    Le broker Mosquitto redistribue chaque message à tous ses abonnés,
    y compris l'émetteur lui-même. On mesure le Δt entre l'envoi et
    la réception de ce même message — c'est le vrai round-trip broker.

    Pourquoi ce changement ?
    Avant : publish sur TOPIC_PING, subscribe sur TOPIC_PONG.
    Problème : personne ne faisait le bridge ping→pong → 100% timeout.
    Maintenant : un seul topic, le broker fait l'écho naturellement.

    Ce test mesure exactement ce qui compte pour le mémoire :
        publish() → réseau loopback → broker Mosquitto → callback on_message
    = latence du bus de communication inter-modules SHOS.

    # ASVS V15.4 : threading.Event pour synchronisation thread-safe
    """

    TOPIC_ECHO = "shos/benchmark/echo"   # topic unique publish + subscribe

    def __init__(self, broker: str = "localhost", port: int = 1883):
        self.broker = broker
        self.port = port
        self._latencies_ms: list[float] = []
        self._errors: int = 0
        self._echo_event = threading.Event()
        self._pending_seq: int = -1       # séquence en attente de retour

    def run(self, n_messages: int = 100, timeout_s: float = 2.0) -> dict:
        """
        Envoie n_messages en self-echo et mesure la latence round-trip.
        """
        if not HAS_MQTT:
            logger.error("paho-mqtt requis pour MQTTLatencyBenchmark.")
            return {"error": "paho_mqtt_not_installed"}

        # Client MQTT — API v2 si disponible, v1 sinon (rétrocompatibilité)
        # ASVS V15.4 : synchronisation via threading.Event
        if MQTT_API_VERSION == 2:
            client = mqtt.Client(
                callback_api_version=CallbackAPIVersion.VERSION2,
                client_id="shos_benchmark_latency",
            )
        else:
            client = mqtt.Client(client_id="shos_benchmark_latency")

        def on_connect(client, userdata, flags, rc, *args):
            rc_code = rc if isinstance(rc, int) else rc.value
            if rc_code == 0:
                # S'abonner sur le topic echo avant de publier
                client.subscribe(self.TOPIC_ECHO, qos=0)
                logger.info("MQTT connecté à %s:%d — abonné à %s",
                            self.broker, self.port, self.TOPIC_ECHO)
            else:
                logger.error("MQTT connexion refusée (rc=%d)", rc_code)

        def on_message(client, userdata, msg):
            # Réception de notre propre message — calcul Δt immédiat
            t_receive = time.monotonic()
            try:
                payload = json.loads(msg.payload.decode())
                seq = payload.get("seq", -1)
                t_sent = payload.get("t_mono", 0)

                # On ne traite que le message qu'on attend (évite les doublons)
                if seq == self._pending_seq and t_sent:
                    latency_ms = (t_receive - t_sent) * 1000
                    self._latencies_ms.append(round(latency_ms, 3))
                    self._echo_event.set()

            except (json.JSONDecodeError, KeyError, TypeError):
                self._errors += 1
                self._echo_event.set()  # débloquer même sur erreur

        client.on_connect = on_connect
        client.on_message = on_message

        try:
            client.connect(self.broker, self.port, keepalive=60)
            client.loop_start()
            time.sleep(0.3)  # laisser on_connect s'exécuter + subscription active

            logger.info("Benchmark MQTT round-trip : %d messages → %s",
                        n_messages, self.TOPIC_ECHO)

            for i in range(n_messages):
                self._pending_seq = i
                self._echo_event.clear()

                payload = json.dumps({
                    "seq": i,
                    "t_mono": time.monotonic(),   # monotonic = précis, pas de drift
                    "t_wall": time.time(),
                    "source": "shos_benchmark",
                })
                client.publish(self.TOPIC_ECHO, payload, qos=0)

                # Attente de notre propre message en retour
                received = self._echo_event.wait(timeout=timeout_s)
                if not received:
                    self._errors += 1
                    logger.warning("Timeout message #%d (%.1fs) — "
                                   "broker surchargé ?", i, timeout_s)

                time.sleep(0.02)  # 20ms entre messages — débit réaliste

        except ConnectionRefusedError:
            logger.error("Broker MQTT inaccessible (%s:%d) — "
                         "Mosquitto démarré ?", self.broker, self.port)
            return {"error": "broker_unreachable"}
        except Exception as exc:
            logger.error("Erreur MQTT inattendue : %s", exc)
            return {"error": str(exc)}
        finally:
            client.loop_stop()
            client.disconnect()

        return self._compute_mqtt_stats(n_messages)

    def _compute_mqtt_stats(self, n_sent: int) -> dict:
        """Calcule les statistiques de latence MQTT."""
        if not self._latencies_ms:
            return {
                "error": "no_responses",
                "sent": n_sent,
                "errors": self._errors,
            }

        sorted_lat = sorted(self._latencies_ms)
        return {
            "sent": n_sent,
            "received": len(self._latencies_ms),
            "errors": self._errors,
            "loss_percent": round(
                (self._errors / n_sent) * 100, 2
            ),
            "latency_ms": {
                "min": round(min(self._latencies_ms), 3),
                "max": round(max(self._latencies_ms), 3),
                "mean": round(statistics.mean(self._latencies_ms), 3),
                "median": round(statistics.median(self._latencies_ms), 3),
                "stdev": round(statistics.stdev(self._latencies_ms), 3)
                if len(self._latencies_ms) > 1 else 0.0,
                "p95": round(sorted_lat[int(len(sorted_lat) * 0.95)], 3),
                "p99": round(sorted_lat[int(len(sorted_lat) * 0.99)], 3),
            },
        }


# =============================================================================
# 7. EXPORT CSV ET RAPPORT TEXTE
# =============================================================================

class ResultsExporter:
    """
    Exporte tous les résultats en CSV et génère le résumé texte
    à coller directement dans le Chapitre 4 du mémoire.

    # ASVS V16.2 : logging structuré des sorties de benchmark
    """

    def __init__(self, output_prefix: str = "benchmark"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path("results")
        self.output_dir.mkdir(exist_ok=True)
        self.prefix = f"{output_prefix}_{timestamp}"

    def export_system_csv(self, samples: list[dict]) -> Path:
        """Exporte les mesures système (CPU/RAM/temp) en CSV."""
        path = self.output_dir / f"{self.prefix}_system.csv"
        if not samples:
            logger.warning("Aucune donnée système à exporter.")
            return path
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=samples[0].keys())
            writer.writeheader()
            writer.writerows(samples)
        logger.info("CSV système exporté : %s (%d lignes)", path, len(samples))
        return path

    def export_fps_csv(self, fps_result: dict) -> Path:
        """Exporte le résumé FPS/Vision en CSV."""
        path = self.output_dir / f"{self.prefix}_fps.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Métrique", "Valeur", "Unité"])
            writer.writerow(["FPS observé", fps_result.get("fps_observed"), "fps"])
            writer.writerow(["Frames reçues", fps_result.get("frames_received"), "frames"])
            writer.writerow(["Durée test", fps_result.get("duration_s"), "s"])
            writer.writerow(["Topic MQTT", fps_result.get("topic_observed"), ""])
            writer.writerow(["Intervalle inter-frame", fps_result.get("interval_mean_ms"), "ms"])
            writer.writerow(["Jitter", fps_result.get("jitter_ms"), "ms"])
            e2e = fps_result.get("latency_e2e", {})
            if e2e:
                writer.writerow(["Latence E2E moyenne", e2e.get("mean_ms"), "ms"])
                writer.writerow(["Latence E2E min", e2e.get("min_ms"), "ms"])
                writer.writerow(["Latence E2E max", e2e.get("max_ms"), "ms"])
                writer.writerow(["Latence E2E P95", e2e.get("p95_ms"), "ms"])
        logger.info("CSV FPS/Vision exporté : %s", path)
        return path

    def export_mqtt_csv(self, mqtt_result: dict) -> Path:
        """Exporte les statistiques MQTT en CSV."""
        path = self.output_dir / f"{self.prefix}_mqtt.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Métrique", "Valeur", "Unité"])
            writer.writerow(["Messages envoyés", mqtt_result.get("sent"), ""])
            writer.writerow(["Messages reçus", mqtt_result.get("received"), ""])
            writer.writerow(["Erreurs/Timeout", mqtt_result.get("errors"), ""])
            writer.writerow(["Perte paquets", mqtt_result.get("loss_percent"), "%"])
            lat = mqtt_result.get("latency_ms", {})
            writer.writerow(["Latence min", lat.get("min"), "ms"])
            writer.writerow(["Latence max", lat.get("max"), "ms"])
            writer.writerow(["Latence moyenne", lat.get("mean"), "ms"])
            writer.writerow(["Latence médiane", lat.get("median"), "ms"])
            writer.writerow(["Latence P95", lat.get("p95"), "ms"])
            writer.writerow(["Latence P99", lat.get("p99"), "ms"])
        logger.info("CSV MQTT exporté : %s", path)
        return path

    def generate_summary(
        self,
        fps_result: dict,
        sys_stats: dict,
        mqtt_result: dict,
        pi_model: str,
        duration_s: int,
    ) -> Path:
        """
        Génère un rapport texte structuré — prêt à copier dans le mémoire.
        Aucune donnée fictive : uniquement les vrais chiffres mesurés.
        """
        path = self.output_dir / f"{self.prefix}_summary.txt"
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            "=" * 70,
            "SHOS — RAPPORT DE BENCHMARK",
            f"Date       : {now}",
            f"Plateforme : {pi_model}",
            f"OS         : {platform.platform()}",
            f"Python     : {sys.version.split()[0]}",
            f"Durée test : {duration_s} secondes",
            "=" * 70,
            "",
            "── 1. PERFORMANCES VIDÉO / DÉTECTION ──────────────────────────────",
        ]

        if fps_result and "error" not in fps_result:
            lines += [
                f"  FPS observé (pipeline)  : {fps_result.get('fps_observed')} fps",
                f"  Frames reçues           : {fps_result.get('frames_received')}",
                f"  Topic MQTT observé      : {fps_result.get('topic_observed')}",
                f"  Intervalle inter-frame  : {fps_result.get('interval_mean_ms')} ms (moy)",
                f"  Jitter inter-frame      : {fps_result.get('jitter_ms')} ms (écart-type)",
            ]
            e2e = fps_result.get("latency_e2e", {})
            if e2e:
                lines += [
                    f"  Latence E2E (moy)       : {e2e.get('mean_ms')} ms",
                    f"  Latence E2E (min)       : {e2e.get('min_ms')} ms",
                    f"  Latence E2E (max)       : {e2e.get('max_ms')} ms",
                    f"  Latence E2E (P95)       : {e2e.get('p95_ms')} ms",
                    f"  Frames avec timestamp   : {fps_result.get('frames_with_timestamp')}",
                ]
            else:
                lines.append(
                    "  Latence E2E             : [ajouter 'timestamp':time.time() "
                    "dans vision_service.py]"
                )
        else:
            lines.append(f"  [Non mesuré — {fps_result.get('error', 'module absent')}]")

        lines += [
            "",
            "── 2. RESSOURCES SYSTÈME ──────────────────────────────────────────",
        ]

        if sys_stats:
            cpu = sys_stats.get("cpu_percent", {})
            ram = sys_stats.get("ram_percent", {})
            temp = sys_stats.get("temperature_c", {})

            if cpu.get("mean") is not None:
                lines += [
                    f"  CPU moyen               : {cpu.get('mean')} %",
                    f"  CPU max (pic)           : {cpu.get('max')} %",
                    f"  CPU min                 : {cpu.get('min')} %",
                ]
            if ram.get("mean") is not None:
                lines += [
                    f"  RAM utilisée (moy)      : {ram.get('mean')} %",
                    f"  RAM utilisée (max)      : {ram.get('max')} %",
                ]
            if temp.get("mean") is not None:
                lines += [
                    f"  Température CPU (moy)   : {temp.get('mean')} °C",
                    f"  Température CPU (max)   : {temp.get('max')} °C",
                ]
                if temp.get("max", 0) > 80:
                    lines.append("  ⚠ ATTENTION : Température > 80°C — refroidissement nécessaire")
            else:
                lines.append("  Température             : [Non disponible sur cette plateforme]")
        else:
            lines.append("  [Non mesuré — psutil absent]")

        lines += [
            "",
            "── 3. LATENCE MQTT ────────────────────────────────────────────────",
        ]

        if mqtt_result and "error" not in mqtt_result:
            lat = mqtt_result.get("latency_ms", {})
            lines += [
                f"  Messages envoyés        : {mqtt_result.get('sent')}",
                f"  Messages reçus          : {mqtt_result.get('received')}",
                f"  Perte paquets           : {mqtt_result.get('loss_percent')} %",
                f"  Latence moyenne         : {lat.get('mean')} ms",
                f"  Latence médiane         : {lat.get('median')} ms",
                f"  Latence min             : {lat.get('min')} ms",
                f"  Latence max             : {lat.get('max')} ms",
                f"  Latence P95             : {lat.get('p95')} ms",
                f"  Latence P99             : {lat.get('p99')} ms",
            ]
        else:
            err = mqtt_result.get("error", "module absent") if mqtt_result else "non lancé"
            lines.append(f"  [Non mesuré — {err}]")
            if mqtt_result and mqtt_result.get("error") == "broker_unreachable":
                lines.append("  → Démarrer Mosquitto : sudo systemctl start mosquitto")

        lines += [
            "",
            "=" * 70,
            "FIN DU RAPPORT — Fichiers CSV détaillés dans le dossier results/",
            "=" * 70,
        ]

        content = "\n".join(lines)
        path.write_text(content, encoding="utf-8")
        logger.info("Résumé exporté : %s", path)

        # Affichage aussi dans le terminal
        print("\n" + content + "\n")
        return path


# =============================================================================
# 8. ORCHESTRATEUR PRINCIPAL
# =============================================================================

def run_benchmark(args: argparse.Namespace) -> None:
    """
    Point d'entrée principal — orchestre les modules selon le mode choisi.

    Mode full corrigé v1.1 :
        Le SystemMonitor démarre EN PREMIER et tourne tout au long
        des tests FPS ET MQTT. Il s'arrête seulement quand tout est fini.
        Bug v1.0 : le monitor s'arrêtait immédiatement si la caméra échouait.
    """
    logger.info("╔══ SHOS BENCHMARK démarré ══╗")
    logger.info("  Plateforme : %s", get_pi_model())
    logger.info("  Mode       : %s", args.mode)
    logger.info("  Durée      : %ds", args.duration)

    exporter = ResultsExporter(output_prefix=args.output)
    fps_result: dict = {}
    sys_stats: dict = {}
    mqtt_result: dict = {}
    sys_monitor = None

    # ── SystemMonitor : démarre AVANT tout le reste ──
    # En mode full, il couvre FPS + MQTT. On l'arrête tout à la fin.
    if args.mode in ("system", "full"):
        sys_monitor = SystemMonitor(interval_s=1.0)
        sys_monitor.start()

    # ── Mode "system" uniquement : attendre duration_s ──
    if args.mode == "system":
        logger.info("Monitoring système (%ds)...", args.duration)
        time.sleep(args.duration)

    # ── Mode "fps" ou "full" : observation MQTT de vision_service ──
    # N'ouvre PAS la caméra — écoute ce que vision_service publie déjà
    if args.mode in ("fps", "full"):
        fps_bench = VisionMQTTObserver(
            broker=args.mqtt_broker,
            port=args.mqtt_port,
            topic=args.vision_topic,
        )
        logger.info("Lancement observation vision via MQTT (%ds)...", args.duration)
        fps_result = fps_bench.run(duration_s=args.duration)
        exporter.export_fps_csv(fps_result)

    # ── Mode "mqtt" ou "full" : benchmark MQTT ──
    if args.mode in ("mqtt", "full"):
        logger.info("Lancement MQTT benchmark (%d messages)...", args.mqtt_messages)
        mqtt_bench = MQTTLatencyBenchmark(
            broker=args.mqtt_broker,
            port=args.mqtt_port,
        )
        mqtt_result = mqtt_bench.run(
            n_messages=args.mqtt_messages,
            timeout_s=2.0,
        )
        exporter.export_mqtt_csv(mqtt_result)

    # ── Arrêt du SystemMonitor APRÈS tous les tests ──
    # Garantit que les stats système couvrent l'ensemble de la session
    if sys_monitor:
        sys_monitor.stop()
        sys_stats = sys_monitor.get_stats()
        exporter.export_system_csv(sys_monitor.get_raw_samples())

    # ── Génération du rapport final ──
    summary_path = exporter.generate_summary(
        fps_result=fps_result,
        sys_stats=sys_stats,
        mqtt_result=mqtt_result,
        pi_model=get_pi_model(),
        duration_s=args.duration,
    )

    logger.info("╚══ Benchmark terminé. Résumé : %s ══╝", summary_path)


# =============================================================================
# 9. CLI — ARGUMENTS EN LIGNE DE COMMANDE
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SHOS Benchmark — Mesures pour mémoire (Raspberry Pi 4B)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python3 benchmark.py                        # Benchmark complet 60s
  python3 benchmark.py --mode fps --duration 30
  python3 benchmark.py --mode mqtt --mqtt-broker 192.168.1.100
  python3 benchmark.py --mode system --duration 120
  python3 benchmark.py --output test_v2       # Préfixe fichiers CSV
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["full", "fps", "system", "mqtt"],
        default="full",
        help="Module à exécuter (défaut: full)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Durée du test en secondes (défaut: 60)",
    )
    parser.add_argument(
        "--vision-topic",
        default="shos/camera/vision",
        dest="vision_topic",
        help="Topic MQTT de vision_service à observer (défaut: shos/camera/vision)",
    )
    parser.add_argument(
        "--mqtt-broker",
        default="localhost",
        dest="mqtt_broker",
        help="Adresse du broker MQTT (défaut: localhost)",
    )
    parser.add_argument(
        "--mqtt-port",
        type=int,
        default=1883,
        dest="mqtt_port",
        help="Port du broker MQTT (défaut: 1883)",
    )
    parser.add_argument(
        "--mqtt-messages",
        type=int,
        default=100,
        dest="mqtt_messages",
        help="Nombre de messages pour le test MQTT (défaut: 100)",
    )
    parser.add_argument(
        "--output",
        default="benchmark",
        help="Préfixe des fichiers de sortie (défaut: benchmark)",
    )
    return parser.parse_args()


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

if __name__ == "__main__":
    args = parse_args()

    # Vérification des dépendances avant de lancer
    missing = []
    if not HAS_PSUTIL:
        missing.append("psutil")
    if args.mode in ("fps", "full") and not HAS_OPENCV:
        missing.append("opencv-python")
    if args.mode in ("fps", "full") and not HAS_MEDIAPIPE:
        missing.append("mediapipe")
    if args.mode in ("mqtt", "full") and not HAS_MQTT:
        missing.append("paho-mqtt")

    if missing:
        print("\n⚠ Dépendances manquantes. Installer avec :")
        print(f"  pip install {' '.join(missing)} --break-system-packages\n")
        if args.mode == "full":
            print("Mode 'full' dégradé — modules disponibles uniquement.\n")

    run_benchmark(args)
