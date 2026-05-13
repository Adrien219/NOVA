#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
SHOS · MODULE VIDEO STREAMING
=============================================================================
Structure : plugins/modules/video_streaming/main.py

Fonctions :
  • Proxy MJPEG dual-source : Picamera2 (:5051) ↔ ESP32-CAM (IP configurable)
  • WebSocket temps réel → métriques capteurs MQTT, FPS, CPU, temp
  • Objets détectés MediaPipe (lu depuis MQTT shos/camera/vision)
  • Transcription vocale Whisper (endpoint POST /api/voice/transcribe)
  • Commandes vocales parsées → actions système via MQTT

Architecture :
  nova.py (port 5000) monte ce module via blueprint Flask
  OU ce module tourne de façon autonome sur le port 5055

Usage autonome :
  python3 main.py
  python3 main.py --port 5055 --esp32 192.168.1.50

Usage via nova.py (blueprint) :
  from plugins.modules.video_streaming.main import streaming_bp
  app.register_blueprint(streaming_bp, url_prefix='/streaming')

Auteur  : SHOS Project
Version : 1.0.0
Date    : 2026-05
=============================================================================
"""

import os
import sys
import json
import time
import queue
import logging
import argparse
import threading
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional

# ── Flask + SocketIO ──────────────────────────────────────────────────────────
from flask import (
    Flask, Blueprint, Response, request, jsonify,
    render_template, stream_with_context
)
from flask_socketio import SocketIO, emit

# ── MQTT ─────────────────────────────────────────────────────────────────────
try:
    import paho.mqtt.client as mqtt
    try:
        from paho.mqtt.client import CallbackAPIVersion
        _MQTT_V2 = True
    except ImportError:
        _MQTT_V2 = False
    HAS_MQTT = True
except ImportError:
    HAS_MQTT = False

# ── Requests (proxy stream) ───────────────────────────────────────────────────
try:
    import requests as _requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# ── psutil (métriques système) ────────────────────────────────────────────────
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# ── Whisper (transcription locale) ───────────────────────────────────────────
try:
    import whisper as _whisper
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SHOS.VideoStreaming")

# =============================================================================
# CONFIG
# =============================================================================

CONFIG = {
    # Sources vidéo
    "picamera_url": "http://localhost:5051/video_feed",
    "esp32_url":    "",          # ex: http://192.168.1.50/stream — vide = désactivé
    "default_source": "picamera",

    # MQTT
    "mqtt_broker": "localhost",
    "mqtt_port":   1883,

    # Flask
    "host": "0.0.0.0",
    "port": 5055,

    # Whisper
    "whisper_model": "base",     # tiny/base/small/medium — base = bon équilibre Pi
    "whisper_language": "fr",

    # Métriques
    "metrics_interval_s": 1.0,

    # MQTT topics écoutés
    "topics": {
        "vision":       "shos/camera/vision",
        "sensors_norm": "shos/sensors/normalized",
        "sensors_raw":  "shos/sensors/raw",
        "telemetry":    "nova/system/telemetry",
        "status":       "shos/system/status",
    },
    # MQTT topics publiés
    "publish": {
        "voice_command": "shos/voice/command",
        "voice_text":    "shos/voice/transcript",
        "stream_status": "shos/streaming/status",
    },
}

# Essayer de charger config.json local
_cfg_path = Path(__file__).parent / "config.json"
if _cfg_path.exists():
    try:
        with open(_cfg_path) as f:
            _overrides = json.load(f)
        CONFIG.update(_overrides)
        logger.info("config.json chargé depuis %s", _cfg_path)
    except Exception as e:
        logger.warning("config.json illisible : %s", e)

# =============================================================================
# STATE PARTAGÉ (thread-safe)
# =============================================================================

class StreamingState:
    """
    État global du module — partagé entre les threads MQTT, métriques et SocketIO.
    Tous les accès concurrents passent par un Lock.
    # ASVS V15.4 : mutex pour accès concurrent thread-safe
    """

    def __init__(self):
        self._lock = threading.Lock()

        # Source vidéo active
        self.active_source: str = CONFIG["default_source"]  # "picamera" | "esp32"

        # Dernières métriques reçues via MQTT
        self.vision: dict = {}          # shos/camera/vision
        self.sensors: dict = {}         # shos/sensors/normalized
        self.sensors_raw: dict = {}     # shos/sensors/raw
        self.telemetry: dict = {}       # nova/system/telemetry
        self.system_status: str = ""

        # Métriques système locales (psutil)
        self.cpu_percent: float = 0.0
        self.ram_percent: float = 0.0
        self.cpu_temp: Optional[float] = None

        # Transcriptions vocales
        self.last_transcript: str = ""
        self.voice_history: list = []   # [{ts, text, source}]
        self.voice_active: bool = False

        # Alertes actives
        self.alerts: list = []          # [{level, message, ts}]

        # Connexion streaming
        self.stream_clients: int = 0
        self.stream_fps_observed: float = 0.0
        self._frame_times: list = []

    def update(self, key: str, value):
        with self._lock:
            setattr(self, key, value)

    def get_snapshot(self) -> dict:
        """Retourne un snapshot JSON-serialisable de tout l'état."""
        with self._lock:
            return {
                "source":         self.active_source,
                "vision":         self.vision,
                "sensors":        self.sensors,
                "sensors_raw":    self.sensors_raw,
                "telemetry":      self.telemetry,
                "system_status":  self.system_status,
                "cpu":            self.cpu_percent,
                "ram":            self.ram_percent,
                "temp":           self.cpu_temp,
                "fps":            self.stream_fps_observed,
                "clients":        self.stream_clients,
                "voice_active":   self.voice_active,
                "last_transcript": self.last_transcript,
                "voice_history":  self.voice_history[-20:],
                "alerts":         self.alerts[-10:],
                "ts":             time.time(),
            }

    def record_frame(self):
        """Enregistre une frame pour calcul FPS observé."""
        with self._lock:
            now = time.monotonic()
            self._frame_times.append(now)
            # Garder seulement les frames de la dernière seconde
            cutoff = now - 1.0
            self._frame_times = [t for t in self._frame_times if t > cutoff]
            self.stream_fps_observed = round(len(self._frame_times), 1)

    def add_alert(self, level: str, message: str):
        with self._lock:
            self.alerts.append({
                "level":   level,   # "info" | "warning" | "danger"
                "message": message,
                "ts":      datetime.now().strftime("%H:%M:%S"),
            })
            # Garder max 50 alertes
            if len(self.alerts) > 50:
                self.alerts = self.alerts[-50:]

    def add_transcript(self, text: str, source: str = "webspeech"):
        with self._lock:
            entry = {
                "text":   text,
                "source": source,  # "webspeech" | "whisper"
                "ts":     datetime.now().strftime("%H:%M:%S"),
            }
            self.voice_history.append(entry)
            self.last_transcript = text


state = StreamingState()

# =============================================================================
# WHISPER — CHARGEMENT LAZY
# =============================================================================

class WhisperEngine:
    """
    Moteur Whisper chargé en lazy (uniquement quand une transcription est demandée).
    Le modèle est gardé en mémoire après le premier chargement.
    """

    def __init__(self):
        self._model = None
        self._lock = threading.Lock()
        self._loading = False

    def _load(self):
        if not HAS_WHISPER:
            logger.warning("whisper non installé — pip install openai-whisper")
            return False
        if self._model is not None:
            return True
        logger.info("Chargement modèle Whisper '%s'...", CONFIG["whisper_model"])
        try:
            self._model = _whisper.load_model(CONFIG["whisper_model"])
            logger.info("Whisper prêt")
            return True
        except Exception as e:
            logger.error("Erreur chargement Whisper : %s", e)
            return False

    def transcribe(self, audio_path: str) -> dict:
        """
        Transcrit un fichier audio WAV/MP3/OGG.
        Retourne {"text": str, "language": str, "duration_s": float}
        """
        with self._lock:
            if not self._load():
                return {"error": "whisper_unavailable"}
            t_start = time.monotonic()
            try:
                result = self._model.transcribe(
                    audio_path,
                    language=CONFIG["whisper_language"],
                    fp16=False,   # fp16 non supporté sur Pi CPU
                )
                duration = round(time.monotonic() - t_start, 2)
                text = result.get("text", "").strip()
                logger.info("Whisper : '%s' (%.1fs)", text[:60], duration)
                return {
                    "text":       text,
                    "language":   result.get("language", CONFIG["whisper_language"]),
                    "duration_s": duration,
                }
            except Exception as e:
                logger.error("Erreur transcription Whisper : %s", e)
                return {"error": str(e)}


whisper_engine = WhisperEngine()

# =============================================================================
# MQTT HANDLER
# =============================================================================

class MQTTHandler:
    """
    Écoute les topics MQTT du système SHOS et alimente le state.
    Détecte les dépassements de seuils et génère des alertes.
    """

    # Seuils d'alerte par défaut (overridable par config.json)
    THRESHOLDS = {
        "gas":      {"warning": 200, "danger": 400},
        "distance": {"warning": 50,  "danger": 20},   # cm
        "temp":     {"warning": 70,  "danger": 80},   # °C CPU
        "sound":    {"warning": 150, "danger": 180},
    }

    def __init__(self):
        if not HAS_MQTT:
            logger.warning("paho-mqtt absent — MQTT désactivé")
            self.client = None
            return

        if _MQTT_V2:
            self.client = mqtt.Client(
                callback_api_version=CallbackAPIVersion.VERSION2,
                client_id="shos_streaming_module",
            )
        else:
            self.client = mqtt.Client(client_id="shos_streaming_module")

        self.client.on_connect    = self._on_connect
        self.client.on_message    = self._on_message
        self.client.on_disconnect = self._on_disconnect

        self._connected = False

    def start(self):
        if self.client is None:
            return
        try:
            self.client.connect(
                CONFIG["mqtt_broker"],
                CONFIG["mqtt_port"],
                keepalive=60,
            )
            self.client.loop_start()
            logger.info("MQTT connecting → %s:%d",
                        CONFIG["mqtt_broker"], CONFIG["mqtt_port"])
        except Exception as e:
            logger.error("MQTT connexion échouée : %s", e)

    def stop(self):
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()

    def publish(self, topic: str, payload: dict, qos: int = 0):
        if self.client and self._connected:
            try:
                self.client.publish(topic, json.dumps(payload), qos=qos)
            except Exception as e:
                logger.warning("MQTT publish error: %s", e)

    def _on_connect(self, client, userdata, flags, rc, *args):
        rc_code = rc if isinstance(rc, int) else rc.value
        if rc_code == 0:
            self._connected = True
            # Abonnement à tous les topics du système
            for topic in CONFIG["topics"].values():
                client.subscribe(topic, qos=0)
            logger.info("MQTT connecté — abonné à %d topics",
                        len(CONFIG["topics"]))
            # Publier statut du module
            self.publish(CONFIG["publish"]["stream_status"], {
                "status": "online",
                "source": state.active_source,
                "ts":     time.time(),
            })
        else:
            logger.error("MQTT connexion refusée (rc=%d)", rc_code)

    def _on_disconnect(self, client, userdata, *args):
        self._connected = False
        logger.warning("MQTT déconnecté")

    def _on_message(self, client, userdata, msg):
        topic = msg.topic
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return

        topics = CONFIG["topics"]

        if topic == topics["vision"]:
            state.update("vision", payload)
            # Détecter mains ou pose → pas d'alerte, juste mise à jour

        elif topic == topics["sensors_norm"]:
            state.update("sensors", payload)
            self._check_sensor_alerts(payload)

        elif topic == topics["sensors_raw"]:
            state.update("sensors_raw", payload)

        elif topic == topics["telemetry"]:
            state.update("telemetry", payload)
            # Alerte température CPU
            temp = payload.get("temp")
            if temp and temp > self.THRESHOLDS["temp"]["danger"]:
                state.add_alert("danger", f"Température CPU critique : {temp}°C")
            elif temp and temp > self.THRESHOLDS["temp"]["warning"]:
                state.add_alert("warning", f"Température CPU élevée : {temp}°C")

        elif topic == topics["status"]:
            state.update("system_status", str(payload) if not isinstance(payload, str) else payload)

    def _check_sensor_alerts(self, sensors: dict):
        """Vérifie les seuils et génère des alertes si dépassés."""
        gas = sensors.get("gas")
        if gas is not None:
            if gas > self.THRESHOLDS["gas"]["danger"]:
                state.add_alert("danger", f"Niveau de gaz dangereux : {gas} ppm")
            elif gas > self.THRESHOLDS["gas"]["warning"]:
                state.add_alert("warning", f"Niveau de gaz élevé : {gas} ppm")

        dist = sensors.get("distance")
        if dist and dist > 0:  # -1 = invalide
            if dist < self.THRESHOLDS["distance"]["danger"]:
                state.add_alert("danger", f"Obstacle très proche : {dist} cm")
            elif dist < self.THRESHOLDS["distance"]["warning"]:
                state.add_alert("warning", f"Obstacle détecté : {dist} cm")

        flame = sensors.get("flame")
        if flame == 1:
            state.add_alert("danger", "FLAMME DÉTECTÉE")


mqtt_handler = MQTTHandler()

# =============================================================================
# SYSTÈME MÉTRIQUES (psutil — thread séparé)
# =============================================================================

class SystemMetricsCollector:
    """Collecte CPU/RAM/temp en arrière-plan et alimente le state."""

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if not HAS_PSUTIL:
            logger.warning("psutil absent — métriques système désactivées")
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, name="MetricsCollector", daemon=True
        )
        self._thread.start()
        # Premier appel cpu_percent retourne 0 — on l'ignore
        psutil.cpu_percent(interval=None)
        logger.info("MetricsCollector démarré")

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            t_start = time.monotonic()

            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent
            temp = self._get_temp()

            state.update("cpu_percent", cpu)
            state.update("ram_percent", ram)
            state.update("cpu_temp", temp)

            elapsed = time.monotonic() - t_start
            time.sleep(max(0, self.interval - elapsed))

    @staticmethod
    def _get_temp() -> Optional[float]:
        try:
            with open("/sys/class/thermal/thermal_zone0/temp") as f:
                return round(int(f.read().strip()) / 1000, 1)
        except Exception:
            pass
        try:
            t = psutil.sensors_temperatures()
            for k in ["cpu_thermal", "coretemp", "acpitz"]:
                if k in t and t[k]:
                    return round(t[k][0].current, 1)
        except Exception:
            pass
        return None


metrics_collector = SystemMetricsCollector(CONFIG["metrics_interval_s"])

# =============================================================================
# PROXY STREAM MJPEG
# =============================================================================

def _get_stream_url() -> str:
    """Retourne l'URL du flux selon la source active."""
    if state.active_source == "esp32":
        url = CONFIG.get("esp32_url", "")
        if url:
            return url
        logger.warning("ESP32 URL non configurée — fallback Picamera")
    return CONFIG["picamera_url"]


def generate_video_stream():
    """
    Générateur MJPEG — proxy vers la source active (Picamera ou ESP32).
    Utilise requests streaming pour un relay fiable de flux continu.
    # Chaque connexion incrémente/décrémente le compteur de clients.
    """
    if not HAS_REQUESTS:
        logger.error("requests non installé — pip install requests")
        yield b""
        return

    state.stream_clients += 1
    url = _get_stream_url()
    logger.info("Stream client connecté — source=%s url=%s clients=%d",
                state.active_source, url, state.stream_clients)

    try:
        with _requests.get(url, stream=True, timeout=10) as r:
            for chunk in r.iter_content(chunk_size=4096):
                if chunk:
                    state.record_frame()
                    yield chunk
    except Exception as e:
        logger.warning("Stream interrompu (%s) : %s",
                       state.active_source, e)
    finally:
        state.stream_clients = max(0, state.stream_clients - 1)
        logger.info("Stream client déconnecté — clients=%d", state.stream_clients)

# =============================================================================
# COMMANDES VOCALES — PARSING
# =============================================================================

VOICE_COMMANDS = {
    # Commandes source vidéo
    "caméra principale":  lambda: _switch_source("picamera"),
    "camera principale":  lambda: _switch_source("picamera"),
    "esp32":              lambda: _switch_source("esp32"),
    "caméra esp":         lambda: _switch_source("esp32"),

    # Commandes alertes
    "effacer alertes":    lambda: state.update("alerts", []),
    "supprimer alertes":  lambda: state.update("alerts", []),

    # Commandes système
    "statut système":     _cmd_status,
    "status systeme":     _cmd_status,
}


def _switch_source(source: str):
    state.update("active_source", source)
    logger.info("Source vidéo changée → %s", source)
    mqtt_handler.publish(CONFIG["publish"]["stream_status"], {
        "status": "source_changed", "source": source, "ts": time.time()
    })


def _cmd_status():
    snap = state.get_snapshot()
    logger.info("Statut : CPU=%.1f%% RAM=%.1f%% Temp=%s°C FPS=%.1f",
                snap["cpu"], snap["ram"], snap["temp"], snap["fps"])


def parse_voice_command(text: str) -> Optional[str]:
    """
    Cherche une commande connue dans le texte transcrit.
    Retourne la clé de commande trouvée ou None.
    """
    text_lower = text.lower().strip()
    for keyword, action in VOICE_COMMANDS.items():
        if keyword in text_lower:
            try:
                action()
                logger.info("Commande vocale exécutée : '%s'", keyword)
            except Exception as e:
                logger.error("Erreur commande vocale '%s': %s", keyword, e)
            return keyword
    return None

# =============================================================================
# FLASK BLUEPRINT / APP
# =============================================================================

streaming_bp = Blueprint(
    "video_streaming",
    __name__,
    template_folder=str(Path(__file__).parent),
    static_folder=str(Path(__file__).parent / "static"),
)


@streaming_bp.route("/")
def index():
    """Interface principale du module streaming."""
    return render_template("interface.html")


@streaming_bp.route("/video")
def video():
    """
    Flux MJPEG — proxy vers la source active.
    Le navigateur ouvre <img src='/streaming/video'> pour recevoir le flux.
    """
    return Response(
        stream_with_context(generate_video_stream()),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@streaming_bp.route("/api/source", methods=["GET"])
def get_source():
    """Retourne la source vidéo active et les sources disponibles."""
    return jsonify({
        "active":    state.active_source,
        "picamera":  bool(CONFIG.get("picamera_url")),
        "esp32":     bool(CONFIG.get("esp32_url")),
        "esp32_url": CONFIG.get("esp32_url", ""),
    })


@streaming_bp.route("/api/source", methods=["POST"])
def set_source():
    """
    Change la source vidéo active.
    Body JSON : {"source": "picamera" | "esp32"}
    # ASVS V5.1 : validation de la valeur avant application
    """
    data = request.get_json(silent=True) or {}
    source = data.get("source", "")
    if source not in ("picamera", "esp32"):
        return jsonify({"error": "Source invalide — valeurs: picamera, esp32"}), 400
    if source == "esp32" and not CONFIG.get("esp32_url"):
        return jsonify({"error": "ESP32 URL non configurée dans config.json"}), 400
    _switch_source(source)
    return jsonify({"active": state.active_source, "ok": True})


@streaming_bp.route("/api/metrics")
def get_metrics():
    """Snapshot complet de toutes les métriques — polling fallback si WebSocket indispo."""
    return jsonify(state.get_snapshot())


@streaming_bp.route("/api/alerts", methods=["GET"])
def get_alerts():
    """Retourne les alertes actives."""
    return jsonify({"alerts": state.alerts})


@streaming_bp.route("/api/alerts", methods=["DELETE"])
def clear_alerts():
    """Efface toutes les alertes."""
    state.update("alerts", [])
    return jsonify({"ok": True})


@streaming_bp.route("/api/voice/transcript", methods=["POST"])
def voice_transcript():
    """
    Reçoit une transcription Web Speech API depuis le navigateur.
    Body JSON : {"text": "...", "confidence": 0.95}
    Publie sur MQTT + parse les commandes.
    # ASVS V5.1 : sanitisation du texte entrant
    """
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()[:500]  # max 500 chars
    confidence = float(data.get("confidence", 0.0))

    if not text:
        return jsonify({"error": "Texte vide"}), 400

    state.add_transcript(text, source="webspeech")
    logger.info("Web Speech → '%s' (conf=%.2f)", text[:60], confidence)

    # Publier sur MQTT
    mqtt_handler.publish(CONFIG["publish"]["voice_text"], {
        "text":       text,
        "confidence": confidence,
        "source":     "webspeech",
        "ts":         time.time(),
    })

    # Parser commandes vocales
    command = parse_voice_command(text)
    if command:
        mqtt_handler.publish(CONFIG["publish"]["voice_command"], {
            "command": command,
            "text":    text,
            "ts":      time.time(),
        })

    return jsonify({
        "ok":      True,
        "command": command,
        "text":    text,
    })


@streaming_bp.route("/api/voice/whisper", methods=["POST"])
def voice_whisper():
    """
    Reçoit un fichier audio et le transcrit avec Whisper.
    Content-Type: multipart/form-data — champ 'audio'
    # ASVS V12.1 : validation type fichier + taille
    """
    if "audio" not in request.files:
        return jsonify({"error": "Champ 'audio' manquant"}), 400

    audio_file = request.files["audio"]

    # Validation : taille max 10MB, extension audio uniquement
    audio_file.seek(0, 2)
    size = audio_file.tell()
    audio_file.seek(0)
    if size > 10 * 1024 * 1024:
        return jsonify({"error": "Fichier trop volumineux (max 10MB)"}), 413

    allowed_ext = {".wav", ".mp3", ".ogg", ".webm", ".m4a", ".flac"}
    ext = Path(audio_file.filename or "audio.wav").suffix.lower()
    if ext not in allowed_ext:
        return jsonify({"error": f"Format non supporté. Acceptés: {allowed_ext}"}), 400

    # Sauvegarder temporairement
    tmp_dir = Path("/tmp/shos_audio")
    tmp_dir.mkdir(exist_ok=True)
    tmp_path = tmp_dir / f"audio_{int(time.time())}{ext}"

    try:
        audio_file.save(str(tmp_path))
        result = whisper_engine.transcribe(str(tmp_path))
    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass

    if "error" in result:
        return jsonify(result), 500

    text = result.get("text", "")
    state.add_transcript(text, source="whisper")

    # Publier + parser commandes
    mqtt_handler.publish(CONFIG["publish"]["voice_text"], {
        "text":     text,
        "source":   "whisper",
        "duration": result.get("duration_s"),
        "ts":       time.time(),
    })
    command = parse_voice_command(text)
    if command:
        mqtt_handler.publish(CONFIG["publish"]["voice_command"], {
            "command": command,
            "text":    text,
            "ts":      time.time(),
        })

    return jsonify({**result, "command": command})


@streaming_bp.route("/api/config", methods=["GET"])
def get_config():
    """Retourne la config publique du module (sans secrets)."""
    return jsonify({
        "esp32_url":       CONFIG.get("esp32_url", ""),
        "picamera_url":    CONFIG.get("picamera_url", ""),
        "whisper_model":   CONFIG.get("whisper_model", "base"),
        "whisper_language": CONFIG.get("whisper_language", "fr"),
        "mqtt_broker":     CONFIG.get("mqtt_broker", "localhost"),
        "has_whisper":     HAS_WHISPER,
        "has_mqtt":        HAS_MQTT,
        "has_psutil":      HAS_PSUTIL,
    })


@streaming_bp.route("/api/config", methods=["POST"])
def update_config():
    """
    Met à jour la config runtime (esp32_url, whisper_language...).
    # ASVS V5.1 : seules les clés whitelistées sont acceptées
    """
    data = request.get_json(silent=True) or {}
    ALLOWED_KEYS = {"esp32_url", "whisper_language", "whisper_model"}

    updated = {}
    for k, v in data.items():
        if k in ALLOWED_KEYS:
            CONFIG[k] = str(v)[:200]
            updated[k] = CONFIG[k]

    # Sauvegarder dans config.json
    try:
        with open(_cfg_path, "w") as f:
            json.dump(CONFIG, f, indent=2)
    except Exception as e:
        logger.warning("Impossible de sauvegarder config.json : %s", e)

    return jsonify({"ok": True, "updated": updated})

# =============================================================================
# SOCKETIO — PUSH TEMPS RÉEL
# =============================================================================

def setup_socketio(socketio: SocketIO):
    """
    Configure les événements SocketIO pour le push temps réel vers le navigateur.
    Appelé après création du socketio (via nova.py ou autonome).
    """

    @socketio.on("connect", namespace="/streaming")
    def on_connect():
        logger.info("SocketIO client connecté sur /streaming")
        emit("config", state.get_snapshot())

    @socketio.on("disconnect", namespace="/streaming")
    def on_disconnect():
        logger.info("SocketIO client déconnecté de /streaming")

    @socketio.on("switch_source", namespace="/streaming")
    def on_switch_source(data):
        source = str(data.get("source", "picamera"))
        if source in ("picamera", "esp32"):
            _switch_source(source)
            socketio.emit("source_changed", {"source": source}, namespace="/streaming")

    @socketio.on("voice_text", namespace="/streaming")
    def on_voice_text(data):
        """Reçoit la transcription Web Speech API directement via WebSocket."""
        text = str(data.get("text", "")).strip()[:500]
        if text:
            state.add_transcript(text, "webspeech")
            command = parse_voice_command(text)
            mqtt_handler.publish(CONFIG["publish"]["voice_text"], {
                "text": text, "source": "webspeech", "ts": time.time()
            })
            socketio.emit("voice_update", {
                "text": text, "command": command
            }, namespace="/streaming")

    @socketio.on("clear_alerts", namespace="/streaming")
    def on_clear_alerts(_):
        state.update("alerts", [])
        socketio.emit("alerts_cleared", {}, namespace="/streaming")


def start_metrics_push(socketio: SocketIO, interval: float = 1.0):
    """
    Thread qui pousse les métriques via SocketIO toutes les `interval` secondes.
    Évite le polling HTTP côté client.
    """
    def _push_loop():
        while True:
            time.sleep(interval)
            try:
                socketio.emit(
                    "metrics_update",
                    state.get_snapshot(),
                    namespace="/streaming",
                )
            except Exception as e:
                logger.debug("SocketIO push error: %s", e)

    t = threading.Thread(target=_push_loop, name="MetricsPush", daemon=True)
    t.start()
    logger.info("MetricsPush démarré (intervalle=%.1fs)", interval)

# =============================================================================
# POINT D'ENTRÉE AUTONOME
# =============================================================================

def create_app(socketio_instance=None) -> tuple:
    """
    Crée l'app Flask + SocketIO pour usage autonome ou intégration nova.py.
    Retourne (app, socketio).
    """
    app = Flask(__name__, template_folder=str(Path(__file__).parent))
    app.config["SECRET_KEY"] = os.urandom(24).hex()
    app.register_blueprint(streaming_bp, url_prefix="/streaming")

    sio = socketio_instance or SocketIO(
        app,
        cors_allowed_origins="*",
        async_mode="threading",
    )

    setup_socketio(sio)
    return app, sio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SHOS Video Streaming Module — autonome"
    )
    parser.add_argument("--port",   type=int, default=CONFIG["port"])
    parser.add_argument("--host",   default=CONFIG["host"])
    parser.add_argument("--esp32",  default="", help="URL stream ESP32-CAM")
    parser.add_argument("--broker", default=CONFIG["mqtt_broker"])
    args = parser.parse_args()

    if args.esp32:
        CONFIG["esp32_url"] = args.esp32
    CONFIG["mqtt_broker"] = args.broker

    logger.info("╔══ SHOS Video Streaming Module ══╗")
    logger.info("  Port    : %d", args.port)
    logger.info("  Broker  : %s:%d", CONFIG["mqtt_broker"], CONFIG["mqtt_port"])
    logger.info("  Whisper : %s", "✓" if HAS_WHISPER else "✗ (pip install openai-whisper)")
    logger.info("  ESP32   : %s", CONFIG["esp32_url"] or "non configuré")

    # Démarrage des services
    mqtt_handler.start()
    metrics_collector.start()

    app, sio = create_app()
    start_metrics_push(sio)

    logger.info("Interface : http://%s:%d/streaming/", args.host, args.port)
    sio.run(app, host=args.host, port=args.port, debug=False, use_reloader=False)
