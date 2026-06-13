#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
 S.H.O.S · DANGER MONITOR V3
===============================================================================
 Surveillance temps réel : distance (ultrason) + gaz (MQ2/MQ135)
 Publie en continu pour garder l'interface toujours à jour.

 Topics MQTT écoutés :
   shos/sensors/normalized          → données capteurs (dist, gas, vibration)
   shos/sensors/raw                 → fallback
   shos/plugins/danger_monitor/cmd  → commandes (start, stop, set_threshold)

 Topics MQTT publiés :
   shos/plugins/danger_monitor/data → état complet (continu)
   shos/alert/critical              → alerte critique (distance ou gaz)
   shos/plugins/voice_assistant/cmd → alerte vocale
   shos/arduino/cmd                 → feedback physique
===============================================================================
"""

import json
import time
import logging
import threading
from pathlib import Path

import paho.mqtt.client as mqtt

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BROKER = "localhost"
PORT   = 1883

T_SENSORS_NORM = "shos/sensors/normalized"
T_SENSORS_RAW  = "shos/sensors/raw"
T_CMD          = "shos/plugins/danger_monitor/cmd"
T_DATA         = "shos/plugins/danger_monitor/data"
T_ALERT        = "shos/alert/critical"
T_VOICE        = "shos/plugins/voice_assistant/cmd"
T_ARDUINO      = "shos/arduino/cmd"

# Seuils par défaut
DIST_CRITICAL  = 30     # cm — CRITIQUE
DIST_WARNING   = 60     # cm — AVERTISSEMENT
GAS_DANGER     = 400    # ppm — DANGER
GAS_WARNING    = 200    # ppm — AVERTISSEMENT
VIB_THRESHOLD  = 0.7   # 0-1 — VIBRATION

# Anti-spam alertes vocales (secondes)
VOICE_COOLDOWN = 5.0
# Intervalle de publication continue (secondes)
PUBLISH_INTERVAL = 0.5

# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] DangerMonitor - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("DangerMonitor")


# =============================================================================
# MODULE DANGER MONITOR V3
# =============================================================================
class DangerMonitorV3:

    def __init__(self):
        logger.info("Initialisation Danger Monitor V3...")

        self.active = True  # Actif par défaut

        # Seuils (modifiables via commande)
        self.dist_critical = DIST_CRITICAL
        self.dist_warning  = DIST_WARNING
        self.gas_danger    = GAS_DANGER
        self.gas_warning   = GAS_WARNING
        self.vib_threshold = VIB_THRESHOLD

        # Dernières valeurs capteurs
        self.sensors = {
            "distance":  -1,
            "gas":        0,
            "vibration":  0,
            "temp":       None,
            "humidity":   None,
            "last_update": 0,
        }

        # Historique alertes (max 20)
        self.alert_history = []
        self.total_alerts  = 0

        # Anti-spam
        self.last_voice_alert  = 0.0
        self.last_alert_ts     = {}  # par type d'alerte

        self._lock = threading.Lock()

        self._setup_mqtt()

        # Thread de publication continue
        self._pub_thread = threading.Thread(
            target=self._publish_loop, daemon=True, name="DM_PublishLoop"
        )
        self._pub_thread.start()

    # ── MQTT ────────────────────────────────────────────────────────────────
    def _setup_mqtt(self):
        try:
            try:
                from paho.mqtt.enums import CallbackAPIVersion
                self.client = mqtt.Client(CallbackAPIVersion.VERSION2, "PLUGIN_DANGER_V3")
            except ImportError:
                self.client = mqtt.Client("PLUGIN_DANGER_V3")
            self.client.on_connect = self._on_connect
            self.client.on_message = self._on_message
            self.client.connect(BROKER, PORT, 60)
            self.client.loop_start()
        except Exception as e:
            logger.error("MQTT : %s", e)

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        logger.info("MQTT connecté (rc=%s)", rc)
        client.subscribe([
            (T_SENSORS_NORM, 1),
            (T_SENSORS_RAW,  1),
            (T_CMD,          1),
        ])
        # Publication du statut initial
        self._publish_status()

    def _on_message(self, client, userdata, msg):
        topic = msg.topic
        try:
            payload = json.loads(msg.payload.decode())
        except Exception:
            return

        if topic in (T_SENSORS_NORM, T_SENSORS_RAW):
            self._handle_sensors(payload)
        elif topic == T_CMD:
            self._handle_cmd(payload)

    # ── Traitement capteurs ─────────────────────────────────────────────────
    def _handle_sensors(self, data: dict):
        with self._lock:
            # Distance (plusieurs noms possibles selon ESP32/Arduino)
            dist = data.get("distance",
                   data.get("dist",
                   data.get("ultrasound",
                   self.sensors["distance"])))
            self.sensors["distance"] = float(dist) if dist is not None else -1

            # Gaz
            gas = data.get("gas",
                  data.get("mq2",
                  data.get("smoke",
                  self.sensors["gas"])))
            self.sensors["gas"] = float(gas) if gas is not None else 0

            # Vibration
            vib = data.get("vibration",
                  data.get("vib",
                  self.sensors["vibration"]))
            self.sensors["vibration"] = float(vib) if vib is not None else 0

            # Température / Humidité (optionnel)
            if "temperature" in data or "temp" in data:
                self.sensors["temp"] = data.get("temperature", data.get("temp"))
            if "humidity" in data or "hum" in data:
                self.sensors["humidity"] = data.get("humidity", data.get("hum"))

            self.sensors["last_update"] = time.time()

        if self.active:
            self._analyze_risks()

    # ── Analyse des risques ─────────────────────────────────────────────────
    def _analyze_risks(self):
        with self._lock:
            dist = self.sensors["distance"]
            gas  = self.sensors["gas"]
            vib  = self.sensors["vibration"]

        alerts = []
        now = time.time()

        # ── Distance ──────────────────────────────────────────────────────
        if 0 <= dist < self.dist_critical:
            alerts.append({
                "type":    "OBSTACLE",
                "value":   round(dist, 1),
                "unit":    "cm",
                "level":   "CRITICAL",
                "msg":     f"Obstacle à {round(dist, 1)} cm",
            })
            self._voice_alert(f"Attention, obstacle à {int(dist)} centimètres")
        elif 0 <= dist < self.dist_warning:
            alerts.append({
                "type":    "OBSTACLE",
                "value":   round(dist, 1),
                "unit":    "cm",
                "level":   "WARNING",
                "msg":     f"Obstacle proche : {round(dist, 1)} cm",
            })

        # ── Gaz ───────────────────────────────────────────────────────────
        if gas > self.gas_danger:
            alerts.append({
                "type":    "GAS",
                "value":   round(gas),
                "unit":    "ppm",
                "level":   "DANGER",
                "msg":     f"Concentration gaz anormale : {round(gas)} ppm",
            })
            self._voice_alert("Alerte. Concentration de gaz anormale")
        elif gas > self.gas_warning:
            alerts.append({
                "type":    "GAS",
                "value":   round(gas),
                "unit":    "ppm",
                "level":   "WARNING",
                "msg":     f"Gaz élevé : {round(gas)} ppm",
            })

        # ── Vibration ─────────────────────────────────────────────────────
        if vib > self.vib_threshold:
            alerts.append({
                "type":    "VIBRATION",
                "value":   round(vib, 2),
                "unit":    "",
                "level":   "WARNING",
                "msg":     f"Vibration détectée : {round(vib, 2)}",
            })

        # ── Traitement alertes ────────────────────────────────────────────
        if alerts:
            self._process_alerts(alerts, now)

    def _process_alerts(self, alerts: list, now: float):
        critical = [a for a in alerts if a["level"] == "CRITICAL"]

        # Historique
        with self._lock:
            for a in alerts:
                self.alert_history.insert(0, {**a, "ts": now})
                self.total_alerts += 1
            if len(self.alert_history) > 20:
                self.alert_history = self.alert_history[:20]

        # Publication alerte critique
        if critical:
            self.client.publish(T_ALERT, json.dumps({
                "source":  "danger_monitor",
                "alerts":  critical,
                "ts":      now,
            }), qos=1)

        # Feedback Arduino
        level = "high" if critical else "medium"
        try:
            self.client.publish(T_ARDUINO, json.dumps({
                "action": "alert", "level": level
            }))
        except Exception:
            pass

        logger.warning("⚠ %d alerte(s) : %s",
                       len(alerts), [a["type"] for a in alerts])

    # ── Publication continue ────────────────────────────────────────────────
    def _publish_loop(self):
        """Publie l'état complet toutes les PUBLISH_INTERVAL secondes."""
        logger.info("Boucle publication démarrée (%.1fs)", PUBLISH_INTERVAL)
        while True:
            try:
                self._publish_state()
            except Exception as e:
                logger.debug("publish_loop : %s", e)
            time.sleep(PUBLISH_INTERVAL)

    def _publish_state(self):
        with self._lock:
            sensors = dict(self.sensors)
            history = list(self.alert_history[:5])
            total   = self.total_alerts

        dist = sensors["distance"]
        gas  = sensors["gas"]
        vib  = sensors["vibration"]

        # Calcul niveaux
        if 0 <= dist < self.dist_critical:
            dist_level = "critical"
        elif 0 <= dist < self.dist_warning:
            dist_level = "warning"
        elif dist >= 0:
            dist_level = "ok"
        else:
            dist_level = "unknown"

        if gas > self.gas_danger:
            gas_level = "danger"
        elif gas > self.gas_warning:
            gas_level = "warning"
        else:
            gas_level = "ok"

        payload = {
            "plugin":       "danger_monitor",
            "type":         "state",
            "active":       self.active,
            "sensors": {
                "distance":     round(dist, 1) if dist >= 0 else -1,
                "gas":          round(gas),
                "vibration":    round(vib, 2),
                "temp":         sensors.get("temp"),
                "humidity":     sensors.get("humidity"),
                "last_update":  sensors["last_update"],
            },
            "levels": {
                "distance":  dist_level,
                "gas":       gas_level,
                "vibration": "warning" if vib > self.vib_threshold else "ok",
            },
            "thresholds": {
                "dist_critical": self.dist_critical,
                "dist_warning":  self.dist_warning,
                "gas_danger":    self.gas_danger,
                "gas_warning":   self.gas_warning,
            },
            "alert_history": history,
            "total_alerts":  total,
            "ts":            time.time(),
        }
        try:
            self.client.publish(T_DATA, json.dumps(payload), qos=0)
        except Exception as e:
            logger.debug("publish_state : %s", e)

    def _publish_status(self):
        self.client.publish(T_DATA, json.dumps({
            "plugin": "danger_monitor",
            "type":   "status",
            "active": self.active,
            "ts":     time.time(),
        }), qos=0)

    # ── Alertes vocales ─────────────────────────────────────────────────────
    def _voice_alert(self, message: str):
        now = time.time()
        if now - self.last_voice_alert > VOICE_COOLDOWN:
            try:
                self.client.publish(T_VOICE, json.dumps({
                    "cmd":  "speak",
                    "text": message,
                }))
                self.last_voice_alert = now
            except Exception:
                pass

    # ── Commandes ────────────────────────────────────────────────────────────
    def _handle_cmd(self, payload: dict):
        cmd = str(payload.get("cmd", "")).lower()

        if cmd in ("start", "activate"):
            self.active = True
            logger.info("▶ Surveillance activée")
            self._publish_status()

        elif cmd in ("stop", "pause"):
            self.active = False
            logger.info("⏸ Surveillance en pause")
            self._publish_status()

        elif cmd == "toggle":
            self.active = not self.active
            logger.info("Toggle → %s", "actif" if self.active else "pause")
            self._publish_status()

        elif cmd == "clear":
            with self._lock:
                self.alert_history.clear()
                self.total_alerts = 0
            logger.info("Historique effacé")

        elif cmd == "set_threshold":
            # ex: { cmd: "set_threshold", dist_critical: 25, gas_danger: 350 }
            if "dist_critical" in payload:
                self.dist_critical = float(payload["dist_critical"])
            if "dist_warning"  in payload:
                self.dist_warning  = float(payload["dist_warning"])
            if "gas_danger"    in payload:
                self.gas_danger    = float(payload["gas_danger"])
            if "gas_warning"   in payload:
                self.gas_warning   = float(payload["gas_warning"])
            logger.info("Seuils mis à jour : dist_crit=%s gas_danger=%s",
                        self.dist_critical, self.gas_danger)

        elif cmd == "get_status":
            self._publish_status()

    # ── Run ──────────────────────────────────────────────────────────────────
    def run(self):
        logger.info("Danger Monitor V3 démarré")
        logger.info("Seuils : dist crit=%dcm warn=%dcm | gaz danger=%d warn=%d",
                    self.dist_critical, self.dist_warning,
                    self.gas_danger, self.gas_warning)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Arrêt")
        finally:
            self.active = False
            try:
                self.client.publish(T_DATA, json.dumps({
                    "plugin": "danger_monitor",
                    "type":   "status",
                    "active": False,
                    "ts":     time.time(),
                }))
                self.client.loop_stop()
                self.client.disconnect()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    DangerMonitorV3().run()