#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S.H.O.S · MODULE VISION OBJET
  Source  : MQTT shos/camera/raw
  Modèle  : YOLO11n (ultralytics) - direct sur Pi
  Serveur : Flask local port 5052 (Flux vidéo autonome)
  Output  : shos/plugins/vision_objet/data (JSON détections)
            shos/camera/processed (frame annotée JPEG bytes)
"""

import os, sys, json, time, logging, subprocess
from pathlib import Path
from threading import Thread, Event, Lock
from datetime import datetime

import cv2
import numpy as np
from paho.mqtt import client as mqtt_client

# ─── LOGGING ───────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("VISION_OBJET")

# ─── CONFIG ────────────────────────────────────────────
BROKER          = "localhost"
PORT            = 1883
TOPIC_IN        = "shos/camera/vision"  # vision_service publie ici les metas JSON
TOPIC_OUT_DATA  = "shos/plugins/vision_objet/data"
TOPIC_OUT_FRAME = "shos/camera/processed"
TOPIC_CMD       = "shos/plugins/vision_objet/cmd"
TOPIC_SNAP_REQ  = "shos/camera/snapshot_request"

MODULE_DIR  = Path(__file__).parent
MODELS_DIR  = MODULE_DIR.parent.parent.parent / "models"
MODEL_PATH  = MODELS_DIR / "yolo11n.pt"
SNAP_DIR    = MODULE_DIR.parent.parent.parent / "data" / "snapshots"
DEPS_FLAG   = MODULE_DIR / ".deps_installed"

CONF_THRESHOLD  = 0.45
FRAME_SKIP      = 2       # traiter 1 frame sur N (charge Pi)
JPEG_QUALITY    = 50
MAX_HISTORY     = 100

# Flask 5052 supprimé - vision_service (port 5051) sert déjà le flux annoté
latest_frame = None
frame_lock = Lock()


# ═══════════════════════════════════════════════════════
# VÉRIFICATION & INSTALLATION DES DÉPENDANCES
# ═══════════════════════════════════════════════════════
def ensure_deps():
    if DEPS_FLAG.exists():
        return
    log.info("📦 Première installation des dépendances...")
    deps = ["ultralytics", "opencv-python", "paho-mqtt", "numpy", "Pillow", "flask"]
    for dep in deps:
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", dep, "-q"],
                check=True
            )
            log.info("  ✅ %s", dep)
        except Exception as e:
            log.warning("  ⚠ %s: %s", dep, e)

    # Télécharger le modèle si absent
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        log.info("📥 Téléchargement YOLO11n...")
        try:
            from ultralytics import YOLO
            YOLO("yolo11n.pt")
            import shutil
            src = Path.home() / ".config" / "Ultralytics" / "yolo11n.pt"
            if not src.exists():
                src = Path("yolo11n.pt")
            if src.exists():
                shutil.copy(src, MODEL_PATH)
        except Exception as e:
            log.warning("⚠ Téléchargement modèle: %s", e)

    DEPS_FLAG.write_text(datetime.now().isoformat())
    log.info("✅ Dépendances prêtes")


# ═══════════════════════════════════════════════════════
# CHARGEUR YOLO
# ═══════════════════════════════════════════════════════
class YOLODetector:
    def __init__(self):
        self.model   = None
        self.ready   = False
        self._load()

    def _load(self):
        try:
            from ultralytics import YOLO
            path = str(MODEL_PATH) if MODEL_PATH.exists() else "yolo11n.pt"
            log.info("🔍 Chargement YOLO: %s", path)
            self.model = YOLO(path)
            self.ready = True
            log.info("✅ YOLO11n prêt")
        except Exception as e:
            log.error("❌ YOLO: %s", e)
            self.ready = False

    def detect(self, frame: np.ndarray) -> tuple:
        """
        Retourne (frame_annotée, liste_de_détections)
        détection = {"label": str, "conf": float, "box": [x1,y1,x2,y2]}
        """
        if not self.ready or self.model is None:
            return frame, []

        try:
            results = self.model.predict(
                frame, conf=CONF_THRESHOLD, verbose=False, stream=False
            )
            detections = []
            annotated  = frame.copy()

            for r in results:
                for box in (r.boxes or []):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf  = float(box.conf[0])
                    label = self.model.names[int(box.cls[0])]

                    detections.append({
                        "label": label,
                        "conf":  round(conf, 3),
                        "box":   [x1, y1, x2, y2]
                    })

                    # Annotation frame
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (77, 166, 255), 2)
                    cv2.rectangle(annotated, (x1, y1 - 18), (x1 + len(label)*9 + 48, y1), (77, 166, 255), -1)
                    cv2.putText(annotated, f"{label} {conf:.2f}",
                                (x1 + 4, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            return annotated, detections

        except Exception as e:
            log.error("❌ Détection: %s", e)
            return frame, []


# ═══════════════════════════════════════════════════════
# MODULE PRINCIPAL
# ═══════════════════════════════════════════════════════
class VisionObjetModule:
    def __init__(self):
        ensure_deps()
        SNAP_DIR.mkdir(parents=True, exist_ok=True)

        # Pas de détecteur local - on utilise vision_service (port 5051)
        self.active      = True
        self.frame_count = 0
        self.det_count   = 0
        self.history     = []
        self.lock        = Lock()
        self._stop       = Event()

        # MQTT
        try:
            from paho.mqtt.enums import CallbackAPIVersion
            self.client = mqtt_client.Client(CallbackAPIVersion.VERSION2, "PLUGIN_VISION_OBJET")
        except Exception:
            self.client = mqtt_client.Client("PLUGIN_VISION_OBJET")

        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

    # ── MQTT ────────────────────────────────────────────
    def _on_connect(self, client, ud, flags, rc, props=None):
        log.info("📡 Vision Objet connecté (rc=%s)", rc)
        client.subscribe([
            (TOPIC_IN,       0),  # shos/camera/vision — JSON meta de vision_service
            (TOPIC_CMD,      0),
            (TOPIC_SNAP_REQ, 0),
        ])

    def _on_message(self, client, ud, msg):
        topic = msg.topic

        # ── Commandes de contrôle ──
        if topic == TOPIC_CMD:
            try:
                cmd = json.loads(msg.payload.decode()).get("cmd", "")
                if cmd == "toggle":
                    self.active = not self.active
                    log.info("🔄 Détection: %s", "ON" if self.active else "OFF")
                elif cmd == "clear":
                    with self.lock:
                        self.history.clear()
                    log.info("🧹 Historique effacé")
                elif cmd == "conf_up":
                    global CONF_THRESHOLD
                    CONF_THRESHOLD = min(0.95, CONF_THRESHOLD + 0.05)
                elif cmd == "conf_down":
                    CONF_THRESHOLD = max(0.10, CONF_THRESHOLD - 0.05)
                elif cmd == "start":
                    self.active = True
                    log.info("▶ Détection démarrée")
                elif cmd == "stop":
                    self.active = False
                    log.info("⏸ Détection arrêtée")
                elif cmd == "snapshot":
                    Thread(target=self._save_snapshot, daemon=True).start()
            except Exception as e:
                log.error("CMD: %s", e)
            return

        # ── Snapshot manuel ──
        if topic == TOPIC_SNAP_REQ:
            Thread(target=self._save_snapshot, daemon=True).start()
            return

        # ── Métadonnées vision_service ──
        if topic == TOPIC_IN and self.active:
            self.frame_count += 1
            if self.frame_count % FRAME_SKIP != 0:
                return
            try:
                meta = json.loads(msg.payload.decode())
                # Filtrer par backend si demandé
                Thread(target=self._process_frame, args=(meta,), daemon=True).start()
            except Exception as e:
                log.debug("Parse meta: %s", e)

    # ── TRAITEMENT ──────────────────────────────────────
    def _process_frame(self, meta: dict):
        """
        Reçoit les métadonnées JSON publiées par vision_service
        sur shos/camera/vision et les relaie vers l'UI.
        Structure : {"backend": "yolo11n", "detections": [...], "fps": 12.0, ...}
        """
        try:
            backend    = meta.get("backend", "unknown")
            detections = meta.get("detections", [])
            fps        = meta.get("fps", 0)

            # Filtrer selon seuil local
            filtered = [
                d for d in detections
                if (d.get("confidence", d.get("conf", 0)) >= CONF_THRESHOLD)
            ]

            if filtered:
                self.det_count += len(filtered)
                entry = {
                    "ts":         datetime.now().strftime("%H:%M:%S"),
                    "detections": filtered,
                    "count":      len(filtered),
                    "backend":    backend,
                }
                with self.lock:
                    self.history.insert(0, entry)
                    if len(self.history) > MAX_HISTORY:
                        self.history.pop()

            # Publier toujours (même vide) pour garder UI en vie
            payload_out = {
                "plugin":     "vision_objet",
                "type":       "detection",
                "ts":         datetime.now().isoformat(),
                "active":     self.active,
                "backend":    backend,
                "detections": filtered,
                "count":      len(filtered),
                "total":      self.det_count,
                "threshold":  CONF_THRESHOLD,
                "fps":        round(fps, 1),
            }
            self.client.publish(TOPIC_OUT_DATA, json.dumps(payload_out))

        except Exception as e:
            log.error("Process frame: %s", e)

    # ── SNAPSHOT ────────────────────────────────────────
    def _save_snapshot(self):
        try:
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = SNAP_DIR / f"snap_vision_{ts}.jpg"
            # Sauvegarder la dernière frame annotée disponible
            with frame_lock:
                frame_to_save = latest_frame.copy() if latest_frame is not None else None
            if frame_to_save is not None:
                cv2.imwrite(str(path), frame_to_save)
                log.info("📸 Snapshot sauvé : %s", path.name)
            else:
                log.warning("📸 Snapshot : aucune frame disponible")
            self.client.publish(TOPIC_OUT_DATA, json.dumps({
                "plugin": "vision_objet",
                "type":   "snapshot_done",
                "file":   path.name,
                "ts":     ts,
            }))
        except Exception as e:
            log.error("Snapshot: %s", e)

    # ── RUN ─────────────────────────────────────────────
    def run(self):
        self.client.connect(BROKER, PORT, keepalive=60)
        log.info("🚀 Vision Objet démarré · YOLO11n · conf=%.2f", CONF_THRESHOLD)
        try:
            self.client.loop_forever()
        except KeyboardInterrupt:
            log.info("⛔ Arrêt")
        finally:
            self.client.disconnect()


if __name__ == "__main__":
    VisionObjetModule().run()