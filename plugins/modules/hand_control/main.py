#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
 S.H.O.S · HAND CONTROL V3 — Contrôle gestuel MediaPipe
===============================================================================
 10 gestes reconnus :
   START_SYSTEM   🖐  Main ouverte      → lance voice_assistant
   STOP_SYSTEM    ✊  Poing fermé       → kill module actif
   READ_TEXT      ☝  Index seul        → lance text_reader
   TRIGGER_VISION 👍  Pouce seul        → lance vision_objet
   PEACE          ✌  V (index+majeur)  → lance danger_monitor
   OK             👌  OK (pouce+index) → snapshot
   POINTING_DOWN  👇  Index pointé bas → volume down
   PINCH          🤏  Pincement        → snapshot vidéo
   SWIPE_LEFT     👈  Swipe gauche     → module précédent
   SWIPE_RIGHT    👉  Swipe droite     → module suivant

 Vérification des dépendances au 1er lancement (flag dans setup_done.flag)
 Source vidéo : MQTT topic shos/camera/raw (frames JPEG encodées)
 Topics publiés :
   shos/plugins/hand_control/data → geste confirmé + landmarks
   shos/camera/processed          → frame annotée
   shos/system/launch_module      → lancer un module
   shos/system/kill_module        → tuer un module
===============================================================================
"""

import os
import sys
import json
import time
import logging
import threading
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# VÉRIFICATION ET INSTALLATION DES DÉPENDANCES (une seule fois)
# ─────────────────────────────────────────────────────────────────────────────
MODULE_DIR = Path(__file__).resolve().parent
SETUP_FLAG = MODULE_DIR / "setup_done.flag"

def check_and_install_dependencies():
    """
    Vérifie que mediapipe, opencv et numpy sont installés.
    Lance l'installation si manquant. Écrit un flag pour ne le faire qu'une fois.
    """
    if SETUP_FLAG.exists():
        return  # Déjà fait

    print("[HandControl] Vérification des dépendances...")
    required = {
        "cv2":       "opencv-python",
        "mediapipe": "mediapipe",
        "numpy":     "numpy",
    }
    missing = []
    for module_name, pkg_name in required.items():
        try:
            __import__(module_name)
        except ImportError:
            missing.append(pkg_name)

    if missing:
        print(f"[HandControl] Installation : {missing}")
        import subprocess
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--break-system-packages", "--quiet"] + missing,
            check=True,
        )
        print("[HandControl] Installation terminée")
    else:
        print("[HandControl] Toutes les dépendances sont présentes")

    SETUP_FLAG.write_text("ok")

check_and_install_dependencies()

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS (après installation)
# ─────────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np
import mediapipe as mp
from paho.mqtt import client as mqtt_client

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BROKER       = "localhost"
MQTT_PORT    = 1883

# Source video : flux MJPEG HTTP (plus fiable que MQTT camera/raw)
VIDEO_STREAM_URL = "http://localhost:5051/video_feed"
CAMERA_FALLBACK_IDX = 0
T_CAMERA_OUT = "shos/camera/processed"
T_RESULTS    = "shos/plugins/hand_control/data"
T_CMD        = "shos/plugins/hand_control/cmd"
T_LAUNCH     = "shos/system/launch_module"
T_KILL       = "shos/system/kill_module"

# Délai de confirmation du geste (secondes)
CONFIRM_THRESHOLD = 0.4  # Réduit de 0.8s → 0.4s
# Délai anti-répétition entre deux actions (secondes)
ACTION_COOLDOWN   = 1.0  # Réduit de 2s → 1s

# Charger le gesture_map depuis config.json
_cfg_path = MODULE_DIR / "config.json"
try:
    with open(_cfg_path, "r", encoding="utf-8") as _f:
        _cfg = json.load(_f)
    GESTURE_MAP = _cfg.get("gesture_map", {})
except Exception:
    GESTURE_MAP = {}

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] HandControl - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("HandControl")


# =============================================================================
# MOTEUR DE RECONNAISSANCE GESTUELLE
# =============================================================================
class GestureEngine:
    """
    Reconnaît 10 gestes à partir des landmarks MediaPipe.
    Tous les gestes sont détectés sur une seule frame (pas de séquence temporelle)
    sauf SWIPE_LEFT et SWIPE_RIGHT qui nécessitent 2 frames consécutives.
    """

    def __init__(self):
        # Initialisation MediaPipe Hands
        self.mp_hands   = mp.solutions.hands
        self.mp_draw    = mp.solutions.drawing_utils
        self.mp_styles  = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,              # 2 mains pour meilleure détection
            min_detection_confidence=0.5, # Réduit de 0.7 → fonctionne de près
            min_tracking_confidence=0.4,  # Réduit de 0.5 → plus stable
        )
        # Historique pour détecter les swipes
        self._prev_wrist_x  = None
        self._swipe_frames  = 0
        logger.info("MediaPipe Hands initialisé")

    def _fingers_up(self, lms, handedness="Right") -> list:
        """
        Retourne [pouce, index, majeur, annulaire, auriculaire] (True = levé).
        Gère main gauche ET droite correctement.
        """
        fingers = []
        # Pouce — direction dépend de la main (gauche/droite)
        if handedness == "Right":
            fingers.append(lms[4].x < lms[3].x)   # main droite : pouce va à gauche
        else:
            fingers.append(lms[4].x > lms[3].x)   # main gauche : pouce va à droite
        # 4 autres doigts : tip.y < pip.y (universel)
        for tip in [8, 12, 16, 20]:
            fingers.append(lms[tip].y < lms[tip - 2].y)
        return fingers

    def _distance(self, a, b) -> float:
        return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5

    def recognize(self, lms, handedness="Right") -> str:
        """Retourne le nom du geste reconnu ou 'NONE'."""
        f = self._fingers_up(lms, handedness)
        up = sum(f)
        wrist_x = lms[0].x

        # ── Détection swipe ─────────────────────────────────────────────────
        if self._prev_wrist_x is not None:
            dx = wrist_x - self._prev_wrist_x
            if dx > 0.08 and up <= 2:
                self._prev_wrist_x = wrist_x
                return "SWIPE_RIGHT"
            elif dx < -0.08 and up <= 2:
                self._prev_wrist_x = wrist_x
                return "SWIPE_LEFT"
        self._prev_wrist_x = wrist_x

        # ── Gestes statiques — logique tolérante (marche de près et de loin) ──

        nb = sum(f)   # nombre total de doigts levés

        # 🖐 Main ouverte — 4 ou 5 doigts levés (tolère pouce mal détecté)
        if nb >= 4:
            return "START_SYSTEM"

        # 👍 Pouce seul (avant STOP pour éviter confusion)
        if f[0] and not f[1] and not f[2] and not f[3]:
            return "TRIGGER_VISION"

        # ✊ Poing — 0 ou 1 doigt levé
        if nb == 0 or (nb == 1 and f[0]):
            return "STOP_SYSTEM"

        # ☝ Index seul — distingue haut (READ) vs bas (POINTING_DOWN)
        if f[1] and not f[2] and not f[3] and not f[4]:
            if lms[8].y > lms[5].y + 0.04:   # tip plus bas que la base → index pointé bas
                return "POINTING_DOWN"
            return "READ_TEXT"

        # ✌ Victoire — index + majeur levés (tolère pouce)
        if f[1] and f[2] and not f[3] and not f[4]:
            return "PEACE"

        # 👌 OK — pouce + index rapprochés
        if not f[2] and not f[3]:
            dist = self._distance(lms[4], lms[8])
            if dist < 0.09:
                return "OK"

        # 🤏 Pincement — pouce + index très rapprochés, autres fermés
        if not f[2] and not f[3] and not f[4]:
            dist = self._distance(lms[4], lms[8])
            if dist < 0.07:
                return "PINCH"

        return "NONE"

    def process_frame(self, frame: np.ndarray):
        """
        Traite une frame BGR.
        Retourne (frame_annotée, geste_reconnu).
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        gesture = "NONE"
        annotated = frame.copy()

        if results.multi_hand_landmarks:
            hands_data = zip(
                results.multi_hand_landmarks,
                results.multi_handedness if results.multi_handedness else []
            )
            for hand_lms, hand_info in hands_data:
                # Récupérer gauche/droite depuis MediaPipe
                try:
                    handedness = hand_info.classification[0].label  # "Left" ou "Right"
                except Exception:
                    handedness = "Right"
                # Dessin landmarks
                self.mp_draw.draw_landmarks(
                    annotated, hand_lms, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_styles.get_default_hand_landmarks_style(),
                    self.mp_styles.get_default_hand_connections_style(),
                )
                g = self.recognize(hand_lms.landmark, handedness)
                if g != "NONE":
                    gesture = g  # prend le premier geste valide

        return annotated, gesture

    def close(self):
        self.hands.close()


# =============================================================================
# PLUGIN PRINCIPAL
# =============================================================================
class HandControlPlugin:

    def __init__(self):
        self.engine   = GestureEngine()
        self.running  = True
        self.active   = True   # peut être pausé par cmd MQTT

        self.last_gesture      = "NONE"
        self.gesture_start_ts  = 0.0
        self.last_action_ts    = {}  # cooldown par geste

        self._setup_mqtt()
        # Thread vidéo (démarré dans run())
        self._video_thread = None

    # ── MQTT ────────────────────────────────────────────────────────────────
    def _setup_mqtt(self):
        try:
            try:
                from paho.mqtt.enums import CallbackAPIVersion
                self.client = mqtt_client.Client(CallbackAPIVersion.VERSION2, "PLUGIN_GESTURE_V3")
            except ImportError:
                self.client = mqtt_client.Client("PLUGIN_GESTURE_V3")
            self.client.on_connect = self._on_connect
            self.client.on_message = self._on_message
            self.client.connect(BROKER, MQTT_PORT, 60)
            self.client.loop_start()
        except Exception as e:
            logger.warning("MQTT : %s", e)
            self.client = None

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        logger.info("MQTT connecté (rc=%s)", rc)
        # Écoute seulement les commandes de contrôle
        client.subscribe([(T_CMD, 1)])
        self._publish(T_RESULTS, {
            "plugin": "hand_control",
            "status": "ready",
            "gesture": "NONE",
            "gesture_map_count": len(GESTURE_MAP),
        })

    def _on_message(self, client, userdata, msg):
        if msg.topic == T_CMD:
            try:
                payload = json.loads(msg.payload.decode())
                self._handle_cmd(payload)
            except Exception:
                pass

    def _open_video_source(self):
        """
        Ouvre la source vidéo :
          1. Flux MJPEG HTTP de vision_service (localhost:5051)
          2. Fallback : caméra locale OpenCV directe
        """
        # Essai flux MJPEG
        try:
            cap = cv2.VideoCapture(VIDEO_STREAM_URL)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    logger.info("Source vidéo : MJPEG HTTP %s", VIDEO_STREAM_URL)
                    return cap
                cap.release()
        except Exception as e:
            logger.debug("MJPEG HTTP : %s", e)

        # Fallback caméra locale
        try:
            cap = cv2.VideoCapture(CAMERA_FALLBACK_IDX)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                logger.info("Source vidéo : caméra locale index %d", CAMERA_FALLBACK_IDX)
                return cap
        except Exception as e:
            logger.warning("Caméra locale : %s", e)

        logger.error("Aucune source vidéo disponible")
        return None

    def _video_loop(self):
        """Boucle de capture vidéo — tourne dans un thread séparé."""
        cap = self._open_video_source()
        if cap is None:
            logger.error("Boucle vidéo annulée — pas de source")
            return

        logger.info("Boucle vidéo démarrée")
        fail_count = 0

        while self.running:
            if not self.active:
                time.sleep(0.1)
                continue

            ret, frame = cap.read()
            if not ret or frame is None:
                fail_count += 1
                if fail_count > 30:
                    logger.warning("Flux perdu — tentative reconnexion...")
                    cap.release()
                    time.sleep(2)
                    cap = self._open_video_source()
                    if cap is None:
                        break
                    fail_count = 0
                time.sleep(0.05)
                continue

            fail_count = 0
            self._process_frame(frame)

        cap.release()
        logger.info("Boucle vidéo arrêtée")

    def _handle_cmd(self, payload):
        cmd = str(payload.get("cmd", "")).lower()
        if cmd == "start":
            self.active = True
            self._publish(T_RESULTS, {"plugin": "hand_control", "status": "started"})
            logger.info("Détection démarrée")
        elif cmd == "stop":
            self.active = False
            self._publish(T_RESULTS, {"plugin": "hand_control", "status": "stopped"})
            logger.info("Détection arrêtée")
        elif cmd == "reset":
            self.last_gesture = "NONE"
            self.gesture_start_ts = 0.0
            self._publish(T_RESULTS, {"plugin": "hand_control", "status": "reset"})
        elif cmd == "snapshot":
            self._publish("shos/camera/snapshot_request",
                          {"trigger": "hand_control", "ts": time.time()})

    # ── Traitement frame ────────────────────────────────────────────────────
    def _process_frame(self, frame: np.ndarray):
        annotated, gesture = self.engine.process_frame(frame)

        # Overlay badge geste
        if gesture != "NONE":
            info = GESTURE_MAP.get(gesture, {})
            label = info.get("emoji", "") + " " + info.get("label", gesture)
            cv2.rectangle(annotated, (0, 0), (300, 28), (0, 0, 0), -1)
            cv2.putText(annotated, label, (8, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (77, 212, 232), 1, cv2.LINE_AA)

        # Envoi frame annotée
        try:
            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 35])
            if self.client:
                self.client.publish(T_CAMERA_OUT, buf.tobytes(), qos=0)
        except Exception:
            pass

        # Confirmation du geste
        now = time.time()
        if gesture != self.last_gesture:
            self.last_gesture     = gesture
            self.gesture_start_ts = now if gesture != "NONE" else 0.0
        elif gesture != "NONE":
            elapsed = now - self.gesture_start_ts
            if elapsed >= CONFIRM_THRESHOLD:
                self._execute_gesture(gesture)
                # Reset timer avec cooldown
                self.gesture_start_ts = now + ACTION_COOLDOWN

    # ── Exécution d'un geste confirmé ───────────────────────────────────────
    def _execute_gesture(self, gesture: str):
        now = time.time()
        # Anti-répétition (cooldown par geste)
        if now - self.last_action_ts.get(gesture, 0) < ACTION_COOLDOWN:
            return
        self.last_action_ts[gesture] = now

        info = GESTURE_MAP.get(gesture)
        if not info:
            logger.debug("Geste non mappé : %s", gesture)
            return

        topic   = info.get("topic", T_RESULTS)
        payload = dict(info.get("payload", {}))
        payload["gesture"]   = gesture
        payload["source"]    = "hand_control"
        payload["ts"]        = now
        payload["confirmed"] = True

        self._publish(topic, payload)
        logger.info("🎯 Geste confirmé : %s → %s", gesture, info.get("label", ""))

        # Publication aussi vers data pour l'UI
        self._publish(T_RESULTS, {
            "plugin":  "hand_control",
            "gesture": gesture,
            "label":   info.get("label", gesture),
            "emoji":   info.get("emoji", ""),
            "action":  info.get("action", ""),
            "status":  "confirmed",
            "ts":      now,
        })

    # ── Helper publish ──────────────────────────────────────────────────────
    def _publish(self, topic: str, data: dict):
        if not self.client:
            return
        try:
            self.client.publish(topic, json.dumps(data), qos=0)
        except Exception as e:
            logger.debug("publish %s : %s", topic, e)

    # ── Run ────────────────────────────────────────────────────────────────
    def run(self):
        logger.info("Hand Control V3 démarré")
        logger.info("Gestes configurés : %d", len(GESTURE_MAP))
        logger.info("Source vidéo : %s", VIDEO_STREAM_URL)

        # Lancer le thread vidéo
        self._video_thread = threading.Thread(
            target=self._video_loop, name="HandVideoLoop", daemon=True
        )
        self._video_thread.start()

        try:
            while self.running:
                time.sleep(0.5)
        except KeyboardInterrupt:
            logger.info("Arrêt")
        finally:
            self.running = False
            self.engine.close()
            if self.client:
                try:
                    self.client.loop_stop()
                    self.client.disconnect()
                except Exception:
                    pass


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    HandControlPlugin().run()