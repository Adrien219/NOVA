#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
 S.H.O.S · TEXT READER V3 — OCR adaptatif Windows + Pi
===============================================================================
 Source vidéo : flux MJPEG HTTP depuis vision_service (port 5051)
 OCR : Tesseract (si installé) → EasyOCR (fallback) → PIL (dernier recours)

 Tesseract est détecté automatiquement :
   Windows : cherche dans Program Files
   Pi      : cherche dans /usr/bin/tesseract

 Topics MQTT écoutés :
   shos/plugins/text_reader/cmd     → commandes (start, stop, toggle, snapshot)

 Topics MQTT publiés :
   shos/plugins/text_reader/data    → texte extrait + statut
   shos/plugins/voice_assistant/cmd → lecture vocale du texte
===============================================================================
"""

import json
import sys
import time
import logging
import threading
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import paho.mqtt.client as mqtt

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BROKER   = "localhost"
MQTT_PORT = 1883

VIDEO_URL = "http://localhost:5051/video_feed"

T_CMD    = "shos/plugins/text_reader/cmd"
T_DATA   = "shos/plugins/text_reader/data"
T_VOICE  = "shos/plugins/voice_assistant/cmd"

# Intervalle entre 2 OCR (secondes) — réduit la charge CPU
OCR_INTERVAL = 2.0

# Détection Tesseract
TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    "/usr/bin/tesseract",
    "/usr/local/bin/tesseract",
]

# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] TextReader - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TextReader")


# =============================================================================
# MOTEUR OCR — Tesseract → EasyOCR → fallback
# =============================================================================
class OCREngine:

    def __init__(self):
        self.backend = None
        self._tesseract_cmd = None
        self._easy = None
        self._detect()

    def _detect(self):
        """Choisit le meilleur backend OCR disponible."""

        # 1. Tesseract
        try:
            import pytesseract
            # Chercher l'exécutable
            tess_found = None
            for path in TESSERACT_PATHS:
                if Path(path).exists():
                    tess_found = path
                    break
            if tess_found:
                pytesseract.pytesseract.tesseract_cmd = tess_found
                # Test rapide
                test = np.zeros((30, 100, 3), dtype=np.uint8)
                pytesseract.image_to_string(test)
                self._tesseract = pytesseract
                self.backend = "tesseract"
                logger.info("OCR backend : Tesseract (%s)", tess_found)
                return
            else:
                logger.warning("Tesseract non trouvé — tentative EasyOCR")
        except Exception as e:
            logger.debug("Tesseract : %s", e)

        # 2. EasyOCR
        try:
            import easyocr
            self._easy = easyocr.Reader(['fr', 'en'], gpu=False, verbose=False)
            self.backend = "easyocr"
            logger.info("OCR backend : EasyOCR")
            return
        except ImportError:
            logger.warning("EasyOCR non installé (pip install easyocr)")
        except Exception as e:
            logger.debug("EasyOCR : %s", e)

        # 3. PIL + Tesseract CLI (fallback subprocess)
        try:
            import subprocess
            result = subprocess.run(
                ["tesseract", "--version"],
                capture_output=True, text=True, timeout=3,
            )
            if result.returncode == 0:
                self.backend = "tesseract_cli"
                logger.info("OCR backend : Tesseract CLI")
                return
        except Exception:
            pass

        logger.error("Aucun backend OCR disponible !")
        logger.error("Installer : pip install pytesseract easyocr")
        logger.error("           + sudo apt install tesseract-ocr tesseract-ocr-fra (Pi)")
        logger.error("           + https://github.com/UB-Mannheim/tesseract/wiki (Windows)")
        self.backend = None

    def is_available(self):
        return self.backend is not None

    def extract(self, frame: np.ndarray) -> str:
        """Lance l'OCR sur une frame BGR. Retourne le texte extrait."""
        if not self.is_available():
            return ""
        try:
            # Prétraitement commun
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Améliorer le contraste
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            # Binarisation Otsu
            _, binary = cv2.threshold(gray, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            if self.backend == "tesseract":
                text = self._tesseract.image_to_string(
                    binary, lang="fra+eng",
                    config="--oem 3 --psm 3",
                )
                return text.strip()

            elif self.backend == "easyocr":
                results = self._easy.readtext(gray, detail=0, paragraph=True)
                return " ".join(results).strip()

            elif self.backend == "tesseract_cli":
                import subprocess, tempfile, os
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                    tmp_in = f.name
                cv2.imwrite(tmp_in, binary)
                tmp_out = tmp_in.replace(".png", "_out")
                subprocess.run(
                    ["tesseract", tmp_in, tmp_out, "-l", "fra+eng",
                     "--oem", "3", "--psm", "3"],
                    capture_output=True, timeout=10,
                )
                txt_path = tmp_out + ".txt"
                text = Path(txt_path).read_text(encoding="utf-8", errors="replace").strip()
                os.unlink(tmp_in)
                os.unlink(txt_path)
                return text

        except Exception as e:
            logger.warning("OCR extract : %s", e)
        return ""


# =============================================================================
# MODULE PRINCIPAL
# =============================================================================
class TextReaderPlugin:

    def __init__(self):
        self.engine  = OCREngine()
        self.active  = True
        self.running = True

        self.last_ocr_time = 0.0
        self.last_text     = ""
        self.ocr_count     = 0
        self.snap_request  = False

        self._lock = threading.Lock()

        self._setup_mqtt()
        # Thread capture vidéo + OCR
        self._ocr_thread = threading.Thread(
            target=self._ocr_loop, name="OCR_Loop", daemon=True
        )
        self._ocr_thread.start()

    # ── MQTT ────────────────────────────────────────────────────────────────
    def _setup_mqtt(self):
        try:
            try:
                from paho.mqtt.enums import CallbackAPIVersion
                self.client = mqtt.Client(CallbackAPIVersion.VERSION2,
                                          "PLUGIN_TEXT_READER_V3")
            except ImportError:
                self.client = mqtt.Client("PLUGIN_TEXT_READER_V3")
            self.client.on_connect = self._on_connect
            self.client.on_message = self._on_message
            self.client.connect(BROKER, MQTT_PORT, 60)
            self.client.loop_start()
        except Exception as e:
            logger.error("MQTT : %s", e)

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        logger.info("MQTT connecté (rc=%s)", rc)
        client.subscribe([(T_CMD, 1)])
        self._publish({
            "type":    "status",
            "status":  "ready",
            "backend": self.engine.backend or "none",
            "active":  self.active,
        })

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
        except Exception:
            return
        self._handle_cmd(payload)

    # ── Commandes ────────────────────────────────────────────────────────────
    def _handle_cmd(self, payload: dict):
        cmd = str(payload.get("cmd", "")).lower()

        if cmd in ("start", "activate"):
            self.active = True
            logger.info("▶ OCR activé")
            self._publish({"type": "status", "status": "active", "active": True})

        elif cmd in ("stop", "pause"):
            self.active = False
            logger.info("⏸ OCR en pause")
            self._publish({"type": "status", "status": "paused", "active": False})

        elif cmd == "toggle":
            self.active = not self.active
            logger.info("Toggle → %s", "actif" if self.active else "pause")
            self._publish({
                "type": "status",
                "status": "active" if self.active else "paused",
                "active": self.active,
            })

        elif cmd == "snapshot":
            with self._lock:
                self.snap_request = True

        elif cmd == "read_aloud":
            # Lire le dernier texte via voice_assistant
            if self.last_text:
                try:
                    self.client.publish(T_VOICE, json.dumps({
                        "cmd":  "speak",
                        "text": self.last_text[:500],  # max 500 chars
                    }))
                    logger.info("Lecture vocale envoyée")
                except Exception as e:
                    logger.warning("read_aloud : %s", e)

        elif cmd == "clear":
            self.last_text = ""
            self.ocr_count = 0
            self._publish({
                "type":   "cleared",
                "active": self.active,
            })

    # ── Boucle OCR ───────────────────────────────────────────────────────────
    def _open_video(self):
        """Ouvre la source vidéo MJPEG (vision_service) ou caméra locale."""
        try:
            cap = cv2.VideoCapture(VIDEO_URL)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    logger.info("Source vidéo : MJPEG %s", VIDEO_URL)
                    return cap
                cap.release()
        except Exception as e:
            logger.debug("MJPEG : %s", e)

        # Fallback caméra locale
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                logger.info("Source vidéo : caméra locale")
                return cap
        except Exception as e:
            logger.warning("Caméra locale : %s", e)
        return None

    def _ocr_loop(self):
        if not self.engine.is_available():
            logger.error("OCR non disponible — boucle annulée")
            self._publish({
                "type":    "error",
                "msg":     "Aucun backend OCR disponible",
                "active":  False,
            })
            return

        logger.info("Boucle OCR démarrée (backend: %s)", self.engine.backend)

        cap = self._open_video()
        if cap is None:
            logger.error("Aucune source vidéo")
            return

        while self.running:
            # Lire la frame
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning("Flux perdu — reconnexion...")
                cap.release()
                time.sleep(2)
                cap = self._open_video()
                if cap is None:
                    break
                continue

            # Snapshot à la demande
            with self._lock:
                do_snap = self.snap_request
                self.snap_request = False

            if do_snap:
                self._do_ocr(frame, forced=True)
                continue

            # OCR périodique
            if not self.active:
                time.sleep(0.2)
                continue

            now = time.time()
            if now - self.last_ocr_time >= OCR_INTERVAL:
                self._do_ocr(frame)

            time.sleep(0.05)  # ~20 fps lecture flux

        cap.release()

    def _do_ocr(self, frame: np.ndarray, forced=False):
        """Lance l'OCR sur la frame courante."""
        self.last_ocr_time = time.time()

        # Rogner au centre (zone de visée)
        h, w = frame.shape[:2]
        margin_h = int(h * 0.18)
        margin_w = int(w * 0.12)
        roi = frame[margin_h: h - margin_h, margin_w: w - margin_w]

        text = self.engine.extract(roi if roi.size > 0 else frame)

        if text and len(text) > 2:
            self.last_text = text
            self.ocr_count += 1
            logger.info("📝 Texte (%d): %s", len(text), text[:60].replace('\n', ' '))
            self._publish({
                "type":    "text",
                "text":    text,
                "chars":   len(text),
                "count":   self.ocr_count,
                "backend": self.engine.backend,
                "forced":  forced,
                "active":  self.active,
            })
        else:
            self._publish({
                "type":    "empty",
                "text":    "",
                "backend": self.engine.backend,
                "active":  self.active,
            })

    # ── Publish ─────────────────────────────────────────────────────────────
    def _publish(self, data: dict):
        try:
            payload = {"plugin": "text_reader", **data, "ts": time.time()}
            self.client.publish(T_DATA, json.dumps(payload), qos=0)
        except Exception as e:
            logger.debug("publish : %s", e)

    # ── Run ──────────────────────────────────────────────────────────────────
    def run(self):
        logger.info("Text Reader V3 démarré")
        logger.info("Backend OCR : %s", self.engine.backend or "AUCUN")
        logger.info("Source vidéo : %s", VIDEO_URL)
        logger.info("Intervalle OCR : %.1fs", OCR_INTERVAL)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Arrêt")
        finally:
            self.running = False
            self._publish({"type": "status", "status": "offline", "active": False})
            try:
                self.client.loop_stop()
                self.client.disconnect()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    TextReaderPlugin().run()