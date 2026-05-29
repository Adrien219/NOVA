#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
 S.H.O.S - VISION SERVICE V3 MULTI-BACKEND
===============================================================================
 3 backends interchangeables a chaud via API REST :

   - MEDIAPIPE    : Hands + Pose (rapide, ~15 fps Pi 4B)
   - YOLO11N      : Detection 80 classes COCO via ultralytics
   - MOONDREAM 2  : VLM 0.5B (caption FR, 1 image / 5-10s)

 Switch dynamique :
   POST /api/backend  { "backend": "yolo11n" }
   GET  /api/backend  -> backend actif + liste disponibles

 Modeles attendus dans ./models/ :
   models/yolo11n_ncnn_model/     (export NCNN) ou yolo11n.pt
   models/moondream-0_5b-int4.gguf
   models/mmproj.gguf

 Si un modele est absent -> backend marque "non disponible".
 MediaPipe charge toujours si installe (pas de fichier requis).
===============================================================================
"""

import os
import sys
import json
import time
import threading
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict

import cv2
import numpy as np
from flask import Flask, Response, request, jsonify
import paho.mqtt.client as mqtt

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
PORT_WEB    = 5051
MQTT_BROKER = "localhost"
MQTT_PORT   = 1883
MQTT_TOPIC  = "shos/camera/vision"

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR   = PROJECT_ROOT / "models"

YOLO_NCNN_PATH       = MODELS_DIR / "yolo11n_ncnn_model"
YOLO_PT_PATH         = MODELS_DIR / "yolo11n.pt"
MOONDREAM_MODEL_PATH = MODELS_DIR / "moondream-0_5b-int4.gguf"
MOONDREAM_MMPROJ     = MODELS_DIR / "mmproj.gguf"

DEFAULT_BACKEND = "mediapipe"
BACKENDS = ("mediapipe", "yolo11n", "moondream")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger("VisionService")

app = Flask(__name__)

# ----- CORS -----
@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


# =============================================================================
# INTERFACE COMMUNE DES BACKENDS
# =============================================================================
class VisionBackend:
    name = "base"
    def is_available(self) -> bool:
        return False
    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, dict]:
        return frame, {"backend": self.name, "detections": []}
    def close(self):
        pass


# -----------------------------------------------------------------------------
# BACKEND 1 : MEDIAPIPE
# -----------------------------------------------------------------------------
class MediaPipeBackend(VisionBackend):
    name = "mediapipe"

    def __init__(self):
        self.available = False
        try:
            import mediapipe as mp
            # Workaround Windows + Py 3.12
            try:
                _ = mp.solutions
            except AttributeError:
                from mediapipe.python import solutions as _sol
                mp.solutions = _sol

            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False, max_num_hands=1,
                min_detection_confidence=0.6, min_tracking_confidence=0.5,
            )
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                min_detection_confidence=0.5, min_tracking_confidence=0.5,
            )
            self.mp_draw = mp.solutions.drawing_utils
            self.mp_draw_styles = mp.solutions.drawing_styles
            self.available = True
            logger.info("✅ MediaPipe initialise (Hands + Pose)")
        except Exception as e:
            logger.error("❌ MediaPipe indisponible : %s", e)

    def is_available(self):
        return self.available

    def process(self, frame):
        if not self.available:
            return frame, {"backend": self.name, "error": "not_available"}

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hr = self.hands.process(rgb)
        pr = self.pose.process(rgb)

        annotated = frame.copy()
        hands_n = 0
        pose_ok = False

        if hr.multi_hand_landmarks:
            hands_n = len(hr.multi_hand_landmarks)
            for lm in hr.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    annotated, lm, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw_styles.get_default_hand_landmarks_style(),
                    self.mp_draw_styles.get_default_hand_connections_style(),
                )
        if pr.pose_landmarks:
            pose_ok = True
            self.mp_draw.draw_landmarks(
                annotated, pr.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_draw_styles.get_default_pose_landmarks_style(),
            )

        return annotated, {
            "backend": self.name,
            "hands_detected": hands_n,
            "pose_detected": pose_ok,
            "detections": [],
        }

    def close(self):
        try:
            self.hands.close()
            self.pose.close()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# BACKEND 2 : YOLO 11n (NCNN ou PT)
# -----------------------------------------------------------------------------
class YoloBackend(VisionBackend):
    name = "yolo11n"

    COCO = [
        "personne","velo","voiture","moto","avion","bus","train","camion","bateau","feu",
        "stop","parcmetre","banc","oiseau","chat","chien","cheval","mouton","vache","elephant",
        "ours","zebre","girafe","sac a dos","parapluie","sac","cravate","valise","frisbee","skis",
        "snowboard","ballon","cerf-volant","batte","gant","skate","surf","raquette","bouteille","verre",
        "tasse","fourchette","couteau","cuillere","bol","banane","pomme","sandwich","orange","brocoli",
        "carotte","hot dog","pizza","donut","gateau","chaise","canape","plante","lit","table",
        "toilettes","TV","laptop","souris","telecommande","clavier","telephone","micro-ondes","four","grille-pain",
        "evier","frigo","livre","horloge","vase","ciseaux","peluche","seche-cheveux","brosse a dents","_",
    ]

    def __init__(self):
        self.model = None
        self.available = False
        self._load()

    def _load(self):
        # Priorite : NCNN > PT
        model_path = None
        if YOLO_NCNN_PATH.exists():
            model_path = YOLO_NCNN_PATH
            logger.info("YOLO : utilisation du modele NCNN %s", model_path)
        elif YOLO_PT_PATH.exists():
            model_path = YOLO_PT_PATH
            logger.info("YOLO : utilisation du modele PyTorch %s", model_path)
        else:
            logger.warning("⚠ YOLO : aucun modele trouve dans %s", MODELS_DIR)
            logger.warning("   Attendu : yolo11n_ncnn_model/ OU yolo11n.pt")
            return

        try:
            from ultralytics import YOLO
            self.model = YOLO(str(model_path), task="detect")
            self.available = True
            logger.info("✅ YOLO11n charge")
        except ImportError:
            logger.warning("⚠ ultralytics non installe - pip install ultralytics")
        except Exception as e:
            logger.error("❌ YOLO load : %s", e)

    def is_available(self):
        return self.available

    def process(self, frame):
        if not self.available:
            return frame, {"backend": self.name, "error": "model_not_loaded"}
        try:
            results = self.model(frame, verbose=False, imgsz=320, conf=0.4)
        except Exception as e:
            return frame, {"backend": self.name, "error": str(e)}

        annotated = frame.copy()
        dets = []
        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                label = self.COCO[cls_id] if cls_id < len(self.COCO) else f"cls_{cls_id}"

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (77, 212, 232), 2)
                txt = f"{label} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 6, y1), (77, 212, 232), -1)
                cv2.putText(annotated, txt, (x1 + 3, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (12, 15, 11), 1, cv2.LINE_AA)
                dets.append({
                    "class": label,
                    "confidence": round(conf, 3),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                })

        return annotated, {
            "backend": self.name,
            "detections": dets,
            "objects_count": len(dets),
        }


# -----------------------------------------------------------------------------
# BACKEND 3 : MOONDREAM 2 (asynchrone)
# -----------------------------------------------------------------------------
class MoondreamBackend(VisionBackend):
    name = "moondream"

    def __init__(self):
        self.available = False
        self.model = None
        self._caption = "Analyse en cours..."
        self._ts = 0.0
        self._inflight = False
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        if not MOONDREAM_MODEL_PATH.exists():
            logger.warning("⚠ Moondream : GGUF introuvable %s", MOONDREAM_MODEL_PATH)
            return
        if not MOONDREAM_MMPROJ.exists():
            logger.warning("⚠ Moondream : mmproj.gguf manquant %s", MOONDREAM_MMPROJ)
            return
        try:
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import MoondreamChatHandler
            self.handler = MoondreamChatHandler(clip_model_path=str(MOONDREAM_MMPROJ))
            self.model = Llama(
                model_path=str(MOONDREAM_MODEL_PATH),
                chat_handler=self.handler,
                n_ctx=2048, verbose=False,
            )
            self.available = True
            logger.info("✅ Moondream 2 charge")
        except ImportError:
            logger.warning("⚠ llama-cpp-python absent - pip install llama-cpp-python")
        except Exception as e:
            logger.warning("⚠ Moondream load : %s", e)

    def is_available(self):
        return self.available

    def _analyze_async(self, frame):
        if self._inflight:
            return
        self._inflight = True

        def _run():
            try:
                import base64
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ok:
                    return
                b64 = base64.b64encode(buf.tobytes()).decode("ascii")
                resp = self.model.create_chat_completion(
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                            {"type": "text", "text": "Decris brievement la scene en francais (1 phrase)."},
                        ],
                    }],
                    max_tokens=80,
                )
                txt = resp["choices"][0]["message"]["content"].strip()
                with self._lock:
                    self._caption = txt
                    self._ts = time.time()
            except Exception as e:
                logger.warning("Moondream inference : %s", e)
            finally:
                self._inflight = False

        threading.Thread(target=_run, daemon=True).start()

    def process(self, frame):
        if not self.available:
            return frame, {"backend": self.name, "error": "model_not_loaded"}

        now = time.time()
        if (now - self._ts) > 5.0:
            self._analyze_async(frame.copy())

        annotated = frame.copy()
        h, w = annotated.shape[:2]
        with self._lock:
            caption = self._caption

        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, h - 60), (w, h), (12, 15, 11), -1)
        cv2.addWeighted(overlay, 0.78, annotated, 0.22, 0, annotated)

        cv2.putText(annotated, "MOONDREAM 2", (10, h - 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (77, 212, 232), 1, cv2.LINE_AA)
        cap = caption[:90] + ("..." if len(caption) > 90 else "")
        cv2.putText(annotated, cap, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
        if self._inflight:
            cv2.circle(annotated, (w - 20, h - 30), 5, (77, 166, 255), -1)

        return annotated, {
            "backend": self.name,
            "caption": caption,
            "last_update": self._ts,
            "inflight": self._inflight,
            "detections": [],
        }


# =============================================================================
# CAMERA SOURCE
# =============================================================================
class CameraSource:
    def __init__(self, width=640, height=480):
        self.width, self.height = width, height
        self.picam2 = None
        self.cap = None
        self.using_picamera = False
        self._open()

    def _open(self):
        try:
            from picamera2 import Picamera2
            self.picam2 = Picamera2()
            cfg = self.picam2.create_preview_configuration(
                main={"format": "BGR888", "size": (self.width, self.height)},
            )
            self.picam2.configure(cfg)
            self.picam2.start()
            self.using_picamera = True
            logger.info("Picamera2 active (%dx%d)", self.width, self.height)
            return
        except ImportError:
            logger.warning("Picamera2 non installe - fallback OpenCV")
        except Exception as e:
            logger.error("Picamera2: %s - fallback OpenCV", e)

        try:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                logger.info("OpenCV VideoCapture(0) ouvert")
            else:
                logger.error("Aucune camera disponible")
                self.cap = None
        except Exception as e:
            logger.error("OpenCV fallback: %s", e)
            self.cap = None

    def read(self):
        if self.using_picamera and self.picam2:
            try:
                return self.picam2.capture_array()
            except Exception:
                return None
        if self.cap:
            ok, f = self.cap.read()
            return f if ok else None
        return None

    def is_ready(self):
        return self.using_picamera or (self.cap is not None and self.cap.isOpened())

    def close(self):
        try:
            if self.picam2:
                self.picam2.stop()
        except Exception:
            pass
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass


# =============================================================================
# VISION SERVICE
# =============================================================================
class VisionService:
    def __init__(self):
        self.camera = CameraSource()

        logger.info("Initialisation des backends...")
        self.backends: Dict[str, VisionBackend] = {
            "mediapipe": MediaPipeBackend(),
            "yolo11n":   YoloBackend(),
            "moondream": MoondreamBackend(),
        }

        # Backend par defaut : premier dispo
        if self.backends[DEFAULT_BACKEND].is_available():
            self.active_name = DEFAULT_BACKEND
        else:
            self.active_name = next(
                (n for n, b in self.backends.items() if b.is_available()),
                DEFAULT_BACKEND,
            )

        self._lock = threading.Lock()
        self.output_frame = None
        self.frame_lock = threading.Lock()
        self.is_running = True
        self.frame_count = 0
        self.last_meta = {"backend": self.active_name}
        self._fps = 0.0
        self._fps_counter = 0
        self._fps_timer = time.time()
        self.mqtt = self._init_mqtt()
        self._last_mqtt = 0.0

        avail = [n for n, b in self.backends.items() if b.is_available()]
        logger.info("Backends disponibles : %s", avail)
        logger.info("Backend initial : %s", self.active_name)

    def _init_mqtt(self):
        try:
            try:
                from paho.mqtt.enums import CallbackAPIVersion
                c = mqtt.Client(CallbackAPIVersion.VERSION2, "NOVA_VISION_V3")
            except ImportError:
                c = mqtt.Client("NOVA_VISION_V3")
            c.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
            c.loop_start()
            logger.info("MQTT connecte %s:%d", MQTT_BROKER, MQTT_PORT)
            return c
        except Exception as e:
            logger.warning("MQTT non disponible : %s", e)
            return None

    def publish_meta(self, meta):
        if not self.mqtt:
            return
        now = time.time()
        if now - self._last_mqtt < 0.2:
            return
        self._last_mqtt = now
        try:
            payload = {**meta, "fps": round(self._fps, 1), "ts": now}
            self.mqtt.publish(MQTT_TOPIC, json.dumps(payload), qos=0)
        except Exception:
            pass

    def set_backend(self, name):
        if name not in self.backends:
            return {"ok": False, "msg": f"Backend inconnu : {name}"}
        if not self.backends[name].is_available():
            return {"ok": False, "msg": f"Backend {name} non disponible (modele manquant ?)"}
        with self._lock:
            self.active_name = name
        logger.info("Backend actif -> %s", name)
        if self.mqtt:
            try:
                self.mqtt.publish("shos/camera/backend",
                                  json.dumps({"backend": name, "ts": time.time()}))
            except Exception:
                pass
        return {"ok": True, "backend": name}

    def get_active(self):
        with self._lock:
            return self.backends[self.active_name]

    def run(self):
        if not self.camera.is_ready():
            logger.error("Pas de camera - boucle vision annulee")
            return
        for _ in range(8):
            self.camera.read()
            time.sleep(0.04)
        logger.info("Vision demarree - backend : %s", self.active_name)

        while self.is_running:
            try:
                frame = self.camera.read()
                if frame is None or frame.size == 0:
                    time.sleep(0.05)
                    continue

                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                if len(frame.shape) != 3 or frame.shape[2] != 3:
                    try:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    except Exception:
                        continue
                frame = np.ascontiguousarray(frame)

                backend = self.get_active()
                try:
                    annotated, meta = backend.process(frame)
                except Exception as e:
                    logger.warning("Backend %s : %s", self.active_name, e)
                    annotated, meta = frame, {"backend": self.active_name, "error": str(e)}

                self._fps_counter += 1
                if time.time() - self._fps_timer >= 1.0:
                    self._fps = self._fps_counter / (time.time() - self._fps_timer)
                    self._fps_counter = 0
                    self._fps_timer = time.time()

                hud = f"{self.active_name.upper()} | {self._fps:.1f} fps"
                cv2.putText(annotated, hud, (8, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (77, 212, 232), 1, cv2.LINE_AA)

                with self.frame_lock:
                    self.output_frame = annotated.copy()
                    self.last_meta = meta

                self.publish_meta(meta)
                self.frame_count += 1
            except Exception as e:
                logger.error("Erreur boucle : %s", e)
                time.sleep(0.1)

    def generate_frames(self):
        while True:
            with self.frame_lock:
                frame = None if self.output_frame is None else self.output_frame.copy()
            if frame is None:
                time.sleep(0.03)
                continue
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ok:
                continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

    def stop(self):
        self.is_running = False
        for b in self.backends.values():
            try:
                b.close()
            except Exception:
                pass
        self.camera.close()
        if self.mqtt:
            try:
                self.mqtt.loop_stop()
                self.mqtt.disconnect()
            except Exception:
                pass


vision = VisionService()


# =============================================================================
# ROUTES FLASK
# =============================================================================
@app.route("/")
def index():
    return """
    <!DOCTYPE html><html><head><title>SHOS Vision V3</title>
    <style>body{background:#0e1117;color:#e8d44d;font-family:monospace;text-align:center;padding:20px}
    img{max-width:90%;border:2px solid #e8d44d;border-radius:10px;margin:20px}
    h1{font-weight:400;letter-spacing:.1em}
    button{background:rgba(232,212,77,.1);border:1px solid #e8d44d;color:#e8d44d;padding:8px 16px;margin:4px;cursor:pointer;border-radius:6px;font-family:inherit}
    button:hover{background:rgba(232,212,77,.2)}</style></head><body>
    <h1>S.H.O.S VISION SERVICE V3</h1>
    <div>
      <button onclick="setBackend('mediapipe')">MediaPipe</button>
      <button onclick="setBackend('yolo11n')">YOLO 11n</button>
      <button onclick="setBackend('moondream')">Moondream 2</button>
    </div>
    <img src="/video_feed">
    <script>
    function setBackend(b){fetch('/api/backend',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({backend:b})}).then(r=>r.json()).then(d=>console.log(d));}
    </script></body></html>
    """

@app.route("/video_feed")
def video_feed():
    return Response(vision.generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/status")
def status():
    return jsonify({
        "status": "online",
        "backend": vision.active_name,
        "frames": vision.frame_count,
        "fps": round(vision._fps, 1),
        "camera": "Picamera2" if vision.camera.using_picamera else "OpenCV",
        "meta": vision.last_meta,
    })

@app.route("/api/backend", methods=["GET"])
def api_get_backend():
    return jsonify({
        "active": vision.active_name,
        "available": {n: b.is_available() for n, b in vision.backends.items()},
        "all": list(BACKENDS),
    })

@app.route("/api/backend", methods=["POST"])
def api_set_backend():
    data = request.get_json(silent=True) or {}
    name = str(data.get("backend", "")).strip().lower()
    if name not in BACKENDS:
        return jsonify({"ok": False, "msg": f"Backend invalide. Valeurs : {BACKENDS}"}), 400
    result = vision.set_backend(name)
    return jsonify(result), (200 if result.get("ok") else 503)

@app.route("/api/last_meta")
def api_last_meta():
    with vision.frame_lock:
        return jsonify(vision.last_meta)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    try:
        t = threading.Thread(target=vision.run, name="VisionLoop", daemon=True)
        t.start()

        logger.info("Serveur Vision V3 sur http://0.0.0.0:%d", PORT_WEB)
        logger.info("Stream  : http://localhost:%d/video_feed", PORT_WEB)
        logger.info("Status  : http://localhost:%d/status", PORT_WEB)
        logger.info("Backend : POST http://localhost:%d/api/backend", PORT_WEB)

        app.run(host="0.0.0.0", port=PORT_WEB,
                debug=False, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        logger.info("Arret demande")
        vision.stop()
    except Exception as e:
        logger.error("Erreur fatale : %s", e)
        vision.stop()