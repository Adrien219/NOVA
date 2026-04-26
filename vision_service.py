import cv2
import mediapipe as mp
import threading
import json
import time
import logging
import numpy as np
from queue import Queue, Empty
from flask import Flask, Response

# --- CONFIGURATION ---
PORT_WEB = 5001
WIDTH, HEIGHT = 640, 480
FPS = 30

# Configuration du logger pour éviter les erreurs NameError
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NOVA_VISION")

app = Flask(__name__)

# ============================================================================
# LIBCAMERA - GESTION ROBUSTE DE LA PI CAMERA
# ============================================================================
class LibCameraCapture:
    def __init__(self, width=WIDTH, height=HEIGHT, fps=FPS):
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.thread_running = False
        self.frame_queue = Queue(maxsize=2)
        self.lock = threading.Lock()
        self.last_frame = None
        
        self._init_camera()
    
    def _init_camera(self):
        try:
            # Pipeline GStreamer pour libcamera
            gst_pipeline = (
                f"libcamerasrc ! "
                f"video/x-raw,width={self.width},height={self.height},framerate={self.fps}/1 ! "
                f"videoconvert ! appsink max-buffers=1 drop=true"
            )
            logger.info("🎥 Initialisation caméra via GStreamer...")
            self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            
            if not self.cap.isOpened():
                logger.warning("⚠️ GStreamer échoué, tentative /dev/video0...")
                self.cap = cv2.VideoCapture(0)
            
            return self.cap.isOpened()
        except Exception as e:
            logger.error(f"❌ Erreur caméra: {e}")
            return False
    
    def _capture_thread(self):
        logger.info("🎬 Thread de capture actif")
        self.thread_running = True
        while self.thread_running:
            if self.cap is None: break
            ret, frame = self.cap.read()
            if ret and frame is not None:
                # Mise à jour de la frame la plus récente
                with self.lock:
                    self.last_frame = frame.copy()
                # Alimentation de la queue pour le processeur IA
                if not self.frame_queue.full():
                    self.frame_queue.put(frame.copy())
            time.sleep(1/self.fps)

    def start(self):
        t = threading.Thread(target=self._capture_thread, daemon=True)
        t.start()

    def get_frame(self):
        with self.lock:
            return self.last_frame

# ============================================================================
# NOVA VISION - TRAITEMENT IA (MEDIAPIPE)
# ============================================================================
class NovaVision:
    def __init__(self, camera_provider):
        self.camera = camera_provider
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.output_frame = None
        self.lock = threading.Lock()

    def process_vision(self):
        logger.info("👁️ Analyse MediaPipe active...")
        while True:
            # On récupère la frame venant du LibCameraCapture
            frame = self.camera.get_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            # Traitement IA
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)

            # Dessin
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )

            # Mise à jour de la frame pour le stream Flask
            with self.lock:
                self.output_frame = frame

    def generate_frames(self):
        while True:
            with self.lock:
                if self.output_frame is None:
                    continue
                ret, buffer = cv2.imencode('.jpg', self.output_frame)
                if not ret: continue
                frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- INITIALISATION ---
cam = LibCameraCapture()
cam.start()
nova_vision = NovaVision(cam)

@app.route('/')
def video_feed():
    return Response(nova_vision.generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    # Thread pour l'IA
    threading.Thread(target=nova_vision.process_vision, daemon=True).start()
    
    logger.info(f"🚀 Serveur VISION démarré sur http://0.0.0.0:{PORT_WEB}")
    app.run(host='0.0.0.0', port=PORT_WEB, debug=False, threaded=True)
