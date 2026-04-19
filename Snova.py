#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S.H.O.S V3.0 - SOLUTION FINALE & COMPLÈTE
✅ Vidéo fonctionnelle
✅ Capteurs affichés
✅ Socket.IO corrigé
✅ Monitoring système
"""

import eventlet
eventlet.monkey_patch()

import cv2, sys, os, time, json, logging, psutil
from pathlib import Path
from datetime import datetime
from threading import Lock, Event, Thread
from queue import Queue, Empty
import numpy as np

from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
from paho.mqtt import client as mqtt_client
from jinja2 import ChoiceLoader, FileSystemLoader

# ============================================================================
# CONFIG
# ============================================================================
logging.getLogger('socketio').setLevel(logging.ERROR)
logging.getLogger('engineio').setLevel(logging.ERROR)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("NOVA")

PROJECT_ROOT = Path(__file__).resolve().parent
template_dir = PROJECT_ROOT / "templates"
modules_root = PROJECT_ROOT / "plugins" / "modules"

app = Flask(__name__, template_folder=str(template_dir))
app.config['SECRET_KEY'] = 'shos_secret_key_2026'

app.jinja_loader = ChoiceLoader([
    FileSystemLoader(str(template_dir)),
    FileSystemLoader(str(modules_root))
])

socketio = SocketIO(
    app, 
    cors_allowed_origins="*", 
    async_mode='eventlet', 
    logger=False, 
    engineio_logger=False,
    ping_timeout=60,
    ping_interval=25
)

SNAPSHOTS_DIR = PROJECT_ROOT / "data" / "snapshots"
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# VARIABLES GLOBALES
# ============================================================================
global_frame = None
frame_lock = Lock()
current_gesture = "NONE"
detection_stats = {"objects": 0, "inference_time_ms": 0}

# ============================================================================
# LIBCAMERA CAPTURE
# ============================================================================
class LibCameraCapture:
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.thread_running = False
        self.frame_queue = Queue(maxsize=2)
        self.lock = Lock()
        self.last_frame = None
        self.frame_count = 0
        self.error_count = 0
        self.max_errors = 10
        self._init_camera()
    
    def _init_camera(self):
        try:
            gst_pipeline = (
                f"libcamerasrc ! "
                f"video/x-raw,width={self.width},height={self.height},framerate={self.fps}/1 ! "
                f"videoconvert ! appsink max-buffers=1 drop=true"
            )
            
            logger.info("🎥 GStreamer...")
            self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            
            if not self.cap.isOpened():
                logger.warning("⚠️ GStreamer fallback cv2.VideoCapture(0)...")
                self.cap = cv2.VideoCapture(0)
                
                if not self.cap.isOpened():
                    logger.error("❌ Caméra non disponible!")
                    self.cap = None
                    return False
            
            if self.cap:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)
                logger.info(f"✅ Caméra: {self.width}x{self.height} @ {self.fps} FPS")
            
            return True
        except Exception as e:
            logger.error(f"❌ Erreur caméra: {e}")
            self.cap = None
            return False
    
    def read(self, timeout=1.0):
        try:
            frame = self.frame_queue.get(timeout=timeout)
            return True, frame
        except Empty:
            if self.last_frame is not None:
                return True, self.last_frame
            return False, None
    
    def _capture_thread(self):
        logger.info("🎬 Thread capture démarré")
        self.thread_running = True
        
        while self.thread_running:
            try:
                if self.cap is None or not self.cap.isOpened():
                    self.error_count += 1
                    if self.error_count > self.max_errors:
                        logger.error("❌ Trop d'erreurs caméra")
                        break
                    eventlet.sleep(1)  # ⬅️ FIX: eventlet.sleep
                    continue
                
                ret, frame = self.cap.read()
                
                if not ret:
                    self.error_count += 1
                    eventlet.sleep(0.1)  # ⬅️ FIX: eventlet.sleep
                    continue
                
                self.error_count = 0
                self.frame_count += 1
                
                with self.lock:
                    self.last_frame = frame
                    global global_frame
                    with frame_lock:
                        global_frame = frame.copy()
                
                try:
                    self.frame_queue.put(frame, block=False)
                except:
                    pass
                
                eventlet.sleep(1 / self.fps)
            
            except Exception as e:
                logger.error(f"❌ Erreur capture: {e}")
                self.error_count += 1
                eventlet.sleep(1)  # ⬅️ FIX: eventlet.sleep
    
    def start(self):
        if self.cap is None:
            logger.error("❌ Caméra non initialisée")
            return False
        
        thread = Thread(target=self._capture_thread, daemon=True)
        thread.start()
        return True
    
    def stop(self):
        self.thread_running = False
        if self.cap:
            self.cap.release()
    
    def get_stats(self):
        return {
            "frames_captured": self.frame_count,
            "errors": self.error_count,
            "has_frame": self.last_frame is not None
        }

# ============================================================================
# MQTT HANDLER
# ============================================================================
class MQTTHandler:
    def __init__(self, broker="localhost", port=1883):
        try:
            try:
                from paho.mqtt.enums import CallbackAPIVersion
                self.client = mqtt_client.Client(mqtt_client.CallbackAPIVersion.VERSION2, "NOVA_WEB_BRIDGE")
            except:
                self.client = mqtt_client.Client("NOVA_WEB_BRIDGE")
            
            self.client.on_connect = self.on_connect
            self.client.on_message = self.on_message
            self.client.connect(broker, port, keepalive=60)
            self.client.loop_start()
            print("✅ [MQTT] Connecté")
        except Exception as e:
            print(f"❌ [MQTT] Erreur: {e}")
    
    def on_connect(self, client, userdata, flags, rc, properties=None):
        print(f"✅ [MQTT] Code: {rc}")
        self.client.subscribe([
            ("shos/sensors/normalized", 1),
            ("shos/sensors/mobile", 1),
            ("shos/sensors/esp32", 1),
        ])
    
    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            topic = msg.topic
            
            if "sensors" in topic:
                # ⬅️ FIX: broadcast=True
                socketio.emit('sensor_update', payload, broadcast=True)
                logger.info(f"📊 Capteurs: {payload}")
        
        except Exception as e:
            logger.error(f"❌ MQTT: {e}")

mqtt_bridge = MQTTHandler()
pi_camera = LibCameraCapture(width=640, height=480, fps=30)
pi_camera.start()

# ============================================================================
# ROUTES FLASK
# ============================================================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/hud')
def hud():
    return render_template('user_interface.html')

def generate_frames():
    global global_frame
    logger.info("📡 Flux vidéo connecté")
    while True:
        if global_frame is not None:
            with frame_lock:
                ret, buffer = cv2.imencode('.jpg', global_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                if not ret:
                    continue
                frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        eventlet.sleep(0.04)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ============================================================================
# SOCKET.IO
# ============================================================================
@socketio.on('connect')
def handle_connect():
    logger.info("✅ Client connecté")
    emit('connection_response', {
        'data': 'Connecté à NOVA',
        'timestamp': datetime.now().isoformat()
    })

@socketio.on('request_camera_status')
def handle_camera_status():
    emit('camera_status', {
        "gesture": current_gesture,
        "pi_camera": pi_camera.get_stats(),
        "timestamp": datetime.now().isoformat()
    })

# ============================================================================
# BACKGROUND MONITORING - ⬅️ COMPLÈTEMENT CORRIGÉ
# ============================================================================
def background_monitoring():
    """Monitoring système - BROADCAST À TOUS LES CLIENTS"""
    while True:
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            ram_percent = psutil.virtual_memory().percent
            disk_percent = psutil.disk_usage('/').percent
            
            # Température RPi
            try:
                with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                    temp_raw = int(f.read()) / 1000
            except:
                temp_raw = 0
            
            # Uptime
            uptime = psutil.boot_time()
            
            # ⬅️ FIX: broadcast=True ET sys_update (pas system_stats)
            socketio.emit('sys_update', {
                'cpu': cpu_percent,
                'ram': ram_percent,
                'disk': disk_percent,
                'temp': temp_raw,
                'uptime': time.time() - uptime,
                'timestamp': datetime.now().isoformat()
            }, broadcast=True)
            
            eventlet.sleep(5)  # ⬅️ FIX: eventlet.sleep
        
        except Exception as e:
            logger.error(f"❌ Monitoring: {e}")
            eventlet.sleep(5)

# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    logger.info("="*80)
    logger.info("🚀 NOVA - DÉMARRAGE")
    logger.info("="*80)
    
    socketio.start_background_task(background_monitoring)
    
    try:
        socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        logger.info("⛔ Arrêt...")
    finally:
        pi_camera.stop()
        logger.info("✅ Arrêté")
