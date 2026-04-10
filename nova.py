# -*- coding: utf-8 -*-
"""
S.H.O.S V3.0 - Application Flask CORRIGÉE pour Raspberry Pi 4B
Modifications : 
- Nettoyage des logs EngineIO/SocketIO
- Correction de l'ordre d'initialisation (NameError: app)
"""

#import eventlet
#eventlet.monkey_patch()

import cv2

import sys, os, time, json, logging, psutil, cv2
from pathlib import Path
from datetime import datetime
from threading import Lock, Event
from queue import Queue, Empty
import numpy as np
import time

from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO, emit
from paho.mqtt import client as mqtt_client
from jinja2 import ChoiceLoader, FileSystemLoader

# ============================================================================
# CONFIGURATION LOGGING
# ============================================================================
logging.getLogger('socketio').setLevel(logging.ERROR)
logging.getLogger('engineio').setLevel(logging.ERROR)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("SHOS_V3.0")

# ============================================================================
# INITIALISATION FLASK & CONFIGURATION MULTI-TEMPLATES
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
template_dir = PROJECT_ROOT / "templates"
modules_dir = PROJECT_ROOT / "modules"

app = Flask(__name__, template_folder=str(template_dir))
app.config['SECRET_KEY'] = 'shos_secret_key_2026'

# SOLUTION MASTER : On autorise Flask à chercher dans /templates ET dans /modules
app.jinja_loader = ChoiceLoader([
    FileSystemLoader(str(template_dir)),
    FileSystemLoader(str(modules_dir))
])

socketio = SocketIO(
    app, 
    cors_allowed_origins="*", 
    async_mode='eventlet', 
    logger=False, 
    engineio_logger=False
)

# ============================================================================
# CONFIGURATION PATHS & MODULES DYNAMIQUES
# ============================================================================
# On définit SNAPSHOTS_DIR en majuscule pour que SnapshotManager le trouve
SNAPSHOTS_DIR = PROJECT_ROOT / "data" / "snapshots"
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

# On garde aussi la version minuscule si ton code en a besoin ailleurs
snapshots_dir = SNAPSHOTS_DIR 

AVAILABLE_MODULES = {}

def load_dynamic_modules():
    global AVAILABLE_MODULES
    if not modules_dir.exists():
        modules_dir.mkdir()
    
    for folder in os.listdir(modules_dir):
        config_file = modules_dir / folder / "config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    AVAILABLE_MODULES[folder] = config
                    logger.info(f"📦 Module chargé : {config['name']}")
            except Exception as e:
                logger.error(f"❌ Erreur config {folder}: {e}")

load_dynamic_modules()

# ============================================================================
# YOLOV8N - DÉTECTION D'OBJETS
# ============================================================================
class YOLOv8nDetector:
    """
    Détection d'objets légère pour Raspberry Pi
    Utilise le modèle YOLOv8n (nano = très rapide)
    """
    
    def __init__(self, model_path="models/yolov8n.pt", confidence=0.5):
        """
        Args:
            model_path: Chemin vers yolov8n.pt
            confidence: Seuil de confiance (0-1)
        """
        self.confidence = confidence
        self.model = None
        self.device = "cpu"  # Raspberry Pi n'a pas de GPU
        
        try:
            from ultralytics import YOLO
            
            if os.path.exists(model_path):
                logger.info(f"✅ Chargement YOLOv8n: {model_path}")
                self.model = YOLO(model_path)
                self.model.to(self.device)
            else:
                logger.warning(f"⚠️ Modèle YOLOv8n non trouvé: {model_path}")
                logger.info("   Télécharger: python -m pip install ultralytics")
                logger.info("   Puis: from ultralytics import YOLO; YOLO('yolov8n.pt')")
        except ImportError:
            logger.warning("⚠️ ultralytics pas installé. Détection désactivée.")
            self.model = None
    
    def detect(self, frame):
        """
        Détecte les objets dans une frame
        
        Returns:
            (frame_annotated, detections_dict)
            detections_dict = {
                "objects": [{"class": "person", "confidence": 0.95, "box": [x1,y1,x2,y2]}, ...],
                "count": 2,
                "inference_time_ms": 45
            }
        """
        if self.model is None:
            return frame, {"objects": [], "count": 0, "inference_time_ms": 0}
        
        try:
            start_time = time.time()
            results = self.model.predict(frame, conf=self.confidence, verbose=False)
            inference_time = (time.time() - start_time) * 1000
            
            detections = []
            annotated_frame = frame.copy()
            
            if results and len(results) > 0:
                result = results[0]
                
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = self.model.names[class_id]
                        
                        detections.append({
                            "class": class_name,
                            "confidence": round(confidence, 3),
                            "box": [x1, y1, x2, y2],
                            "class_id": class_id
                        })
                        
                        # Dessiner la boîte
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{class_name} {confidence:.2f}"
                        cv2.putText(annotated_frame, label, (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            return annotated_frame, {
                "objects": detections,
                "count": len(detections),
                "inference_time_ms": round(inference_time, 2)
            }
        
        except Exception as e:
            logger.error(f"❌ Erreur YOLOv8n: {e}")
            return frame, {"objects": [], "count": 0, "inference_time_ms": 0, "error": str(e)}


# ============================================================================
# LIBCAMERA - GESTION ROBUSTE DE LA PI CAMERA
# ============================================================================
class LibCameraCapture:
    """
    Capture vidéo compatible avec libcamera (Bullseye/Bookworm)
    Utilise GStreamer comme pipeline pour éviter les blocages OpenCV
    """
    
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.thread_running = False
        self.frame_queue = Queue(maxsize=2)  # Garder au max 2 frames
        self.lock = Lock()
        self.last_frame = None
        self.frame_count = 0
        self.error_count = 0
        self.max_errors = 10
        
        self._init_camera()
    
    def _init_camera(self):
        """Initialise la caméra avec GStreamer/libcamera"""
        try:
            # Pipeline GStreamer compatible libcamera
            # Alternative 1: Utiliser libcamera directement
            gst_pipeline = (
                f"libcamerasrc ! "
                f"video/x-raw,width={self.width},height={self.height},framerate={self.fps}/1 ! "
                f"videoconvert ! appsink max-buffers=1 drop=true"
            )
            
            logger.info("🎥 Initialisation caméra Pi avec GStreamer...")
            self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            
            if not self.cap.isOpened():
                logger.warning("⚠️ GStreamer échoué, tentative cv2.VideoCapture(0)...")
                self.cap = cv2.VideoCapture(0)
                
                if not self.cap.isOpened():
                    logger.error("❌ Caméra non disponible!")
                    self.cap = None
                    return False
            
            # Configurer les propriétés
            if self.cap:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)
                
                logger.info(f"✅ Caméra Pi initialisée: {self.width}x{self.height} @ {self.fps} FPS")
            
            return True
        
        except Exception as e:
            logger.error(f"❌ Erreur initialisation caméra: {e}")
            self.cap = None
            return False
    
    def read(self, timeout=1.0):
        """
        Lit une frame de manière non-bloquante
        
        Returns:
            (success, frame) ou (False, None) si erreur
        """
        try:
            frame = self.frame_queue.get(timeout=timeout)
            return True, frame
        except Empty:
            if self.last_frame is not None:
                return True, self.last_frame
            return False, None
    
    def _capture_thread(self):
        """Thread de capture (peut bloquer sans affecter Flask)"""
        logger.info("🎬 Thread de capture démarré")
        self.thread_running = True
        
        while self.thread_running:
            try:
                if self.cap is None or not self.cap.isOpened():
                    self.error_count += 1
                    if self.error_count > self.max_errors:
                        logger.error("❌ Trop d'erreurs caméra, arrêt capture")
                        break
                    time.sleep(1)
                    continue
                
                ret, frame = self.cap.read()
                
                if not ret:
                    self.error_count += 1
                    time.sleep(0.1)
                    continue
                
                self.error_count = 0
                self.frame_count += 1
                
                with self.lock:
                    self.last_frame = frame
                
                # Ajouter à queue (drop si full)
                try:
                    self.frame_queue.put(frame, block=False)
                except:
                    pass  # Queue full, drop frame
                
                eventlet.sleep(1 / self.fps)
            
            except Exception as e:
                logger.error(f"❌ Erreur capture: {e}")
                self.error_count += 1
                eventlet.sleep(1)
    
    def start(self):
        """Démarrer le thread de capture"""
        if self.cap is None:
            logger.error("❌ Caméra non initialisée")
            return False
        
        from threading import Thread
        thread = Thread(target=self._capture_thread, daemon=True)
        thread.start()
        return True
    
    def stop(self):
        """Arrêter le thread de capture"""
        self.thread_running = False
        if self.cap:
            self.cap.release()
    
    def get_stats(self):
        """Retourner les stats de capture"""
        return {
            "frames_captured": self.frame_count,
            "errors": self.error_count,
            "has_frame": self.last_frame is not None,
            "frame_shape": self.last_frame.shape if self.last_frame is not None else None
        }


# ============================================================================
# GESTURE RECOGNITION - DÉTECTION DE GESTES
# ============================================================================
class GestureDetector:
    """Détecteur de gestes avec MediaPipe"""
    
    def __init__(self):
        self.current_gesture = "NONE"
        self.last_gesture = "NONE"
        
        try:
            import mediapipe as mp
            self.mp = mp
            self.hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            self.available = True
            logger.info("✅ MediaPipe chargé")
        except ImportError:
            self.available = False
            logger.warning("⚠️ MediaPipe non disponible - gestes désactivés")
    
    def detect(self, frame):
        """
        Détecte un geste
        
        Returns:
            gesture_name (str): "NONE", "THUMBS_UP", "PEACE", "PALM", "FIST", etc.
        """
        if not self.available:
            return "NONE", frame
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
                hand = results.multi_hand_landmarks[0]
                gesture = self._classify_gesture(hand)
                self.current_gesture = gesture
                return gesture, frame
            
            self.current_gesture = "NONE"
            return "NONE", frame
        
        except Exception as e:
            logger.error(f"❌ Erreur détection geste: {e}")
            return "NONE", frame
    
    def _classify_gesture(self, hand):
        """Classifie un geste basé sur les landmarks"""
        # Landmarks importants
        thumb_tip = hand.landmark[4]
        index_tip = hand.landmark[8]
        middle_tip = hand.landmark[12]
        ring_tip = hand.landmark[16]
        pinky_tip = hand.landmark[20]
        
        palm = hand.landmark[0]
        
        # Calcul distances
        thumb_up = thumb_tip.y < palm.y
        index_up = index_tip.y < palm.y
        middle_up = middle_tip.y < palm.y
        ring_up = ring_tip.y < palm.y
        pinky_up = pinky_tip.y < palm.y
        
        # Détection simples
        if thumb_up and not (index_up or middle_up or ring_up or pinky_up):
            return "THUMBS_UP"
        
        if index_up and middle_up and not (ring_up or pinky_up):
            return "PEACE"
        
        if index_up and not (middle_up or ring_up or pinky_up):
            return "POINT"
        
        if thumb_up and index_up and middle_up and ring_up and pinky_up:
            return "PALM"
        
        if not (thumb_up or index_up or middle_up or ring_up or pinky_up):
            return "FIST"
        
        return "UNKNOWN"


# ============================================================================
# SNAPSHOT MANAGER - CAPTURE PHOTOS
# ============================================================================
class SnapshotManager:
    """Gère la capture de snapshots avec horodatage"""
    
    def __init__(self, snapshots_dir=SNAPSHOTS_DIR):
        self.snapshots_dir = Path(snapshots_dir)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📸 Snapshots dir: {self.snapshots_dir}")
    
    def capture(self, frame, gesture="UNKNOWN", metadata=None):
        """
        Capture un snapshot
        
        Args:
            frame: Frame OpenCV (BGR)
            gesture: Geste détecté
            metadata: Dict optionnel avec infos additionnelles
        
        Returns:
            filepath (Path) ou None
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"snapshot_{gesture}_{timestamp}.jpg"
            filepath = self.snapshots_dir / filename
            
            # Encodage JPEG
            success, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            if success:
                with open(filepath, 'wb') as f:
                    f.write(encoded.tobytes())
                
                # Sauvegarde metadata JSON
                metadata_file = filepath.with_suffix('.json')
                metadata_dict = {
                    "timestamp": datetime.now().isoformat(),
                    "gesture": gesture,
                    "file": str(filepath),
                    "frame_shape": frame.shape if frame is not None else None,
                    **(metadata or {})
                }
                with open(metadata_file, 'w') as f:
                    json.dump(metadata_dict, f, indent=2)
                
                logger.info(f"📸 Snapshot capturé: {filename}")
                return filepath
        
        except Exception as e:
            logger.error(f"❌ Erreur snapshot: {e}")
        
        return None
    
    def get_recent_snapshots(self, limit=10):
        """Récupère les derniers snapshots"""
        snapshots = sorted(
            self.snapshots_dir.glob("snapshot_*.jpg"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )[:limit]
        
        return [
            {
                "filename": s.name,
                "timestamp": s.stat().st_mtime,
                "url": f"/snapshots/{s.name}"
            }
            for s in snapshots
        ]


# ============================================================================
# FLASK APPLICATION
# ============================================================================

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max

socketio = SocketIO(
    app,
    async_mode='threading', # Change eventlet par threading
    cors_allowed_origins="*",
    logger=False,           # Mets sur False pour gagner en vitesse
    engineio_logger=False   # Mets sur False
)

# Initialiser les modules
logger.info("🚀 Initialisation S.H.O.S V3.0...")

pi_camera = LibCameraCapture(width=640, height=480, fps=30)
pi_camera.start()

gesture_detector = GestureDetector()
yolo_detector = YOLOv8nDetector(model_path="models/yolov8n.pt", confidence=0.5)
snapshot_manager = SnapshotManager()

# Variables globales
last_frame_pi = None
current_gesture = "NONE"
detection_stats = {"objects": 0, "inference_time_ms": 0}
camera_stats = {"frames": 0, "errors": 0}


# ============================================================================
# GESTIONNAIRE MQTT (RE-AJOUTÉ)
# ============================================================================
# --- AJOUTE CE BLOC JUSTE AVANT TES ROUTES ---

class MQTTHandler:
    def __init__(self, host='localhost', port=1883):
        self.host = host
        self.port = port
        # Version compatible avec ton installation
        try:
            self.client = mqtt_client.Client(mqtt_client.CallbackAPIVersion.VERSION2, "SHOS_V3_BRIDGE")
        except:
            self.client = mqtt_client.Client("SHOS_V3_BRIDGE")
            
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        
        try:
            self.client.connect(self.host, self.port, 60)
            self.client.loop_start()
            print("📡 [MQTT] Pont MQTT-Web initialisé") # Utilise print pour être sûr de le voir
        except Exception as e:
            print(f"❌ [MQTT] Erreur de connexion: {e}")
            
    def _on_connect(self, client, userdata, flags, rc, properties=None):
        print(f"✅ [MQTT] Connecté au Broker avec code {rc}")
        client.subscribe([
            ("shos/sensors/normalized", 1),
            ("shos/sensors/mobile", 1),
            ("shos/sensors/esp32", 1)
        ])
        
    def _on_message(self, client, userdata, msg):
            try:
                # Décodage du JSON envoyé par le téléphone
                payload = json.loads(msg.payload.decode('utf-8'))
                
                # CAS 1 : Données de l'Arduino (déjà là)
                if msg.topic == "shos/sensors/normalized":
                    socketio.emit('sensor_update', payload)
                
                # CAS 2 : Données du TÉLÉPHONE (Ajout/Vérification)
                elif msg.topic == "shos/sensors/mobile":
                    # On envoie un événement spécifique 'mobile_update'
                    socketio.emit('mobile_update', payload)
                    # Optionnel : log pour débugger au début
                    # print(f"📲 [MOBILE] Accel: {payload.get('accel_x')}")

                print(f"📥 [MQTT] Donnée reçue sur {msg.topic}")
            except Exception as e:
                print(f"❌ [MQTT] Erreur parsing sur {msg.topic}: {e}")

# --- CETTE LIGNE EST LA PLUS IMPORTANTE ---
mqtt_bridge = MQTTHandler()



# ============================================================================
# ROUTES PRINCIPALES (WEB UI)
# ============================================================================

# Profil par défaut
DEFAULT_PROFILE = {'name': 'ADRIEN_ASUS', 'id': '01'}

@app.route('/')
def index():
    # On passe AVAILABLE_MODULES pour que l'accueil affiche les tuiles dynamiquement
    return render_template('index.html', modules=AVAILABLE_MODULES, profile=DEFAULT_PROFILE)

from flask import render_template_string

@app.route('/module/<module_id>')
def universal_module_route(module_id):
    if module_id in AVAILABLE_MODULES:
        config = AVAILABLE_MODULES[module_id]
        # On construit le chemin ABSOLU vers le fichier
        file_path = modules_dir / module_id / config.get('template', 'interface.html')
        
        if not file_path.exists():
            logger.error(f"❌ Fichier manquant : {file_path}")
            return f"Fichier introuvable : {file_path}", 404
            
        try:
            # On lit le contenu du fichier HTML directement
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # On utilise render_template_string pour injecter les variables
            return render_template_string(html_content, 
                                        config=config, 
                                        profile={'name': 'Adrien'})
        except Exception as e:
            logger.error(f"❌ Erreur lecture module {module_id}: {e}")
            return f"Erreur de rendu : {str(e)}", 500
            
    return "Module non trouvé", 404

@app.route('/profile_manager')
def profile_manager():
    try:
        return render_template('profile_manager.html', modules=AVAILABLE_MODULES, profile=DEFAULT_PROFILE)
    except Exception as e:
        return f"Erreur Profile Manager: {str(e)}", 500

@app.route('/hud')
def hud():
    return render_template('hud.html', profile=DEFAULT_PROFILE)

@app.route('/user_interface')
def user_interface():
    return render_template('user_interface.html', profile=DEFAULT_PROFILE)

@app.route('/save_profile', methods=['POST'])
def save_profile():
    from flask import request
    data = request.json
    logger.info(f"👤 Nouveau profil reçu : {data.get('name')}")
    return jsonify({"status": "success"})

# Routes statiques restantes
@app.route('/dashboard')
def dashboard(): return render_template('dashboard.html')

@app.route('/diagnostic')
def diagnostic(): return render_template('diagnostic.html')

@app.route('/settings')
def settings(): return render_template('settings.html')

@app.route('/mobile')
def mobile(): return render_template('mobile.html')

# ============================================================================
# FLUX VIDÉO & API
# ============================================================================

@app.route('/video_feed_pi')
def video_feed_pi():
    def gen_frames():
        while True:
            # On ne fait QUE lire et envoyer, l'IA sera traitée ailleurs
            ret, frame = pi_camera.read(timeout=0.2)
            if not ret or frame is None:
                time.sleep(0.1)
                continue

            try:
                # Compression légère pour la fluidité
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                time.sleep(0.04) # Limite à 25 FPS
            except Exception as e:
                continue
    
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ============================================================================
# SOCKET.IO EVENTS
# ============================================================================

@socketio.on('connect')
def handle_connect():
    """Client connecté"""
    logger.info(f"✅ Client connecté")
    emit('connection_response', {
        'data': 'Connecté à S.H.O.S V3.0',
        'timestamp': datetime.now().isoformat()
    })

@socketio.on('request_camera_status')
def handle_camera_status():
    """Demande du status caméra"""
    emit('camera_status', {
        "gesture": current_gesture,
        "detection_stats": detection_stats,
        "pi_camera": pi_camera.get_stats(),
        "timestamp": datetime.now().isoformat()
    })

@socketio.on('request_snapshot')
def handle_snapshot_request():
    """Demande de snapshot manuel"""
    if last_frame_pi is not None:
        from io import BytesIO
        frame_array = np.frombuffer(last_frame_pi, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        
        filepath = snapshot_manager.capture(
            frame,
            gesture="MANUAL",
            metadata={"triggered_by": "user"}
        )
        
        emit('snapshot_captured', {
            'filepath': str(filepath),
            'timestamp': datetime.now().isoformat()
        })


# ============================================================================
# BACKGROUND TASKS
# ============================================================================

def background_monitoring():
    """Monitoring système et caméra"""
    while True:
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            ram_percent = psutil.virtual_memory().percent
            
            socketio.emit('system_stats', {
                'cpu': cpu_percent,
                'ram': ram_percent,
                'timestamp': datetime.now().isoformat()
            })
            
            time.sleep(5)
        
        except Exception as e:
            logger.error(f"❌ Erreur monitoring: {e}")
            eventlet.sleep(5)


# ============================================================================
# MAIN
# ============================================================================


if __name__ == '__main__':
    logger.info("="*80)
    logger.info("🎬 S.H.O.S V3.0 - Démarrage du serveur Flask")
    logger.info("="*80)
    
    # Démarrer monitoring
    socketio.start_background_task(background_monitoring)
    
    try:
        socketio.run(
            app,
            host='0.0.0.0',
            port=5000,
            debug=False,
            use_reloader=False,
            log_output=True
        )
    
    except KeyboardInterrupt:
        logger.info("⛔ Arrêt du serveur...")
    
    finally:
        pi_camera.stop()
        logger.info("✅ Serveur arrêté")
