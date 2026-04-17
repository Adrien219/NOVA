# -*- coding: utf-8 -*-
"""
S.H.O.S V3.0 - Application Flask CORRIGÉE pour Raspberry Pi 4B
Modifications : 
- Nettoyage des logs EngineIO/SocketIO
- Correction de l'ordre d'initialisation (NameError: app)
"""

import eventlet
eventlet.monkey_patch()

import cv2

import sys, os, time, json, logging, psutil, cv2
from pathlib import Path
from datetime import datetime
from threading import Lock, Event
from queue import Queue, Empty
import numpy as np
import time


from flask import Flask, render_template
from flask_socketio import SocketIO
from jinja2 import ChoiceLoader, FileSystemLoader
from pathlib import Path

from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
from paho.mqtt import client as mqtt_client
from jinja2 import ChoiceLoader, FileSystemLoader
# --- AJOUTE CECI AU DÉBUT (vers ligne 60) ---
from utils import ConfigManager # Assure-toi que ConfigManager est dans utils.py
config_manager = ConfigManager()

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

# Chemin absolu du projet pour éviter les erreurs de dossier courant
PROJECT_ROOT = Path(__file__).resolve().parent
template_dir = PROJECT_ROOT / "templates"
# On pointe précisément vers le dossier qui contient les sous-dossiers des modules
modules_root = PROJECT_ROOT / "plugins" / "modules"

# Initialisation de l'application Flask
app = Flask(__name__, template_folder=str(template_dir))
app.config['SECRET_KEY'] = 'shos_secret_key_2026'

# --- LE LOADER : C'est ici que la magie opère ---
# On utilise ChoiceLoader pour scanner 'templates' PUIS 'plugins/modules'
app.jinja_loader = ChoiceLoader([
    FileSystemLoader(str(template_dir)),
    FileSystemLoader(str(modules_root))
])

# Initialisation de SocketIO
socketio = SocketIO(
    app, 
    cors_allowed_origins="*", 
    async_mode='eventlet', 
    logger=False, 
    engineio_logger=False
)

# Petit check de sécurité au démarrage dans la console
if not (modules_root / "hand_control" / "interface.html").exists():
    print(f"⚠️ ATTENTION : Fichier hand_control/interface.html introuvable dans {modules_root}")
else:
    print(f"✅ Structure des modules validée.")
# ============================================================================
# CONFIGURATION PATHS & MODULES DYNAMIQUES (VERSION FINALE)
# ============================================================================
# On définit SNAPSHOTS_DIR pour le SnapshotManager
SNAPSHOTS_DIR = PROJECT_ROOT / "data" / "snapshots"
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

# Définition du chemin des modules conforme à ton arborescence : plugins/modules
PLUGINS_BASE_DIR = PROJECT_ROOT / "plugins" / "modules"

modules_dir = PLUGINS_BASE_DIR  # Alias pour la compatibilité avec le reste du code

# MISE À JOUR DU LOADER JINJA2 : Indispensable pour que Flask trouve les interface.html
# On ajoute le nouveau chemin PLUGINS_BASE_DIR à la liste des dossiers de templates
app.jinja_loader = ChoiceLoader([
    FileSystemLoader(str(template_dir)),
    FileSystemLoader(str(PLUGINS_BASE_DIR)) # Ajoutez cette ligne
])

AVAILABLE_MODULES = {}

def load_dynamic_modules():
    global AVAILABLE_MODULES
    if not PLUGINS_BASE_DIR.exists():
        PLUGINS_BASE_DIR.mkdir(parents=True)
    
    print(f"\n🔍 [SCAN] Recherche de modules dans : {PLUGINS_BASE_DIR}")
    
    # On liste les dossiers dans plugins/modules/
    for folder in os.listdir(PLUGINS_BASE_DIR):
        current_folder_path = PLUGINS_BASE_DIR / folder
        
        # On ne traite que les répertoires pour éviter les fichiers racines comme hand.py
        if current_folder_path.is_dir():
            config_file = current_folder_path / "config.json"
            
            if config_file.exists():
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                        # On indexe par l'ID du dossier (ex: 'voice_assistant')
                        AVAILABLE_MODULES[folder] = config
                        logger.info(f"📦 Module chargé : {config.get('name', folder)}")
                except Exception as e:
                    logger.error(f"❌ Erreur lecture config dans {folder}: {e}")
            else:
                # Ignore les dossiers systèmes ou de modèles sans config.json
                print(f"ℹ️  Dossier ignoré (pas de config.json) : {folder}")

    print(f"✅ [SCAN] {len(AVAILABLE_MODULES)} modules détectés et prêts.\n")

# Lancement immédiat du scan au démarrage
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
# GESTIONNAIRE MQTT UNIVERSEL (DYNAMIC DISCOVERY)
# ============================================================================

class MQTTHandler:
    def __init__(self, host='localhost', port=1883):
        self.host = host
        self.port = port
        self.active_modules = {} # Suivi des modules enregistrés en temps réel

        try:
            # Support paho-mqtt 2.0 (CallbackAPIVersion) avec fallback 1.x
            from paho.mqtt.enums import CallbackAPIVersion
            self.client = mqtt_client.Client(CallbackAPIVersion.VERSION1, "NOVA_WEB_BRIDGE")
        except:
            self.client = mqtt_client.Client("NOVA_WEB_BRIDGE")
            
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        
        try:
            self.client.connect(self.host, self.port, 60)
            self.client.loop_start()
            print("📡 [MQTT] Pont Dynamique N.O.V.A initialisé")
        except Exception as e:
            print(f"❌ [MQTT] Erreur de connexion : {e}")
            
    def _on_connect(self, client, userdata, flags, rc, properties=None):
        print(f"✅ [MQTT] Connecté au Broker avec code {rc}")
        # --- C'EST ICI QUE TOUT SE JOUE ---
        # On dit à nova.py d'écouter ces 4 topics spécifiques :
        client.subscribe([
            ("shos/sensors/normalized", 1),
            ("shos/sensors/mobile", 1),
            ("shos/sensors/esp32", 1),
            ("shos/benchmark/ping", 1)  # <--- LE VOILÀ, LE MAILLON MANQUANT !
        ])
        print("📡 [MQTT] En écoute des capteurs ET du benchmark...")
        
    def _on_message(self, client, userdata, msg):
        try:
            topic = msg.topic
            payload_raw = msg.payload.decode('utf-8')
            
            # --- LOGIQUE DE BENCHMARK (ÉCHO) ---
            if topic == "shos/benchmark/ping":
                self.client.publish("shos/benchmark/pong", payload_raw)
                return 

            # Décodage JSON pour les autres messages
            payload = json.loads(payload_raw)

            if topic == "nova/modules/register":
                mod_name = payload.get('name', 'unknown')
                self.active_modules[mod_name] = payload
                socketio.emit('module_registered', payload)
                print(f"✨ [DISCOVERY] Nouveau module : {mod_name}")

            elif "/data" in topic:
                parts = topic.split('/')
                mod_name = parts[1] if parts[0] == "shos" else parts[2]
                socketio.emit('module_update', {'module': mod_name, 'data': payload})

            elif topic == "shos/sensors/normalized":
                socketio.emit('sensor_update', payload)
            
        except Exception as e:
            print(f"❌ [MQTT] Erreur routing sur {msg.topic}: {e}")

# --- INSTANCIATION DU PONT ---
mqtt_bridge = MQTTHandler()



# ============================================================================
# ROUTES PRINCIPALES (WEB UI)
# ============================================================================

# Profil par défaut
DEFAULT_PROFILE = {'name': 'ADRIEN_ASUS', 'id': '01'}

@app.route('/')
def index():
    """Page d'accueil avec les tuiles de modules dynamiques"""
    return render_template('index.html', modules=AVAILABLE_MODULES, profile=DEFAULT_PROFILE)

@app.route('/dashboard')
def dashboard(): return render_template('dashboard.html')

@app.route('/diagnostic')
def diagnostic(): return render_template('diagnostic.html')

@app.route('/settings')
def settings(): return render_template('settings.html')

@app.route('/mobile')
def mobile(): return render_template('mobile.html')

@app.route('/diagnostic_ultimate')
def diagnostic_ultimate(): 
    return render_template('diagnostic_ultimate.html')

@app.route('/profiles')
def profiles_page():
    # On passe les profils à la page pour qu'elle puisse les afficher
    config = config_manager.load_config()
    return render_template('profiles.html', profiles=config.get("profiles", {}))

@app.route('/hud')
def hud(): return render_template('hud.html', profile=DEFAULT_PROFILE)

@app.route('/user_interface')
def user_interface(): 
    return render_template('user_interface.html', profile=DEFAULT_PROFILE)

# ============================================================================
# GESTION DES PROFILS (API & MANAGER)
# ============================================================================

@app.route('/profile_manager')
def profile_manager():
    """Interface de création et gestion des profils"""
    try:
        return render_template('profile_manager.html', modules=AVAILABLE_MODULES, profile=DEFAULT_PROFILE)
    except Exception as e:
        logger.error(f"❌ Erreur Profile Manager: {e}")
        return f"Erreur Profile Manager: {str(e)}", 500

@app.route('/api/get_profiles', methods=['GET'])
def get_profiles():
    """Récupère la liste des profils enregistrés"""
    try:
        config = config_manager.load_config()
        return jsonify({
            "profiles": config.get("profiles", {}),
            "current": config.get("current_active_profile", "")
        })
    except Exception as e:
        logger.error(f"❌ Erreur API get_profiles: {e}")
        return jsonify({"profiles": {}, "current": ""}), 500

@app.route('/api/activate_profile', methods=['POST'])
def api_activate_profile():
    """Active un profil spécifique"""
    try:
        data = request.json
        profile_id = data.get('profile_id')
        success = config_manager.activate_profile(profile_id)
        if success:
            logger.info(f"🚀 Profil activé via API : {profile_id}")
            return jsonify({"status": "success"})
        return jsonify({"status": "error", "message": "Échec activation"}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/save_profile', methods=['POST'])
def save_profile():
    """Sauvegarde un nouveau profil et notifie le système via MQTT"""
    try:
        data = request.json
        if not data:
            return jsonify({"status": "error", "message": "Données vides"}), 400

        profile_name = data.get('name')
        profile_id = profile_name.lower().replace(" ", "_")

        # 1. Enregistrement dans config.json
        config = config_manager.load_config()
        config["profiles"][profile_id] = {
            "name": profile_name,
            "main_module": data.get('main_module'),
            "secondary_modules": data.get('secondary_modules', [])
        }
        config_manager.save_config(config)

        # 2. Notification MQTT pour changement immédiat sur le HUD/Casque
        profile_payload = {
            "action": "SWITCH_PROFILE",
            "profile_name": profile_name,
            "allowed_modules": {data.get('main_module'): 1}
        }
        mqtt_bridge.client.publish("shos/system/profile/switch", json.dumps(profile_payload))

        return jsonify({"status": "success", "profile": profile_name})
    except Exception as e:
        logger.error(f"❌ Erreur critique save_profile : {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# ============================================================================
# ROUTES MODULES DYNAMIQUES 
# ============================================================================
# On utilise <path:module_id> pour capturer l'ID du module, même s'il contient des slashs
@app.route('/module/<path:module_id>')
def universal_module_route(module_id):
    """Charge l'interface d'un module en utilisant le loader configuré"""
    
    # Nettoyage de l'ID (au cas où il y a un slash à la fin)
    module_id = module_id.strip('/')
    logger.info(f"🔍 S.H.O.S demande l'accès au module : '{module_id}'")
    
    if module_id in AVAILABLE_MODULES:
        config = AVAILABLE_MODULES[module_id]
        # On récupère le nom du template depuis le config.json, sinon par défaut 'interface.html'
        template_name = config.get('template', 'interface.html')
        
        # Le chemin exact que Jinja va chercher (ex: "hand_control/interface.html")
        # Rappel : Votre ChoiceLoader doit inclure le dossier "plugins/modules"
        template_path = f"{module_id}/{template_name}"
        logger.info(f"📂 Tentative de chargement du fichier : {template_path}")
        
        try:
            return render_template(template_path, config=config, profile=DEFAULT_PROFILE)
        except Exception as e:
            logger.error(f"❌ Erreur de rendu pour {module_id} : {e}")
            return (f"<div style='color:red; font-family:sans-serif; padding:20px;'>"
                    f"<h1>Erreur d'affichage</h1>"
                    f"<p>Flask n'arrive pas à trouver le fichier : <b>{template_path}</b></p>"
                    f"<p>Vérifie que dans ton dossier <b>plugins/modules/</b>, tu as bien un dossier nommé <b>{module_id}</b> contenant un fichier <b>{template_name}</b>.</p>"
                    f"</div>"), 500
            
    logger.warning(f"⚠️ Module '{module_id}' introuvable dans AVAILABLE_MODULES.")
    return (f"<div style='color:orange; font-family:sans-serif; padding:20px;'>"
            f"<h1>Erreur 404</h1>"
            f"<p>Le module '<b>{module_id}</b>' n'est pas chargé par le système S.H.O.S.</p>"
            f"</div>"), 404

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

# --- LOGIQUE DE STREAMING VIDÉO ---

def generate_frames():
    """Capture le flux de la caméra et le prépare pour le web"""
    # 0 est l'index de ta caméra USB ou Raspberry Pi
    camera = cv2.VideoCapture(0) 
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # On compresse l'image en JPEG pour qu'elle soit légère
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            # On envoie l'image au format "flux continu" (MJPEG)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Cette route distribue le flux vidéo à ton HTML"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')





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
