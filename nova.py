# -*- coding: utf-8 -*-
"""
S.H.O.S V3.0 - Application Flask pour Raspberry Pi 4B
"""

import subprocess
import signal
import sys, os, time, json, logging, psutil
#import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from threading import Lock

from flask import Flask, render_template, Response, jsonify, request, render_template_string, stream_with_context
from flask_socketio import SocketIO, emit
from paho.mqtt import client as mqtt_client
from jinja2 import ChoiceLoader, FileSystemLoader
from utils import ConfigManager

# paho-mqtt API v2 detection
try:
    from paho.mqtt.enums import CallbackAPIVersion as _MQTTCallbackAPI
    _MQTT_V2 = True
except ImportError:
    _MQTT_V2 = False

config_manager = ConfigManager()

global_frame = None
frame_lock = Lock()

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
# INITIALISATION FLASK, PATHS & CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
template_dir = PROJECT_ROOT / "templates"
PLUGINS_BASE_DIR = PROJECT_ROOT / "plugins" / "modules"
modules_dir = PLUGINS_BASE_DIR
SNAPSHOTS_DIR = PROJECT_ROOT / "data" / "snapshots"
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder=str(template_dir))
app.config['SECRET_KEY'] = os.environ.get('SHOS_SECRET_KEY', 'shos_secret_key_2026')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

app.jinja_loader = ChoiceLoader([
    FileSystemLoader(str(template_dir)),
    FileSystemLoader(str(PLUGINS_BASE_DIR))
])

socketio = SocketIO(
    app,
    async_mode='threading',
    cors_allowed_origins="*",
    logger=False,
    engineio_logger=False,
    transports=['websocket', 'polling']
)

if not (PLUGINS_BASE_DIR / "hand_control" / "interface.html").exists():
    print(f"⚠️ ATTENTION : Fichier hand_control/interface.html introuvable dans {PLUGINS_BASE_DIR}")
else:
    print(f"✅ Structure des modules validée.")

# ============================================================================
# MODULES DYNAMIQUES
# ============================================================================

AVAILABLE_MODULES = {}

def load_dynamic_modules():
    global AVAILABLE_MODULES
    if not PLUGINS_BASE_DIR.exists():
        PLUGINS_BASE_DIR.mkdir(parents=True)

    print(f"\n🔍 [SCAN] Recherche de modules dans : {PLUGINS_BASE_DIR}")

    for folder in os.listdir(PLUGINS_BASE_DIR):
        current_folder_path = PLUGINS_BASE_DIR / folder

        if current_folder_path.is_dir():
            config_file = current_folder_path / "config.json"

            if config_file.exists():
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                        AVAILABLE_MODULES[folder] = config
                        logger.info(f"📦 Module chargé : {config.get('name', folder)}")
                except Exception as e:
                    logger.error(f"❌ Erreur lecture config dans {folder}: {e}")
            else:
                print(f"ℹ️  Dossier ignoré (pas de config.json) : {folder}")

    print(f"✅ [SCAN] {len(AVAILABLE_MODULES)} modules détectés et prêts.\n")

load_dynamic_modules()

# ── VIDEO STREAMING BLUEPRINT ────────────────────────────────────────────────
_streaming_mqtt    = None
_streaming_metrics = None
_streaming_ok      = False

def _mount_streaming_blueprint():
    global _streaming_mqtt, _streaming_metrics, _streaming_ok
    streaming_dir = PLUGINS_BASE_DIR / "video_streaming"
    if not streaming_dir.exists():
        logger.warning("⚠  video_streaming/ introuvable dans plugins/modules/")
        return
    plugins_str = str(PLUGINS_BASE_DIR)
    if plugins_str not in sys.path:
        sys.path.insert(0, plugins_str)
    try:
        from video_streaming.main import (
            streaming_bp,
            mqtt_handler       as _vs_mqtt,
            metrics_collector  as _vs_metrics,
            setup_socketio     as _vs_setup,
            start_metrics_push as _vs_push,
        )
        app.register_blueprint(streaming_bp, url_prefix='/streaming')
        _streaming_mqtt    = _vs_mqtt
        _streaming_metrics = _vs_metrics
        _streaming_ok      = True
        logger.info("📡 Blueprint /streaming/ monté")
    except Exception as e:
        logger.warning("⚠  video_streaming non chargé : %s", e)

_mount_streaming_blueprint()



# ============================================================================
# MEDIAPIPE - DÉTECTION D'OBJETS (REMPLACE YOLO)
# ============================================================================
#class MediaPipeObjectDetector:
 #   """
 #   Détection d'objets ultra-légère pour Raspberry Pi.
  #  Utilise EfficientDet-Lite0 (.tflite) au lieu de YOLO.
  #  """

  #  def __init__(self, model_path="models/efficientdet_lite0.tflite", confidence=0.5):
  #      self.confidence = confidence
  #      self.available = True

   #     try:
    #        import mediapipe as mp
    #        from mediapipe.tasks import python
    #        from mediapipe.tasks.python import vision

    #        if os.path.exists(model_path):
    #            logger.info(f"✅ Chargement MediaPipe Object Detector: {model_path}")
     #           base_options = python.BaseOptions(model_asset_path=model_path)

                # On utilise le mode IMAGE pour une intégration facile comme remplacement de YOLO
     #           options = vision.ObjectDetectorOptions(
      #              base_options=base_options,
      #              score_threshold=self.confidence,
     #               running_mode=vision.RunningMode.IMAGE
      #            )
      #          self.detector = vision.ObjectDetector.create_from_options(options)
   #         else:
    #            logger.warning(f"⚠️ Modèle non trouvé: {model_path}")
   #             self.available = False

   #     except ImportError:
   #         logger.warning("⚠️ MediaPipe Tasks pas installé. Détection désactivée.")
   #         self.available = False

  #  def detect(self, frame):
   #     """
   #     Détecte les objets avec la même structure de retour que l'ancien YOLO.
  #      """
   #     if not self.available or frame is None:
         #     return frame, {"objects": [], "count": 0, "inference_time_ms": 0}

   #     import mediapipe as mp
  #      import time
   #     import cv2

    #    start_time = time.time()

    #    try:
            # MediaPipe attend une image au format RGB
     #       rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      #      mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Inférence
     #       detection_result = self.detector.detect(mp_image)
     #       inference_time = (time.time() - start_time) * 1000

      #      detections = []
     #       annotated_frame = frame.copy()

      #      for detection in detection_result.detections:
                # Extraction de la boîte
     #           bbox = detection.bounding_box
      #          x1 = int(bbox.origin_x)
     #           y1 = int(bbox.origin_y)
     #           x2 = x1 + int(bbox.width)
     #          y2 = y1 + int(bbox.height)

                # Extraction de la catégorie
     #           category = detection.categories[0]
      #          class_name = category.category_name
      #          conf = float(category.score)

       #         detections.append({
       #             "class": class_name,
       #             "confidence": round(conf, 3),
       #             "box": [x1, y1, x2, y2]
       #         })

                # Dessin sur l'image
       #         cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #        label = f"{class_name} {conf:.2f}"
        #        cv2.putText(annotated_frame, label, (x1, max(y1-10, 10)),
       #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

      #      return annotated_frame, {
       #         "objects": detections,
       #         "count": len(detections),
       #         "inference_time_ms": round(inference_time, 2)
       #    }

        #except Exception as e:
       #     logger.error(f"❌ Erreur MediaPipe Detect: {e}")
       #     return frame, {"objects": [], "count": 0, "inference_time_ms": 0, "error": str(e)}


# ============================================================================
# YOLOV8N - DÉTECTION D'OBJETS
# ============================================================================
#class YOLOv8nDetector:
   # """
   # Détection d'objets légère pour Raspberry Pi
   # Utilise le modèle YOLOv8n (nano = très rapide)
   # """

 #   def __init__(self, model_path="models/yolov8n.pt", confidence=0.5):
    #     """
    #     Args:
    #         model_path: Chemin vers yolov8n.pt
    #         confidence: Seuil de confiance (0-1)
     #    """
    #     self.confidence = confidence
     #    self.model = None
    #     self.device = "cpu"  # Raspberry Pi n'a pas de GPU

   #      try:
    #         from ultralytics import YOLO

    #         if os.path.exists(model_path):
    #             logger.info(f"✅ Chargement YOLOv8n: {model_path}")
   #              self.model = YOLO(model_path)
    #             self.model.to(self.device)
    #         else:
    #             logger.warning(f"⚠️ Modèle YOLOv8n non trouvé: {model_path}")
    #             logger.info("   Télécharger: python -m pip install ultralytics")
   #              logger.info("   Puis: from ultralytics import YOLO; YOLO('yolov8n.pt')")
   #      except ImportError:
    #         logger.warning("⚠️ ultralytics pas installé. Détection désactivée.")
    #         self.model = None

  #  def detect(self, frame):
    #     """
     #    Détecte les objets dans une frame

     #    Returns:
       #      (frame_annotated, detections_dict)
       #      detections_dict = {
      #           "objects": [{"class": "person", "confidence": 0.95, "box": [x1,y1,x2,y2]}, ...],
       #          "inference_time_ms": 45
       #      }
      #   """
   #     if self.model is None:
  #          return frame, {"objects": [], "count": 0, "inference_time_ms": 0}

   #     try:
   #         start_time = time.time()
   #         results = self.model.predict(frame, conf=self.confidence, verbose=False)
   #         inference_time = (time.time() - start_time) * 1000

   #         detections = []
   #         annotated_frame = frame.copy()

   #         if results and len(results) > 0:
   #             result = results[0]

  #              if result.boxes is not None:
   #                 for box in result.boxes:
   #                     x1, y1, x2, y2 = map(int, box.xyxy[0])
   #                     confidence = float(box.conf[0])
  #                      class_id = int(box.cls[0])
   #                     class_name = self.model.names[class_id]

   #                     detections.append({
   #                         "class": class_name,
   #                         "confidence": round(confidence, 3),
   #                         "box": [x1, y1, x2, y2],
   #                         "class_id": class_id
     #                      })

   #                     # Dessiner la boîte
   #                     cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
   #                     label = f"{class_name} {confidence:.2f}"
   #                     cv2.putText(annotated_frame, label, (x1, y1-10),
   #                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    #        return annotated_frame, {
    #            "objects": detections,
      #          "count": len(detections),
     #           "inference_time_ms": round(inference_time, 2)
      #      }

     #   except Exception as e:
      #      logger.error(f"❌ Erreur YOLOv8n: {e}")
       #     return frame, {"objects": [], "count": 0, "inference_time_ms": 0, "error": str(e)}


# ============================================================================
# GESTURE RECOGNITION - DÉTECTION DE GESTES
# ============================================================================
#class GestureDetector:
   # """Détecteur de gestes avec MediaPipe"""

   # def __init__(self):
   #     self.current_gesture = "NONE"
   #     self.last_gesture = "NONE"

   #     try:
   #         import mediapipe as mp
  #          self.mp = mp
   #         self.hands = mp.solutions.hands.Hands(
   #             static_image_mode=False,
    #            max_num_hands=1,
    #            min_detection_confidence=0.7,
    #            min_tracking_confidence=0.5
   #         )
    #        self.available = True
    #        logger.info("✅ MediaPipe chargé")
   #     except ImportError:
   #         self.available = False
   #         logger.warning("⚠️ MediaPipe non disponible - gestes désactivés")

  #  def detect(self, frame):
   #     """
   #     Détecte un geste

    #    Returns:
   #         gesture_name (str): "NONE", "THUMBS_UP", "PEACE", "PALM", "FIST", etc.
    #    """
   #     if not self.available:
  #          return "NONE", frame

   #     try:
   #         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #        results = self.hands.process(rgb_frame)

   #         if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
   #             hand = results.multi_hand_landmarks[0]
   #             gesture = self._classify_gesture(hand)
  #              self.current_gesture = gesture
  #              return gesture, frame

   #         self.current_gesture = "NONE"
   #         return "NONE", frame

  #      except Exception as e:
   #         logger.error(f"❌ Erreur détection geste: {e}")
   #         return "NONE", frame

   # def _classify_gesture(self, hand):
   #     """Classifie un geste basé sur les landmarks"""
        # Landmarks importants
  #      thumb_tip = hand.landmark[4]
   #     index_tip = hand.landmark[8]
  #      middle_tip = hand.landmark[12]
  #      ring_tip = hand.landmark[16]
  #      pinky_tip = hand.landmark[20]

  #      palm = hand.landmark[0]

        # Calcul distances
 #       thumb_up = thumb_tip.y < palm.y
 #       index_up = index_tip.y < palm.y
  #      middle_up = middle_tip.y < palm.y
   #     ring_up = ring_tip.y < palm.y
  #      pinky_up = pinky_tip.y < palm.y

        # Détection simples
    #    if thumb_up and not (index_up or middle_up or ring_up or pinky_up):
    #        return "THUMBS_UP"

    #    if index_up and middle_up and not (ring_up or pinky_up):
    #        return "PEACE"

   #     if index_up and not (middle_up or ring_up or pinky_up):
   #         return "POINT"

   #     if thumb_up and index_up and middle_up and ring_up and pinky_up:
  #          return "PALM"

 #       if not (thumb_up or index_up or middle_up or ring_up or pinky_up):
  #          return "FIST"

  #      return "UNKNOWN"


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

            success, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

            if success:
                with open(filepath, 'wb') as f:
                    f.write(encoded.tobytes())

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
# INITIALISATION DES MODULES
# ============================================================================

logger.info("🚀 Initialisation S.H.O.S V3.0...")

#pi_camera = LibCameraCapture(width=640, height=480, fps=30)
#pi_camera.start()

#gesture_detector = GestureDetector()
#object_detector = MediaPipeObjectDetector(model_path="models/efficientdet_lite0.tflite", confidence=0.5)
#yolo_detector = YOLOv8nDetector(model_path="models/yolov8n.pt", confidence=0.5)

snapshot_manager = SnapshotManager()

# Variables globales
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
        self.active_modules = {}

        try:
            from paho.mqtt.enums import CallbackAPIVersion
            self.client = mqtt_client.Client(CallbackAPIVersion.VERSION2, "NOVA_WEB_BRIDGE")
        except ImportError:
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
        print(f"✅ [MQTT] Connecté au Broker (Code: {rc})")
        client.subscribe([
            ("shos/sensors/normalized", 1),
            ("shos/sensors/raw", 1),
            ("shos/camera/processed", 1),
            ("shos/benchmark/ping", 1),
            ("shos/plugins/+/data", 1)

            ("shos/orchestrator/decision",  1),
            ("shos/orchestrator/modules",   1),
            ("shos/orchestrator/energy",    1),
            ("shos/orchestrator/profile",   1),
        ])
        print("📡 [MQTT] Écoute active : Vidéo IA + Capteurs")

    def _on_message(self, client, userdata, msg):
        global global_frame
        topic = msg.topic

        try:
            if topic == "shos/camera/processed":
                nparr = np.frombuffer(msg.payload, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is not None:
                    with frame_lock:
                        global_frame = frame
                return

            payload_raw = msg.payload.decode('utf-8')

            if topic == "shos/benchmark/ping":
                self.client.publish("shos/benchmark/pong", payload_raw)
                return

            payload = json.loads(payload_raw)

            if topic in ("shos/sensors/normalized", "shos/sensors/raw"):
                socketio.emit('sensor_update', payload)

            elif "plugins" in topic:
                parts = topic.split('/')
                mod_name = parts[2]
                socketio.emit('module_update', {'module': mod_name, 'data': payload})
                socketio.emit('plugin_data', payload)

            elif topic.startswith("shos/orchestrator/"):
                kind = topic.split("/")[-1]
                socketio.emit(f'orchestrator_{kind}', payload)

        except Exception as e:
            if topic != "shos/camera/processed":
                print(f"❌ [MQTT] Erreur routing sur {topic}: {e}")

mqtt_bridge = MQTTHandler()

# ============================================================================
# ROUTES PRINCIPALES (WEB UI)
# ============================================================================

DEFAULT_PROFILE = {'name': 'ADRIEN_ASUS', 'id': '01'}

@app.route('/')
def index():
    """Page d'accueil avec les tuiles de modules dynamiques"""
    return render_template('index.html', modules=AVAILABLE_MODULES, profile=DEFAULT_PROFILE)

@app.route('/dashboard')
def dashboard(): return render_template('dashboard.html')

@app.route('/benchmark')
def benchmark_ui(): return render_template('benchmark.html')

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
    return render_template('profiles.html')

@app.route('/hud')
def hud(): return render_template('hud.html', profile=DEFAULT_PROFILE)

def resolve_profile(profile_id: str = "") -> dict:
    """Charge le profil complet depuis config_manager."""
    try:
        config   = config_manager.load_config()
        profiles = config.get("profiles", {})
        pid = profile_id.strip() or config.get("current_active_profile", "")
        data = profiles.get(pid, {})
        if data:
            return {
                "id":               pid,
                "name":             data.get("name", pid),
                "icon":             data.get("icon", "\U0001f464"),
                "main_module":      data.get("main_module", ""),
                "secondary_modules":data.get("secondary_modules", []),
                "description":      data.get("description", ""),
            }
    except Exception as e:
        logger.error("resolve_profile : %s", e)
    return DEFAULT_PROFILE

@app.route('/user_interface')
def user_interface():
    """Interface utilisateur — profil actif chargé dynamiquement."""
    profile_id = request.args.get('profile', '').strip()
    profile    = resolve_profile(profile_id)
    logger.info("\U0001f5a5  user_interface — profil : %s | main : %s | sec : %s",
                profile['name'], profile['main_module'], profile['secondary_modules'])
    return render_template('user_interface.html', profile=profile)

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

        config = config_manager.load_config()
        config["profiles"][profile_id] = {
            "name": profile_name,
            "main_module": data.get('main_module'),
            "secondary_modules": data.get('secondary_modules', [])
        }
        config_manager.save_config(config)

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

@app.route('/module/<module_id>/config')
def module_config_route(module_id):
    """Sert la page de configuration d'un module si config.html existe."""
    if module_id not in AVAILABLE_MODULES:
        return f"<h2>404</h2><p>Module {module_id} non chargé.</p>", 404
    
    config_path = PLUGINS_BASE_DIR / module_id / "config.html"
    if not config_path.exists():
        return f"<h2>404</h2><p>config.html introuvable pour {module_id}.</p>", 404
    
    try:
        profile_id = request.args.get('profile', '').strip()
        profile = resolve_profile(profile_id)
        return render_template(f'{module_id}/config.html',
                               config=AVAILABLE_MODULES[module_id],
                               profile=profile)
    except Exception as e:
        return f"<h2>Erreur</h2><p>{e}</p>", 500


# ============================================================================
# ROUTES MODULES DYNAMIQUES
# ============================================================================

@app.route('/module/<path:module_id>')
def universal_module_route(module_id):
    module_id = module_id.strip('/')

    # Rejette les tentatives de path traversal
    if '..' in module_id or module_id.startswith('/'):
        return "<h2>Erreur 400</h2><p>Identifiant de module invalide.</p>", 400

    logger.info(f"🔍 Demande d'accès au module : '{module_id}'")

    if module_id not in AVAILABLE_MODULES:
        return f"<h2>Erreur 404</h2><p>Le module '{module_id}' n'est pas chargé.</p>", 404

    config = AVAILABLE_MODULES[module_id]
    template_name = config.get('template', 'interface.html')

    chemin_physique = (PLUGINS_BASE_DIR / module_id / template_name).resolve()
    modules_root_resolved = PLUGINS_BASE_DIR.resolve()

    # Vérifie que le chemin résolu reste sous PLUGINS_BASE_DIR
    if not chemin_physique.is_relative_to(modules_root_resolved):
        logger.error(f"❌ Tentative d'accès hors sandbox : {chemin_physique}")
        return "<h2>Erreur 403</h2><p>Accès refusé.</p>", 403

    if not chemin_physique.exists():
        logger.error(f"❌ Fichier introuvable sur le disque : {chemin_physique}")
        return f"<h2>Erreur Critique</h2><p>Fichier introuvable : <br><b>{chemin_physique}</b></p>", 500

    try:
        jinja_template = f"{module_id}/{template_name}"
        logger.info(f"⚡ Rendu du template : {jinja_template}")
        return render_template(jinja_template, config=config, profile=DEFAULT_PROFILE)
    except Exception as e:
        logger.error(f"❌ Erreur lors du rendu : {e}")
        return f"<h2>Erreur de rendu</h2><p>{e}</p>", 500

# ============================================================================
# API SUPPRESSION PROFIL
# ============================================================================

@app.route('/api/delete_profile', methods=['POST'])
def api_delete_profile():
    """Supprime un profil. Body : { "id": "xxx" }"""
    try:
        data = request.get_json(silent=True) or {}
        pid  = data.get('id', '').strip()
        if not pid:
            return jsonify({"status":"error","message":"id manquant"}), 400
        config = config_manager.load_config()
        if pid not in config.get("profiles", {}):
            return jsonify({"status":"error","message":"Profil introuvable"}), 404
        del config["profiles"][pid]
        if config.get("current_active_profile") == pid:
            config["current_active_profile"] = ""
        config_manager.save_config(config)
        socketio.emit('profile_deleted', {'profile_id': pid})
        return jsonify({"status":"success"})
    except Exception as e:
        return jsonify({"status":"error","message":str(e)}), 500

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

@socketio.on('delete_profile')
def handle_delete_profile_socket(data):
    pid = (data or {}).get('id', '').strip()
    if not pid:
        return
    try:
        config = config_manager.load_config()
        if pid in config.get("profiles", {}):
            del config["profiles"][pid]
            if config.get("current_active_profile") == pid:
                config["current_active_profile"] = ""
            config_manager.save_config(config)
            emit('profile_deleted', {'profile_id': pid}, broadcast=True)
    except Exception as e:
        logger.error("delete_profile socket : %s", e)

@socketio.on('request_camera_status')
def handle_camera_status():
    """Demande du status caméra"""
    emit('camera_status', {
        "gesture": current_gesture,
        "detection_stats": detection_stats,
        "timestamp": datetime.now().isoformat()
    })

@socketio.on('request_snapshot')
def handle_snapshot_request():
    with frame_lock:
        frame = global_frame.copy() if global_frame is not None else None

    if frame is not None:
        filepath = snapshot_manager.capture(
            frame,
            gesture="MQTT_STREAM",
            metadata={"triggered_by": "user"}
        )
        emit('snapshot_captured', {
            'filepath': str(filepath),
            'timestamp': datetime.now().isoformat()
        })
    else:
        logger.warning("📸 Impossible de capturer : aucune image dans global_frame")

# ============================================================================
# BACKGROUND TASKS
# ============================================================================

def background_monitoring():
    """Monitoring système et caméra"""
    disk_path = str(PROJECT_ROOT.anchor)  # '/' sur Linux, 'C:\' sur Windows
    while True:
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            ram_percent = psutil.virtual_memory().percent
            disk_percent = psutil.disk_usage(disk_path).percent

            try:
                with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                    temp_raw = int(f.read()) / 1000
            except OSError:
                temp_raw = 0

            uptime = psutil.boot_time()

            socketio.emit('sys_update', {
                'cpu': cpu_percent,
                'ram': ram_percent,
                'disk': disk_percent,
                'temp': temp_raw,
                'uptime': time.time() - uptime,
                'timestamp': datetime.now().isoformat()
            })

            time.sleep(5)

        except Exception as e:
            logger.error(f"❌ Erreur monitoring: {e}")
            time.sleep(5)


VISION_SERVICE_URL = "http://localhost:5051/video_feed"

def generate_frames():
    """Proxy MJPEG vers vision_service:5051 via requests streaming."""
    try:
        import requests as _req
        with _req.get(VISION_SERVICE_URL, stream=True, timeout=8) as r:
            for chunk in r.iter_content(chunk_size=4096):
                if chunk:
                    yield chunk
    except Exception as e:
        logger.warning("⚠  Proxy MJPEG interrompu : %s", e)

@app.route('/video_feed')
def video_feed():
    """Proxy MJPEG → vision_service:5051."""
    return Response(
        stream_with_context(generate_frames()),
        mimetype='multipart/x-mixed-replace; boundary=frame',
    )


# --- GESTION DES SERVICES NOVA ---
running_processes = []

def start_nova_services():
    """Lance l'orchestrateur et la vision en arrière-plan"""
    services = ["orchestrator.py", "vision_service.py"]

    for script in services:
        try:
            p = subprocess.Popen([sys.executable, script])
            running_processes.append(p)
            logger.info(f"🚀 Service démarré : {script} (PID: {p.pid})")
        except Exception as e:
            logger.error(f"❌ Erreur lors du lancement de {script} : {e}")

def stop_nova_services(sig, frame):
    """Arrête proprement tous les services lors d'un Ctrl+C"""
    logger.info("🛑 Arrêt global des services NOVA...")
    for p in running_processes:
        p.terminate()
    sys.exit(0)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    logger.info("="*70)
    logger.info("   S.H.O.S V3.0 — Démarrage du superviseur")
    logger.info("   Modules : %s", list(AVAILABLE_MODULES.keys()))
    logger.info("   Streaming Blueprint : %s", "OK" if _streaming_ok else "non monté")
    logger.info("="*70)

    start_nova_services()

    # Démarrer les services video_streaming si Blueprint monté
    if _streaming_ok and _streaming_mqtt and _streaming_metrics:
        try:
            _streaming_mqtt.start()
            _streaming_metrics.start()
            logger.info("📡 video_streaming services démarrés")
        except Exception as e:
            logger.warning("⚠  video_streaming services : %s", e)

    signal.signal(signal.SIGINT, stop_nova_services)

    socketio.start_background_task(background_monitoring)

    try:
        socketio.run(
            app,
            host='0.0.0.0',
            port=5000,
            debug=False,
            use_reloader=False
        )
    except Exception as e:
        logger.error(f"Erreur fatale du serveur : {e}")
    finally:
        for p in running_processes:
            p.terminate()
