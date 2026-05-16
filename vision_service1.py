#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NOVA VISION SERVICE - Picamera2 + MediaPipe
✅ Capture natif libcamera
✅ Détection main MediaPipe
✅ Stream MJPEG Flask
✅ Publish MQTT
"""

import cv2
import mediapipe as mp
import threading
import time
import logging
import numpy as np
import json
from flask import Flask, Response
import paho.mqtt.client as mqtt

# ============================================================================
# CONFIG
# ============================================================================
PORT_WEB = 5051   # 5050 = réservé au service principal SHOS
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "shos/camera/vision"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger("NOVA_VISION")

app = Flask(__name__)

# ============================================================================
# PICAMERA2 + MEDIAPIPE VISION
# ============================================================================
class NovaVision:
    def __init__(self):
        """Initialise Picamera2 et MediaPipe"""
        
        # === MEDIAPIPE ===
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        logger.info("✅ MediaPipe initialisé")
        
        # === PICAMERA2 (NATIF LIBCAMERA) ===
        # Pré-initialisation — évite AttributeError si toutes les tentatives échouent
        self.picam2 = None
        self.cap = None
        self.use_picamera2 = False

        try:
            from picamera2 import Picamera2, Preview

            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(
                main={"format": 'BGR888', "size": (640, 480)},
                lores={"format": 'YUV420', "size": (320, 240)}
            )
            self.picam2.configure(config)
            self.picam2.start()

            logger.info("✅ Picamera2 initialisée (libcamera natif)")
            self.use_picamera2 = True

        except ImportError:
            logger.warning("⚠️ Picamera2 indisponible, fallback cv2.VideoCapture(0)")
            self._init_opencv_fallback()

        except Exception as e:
            # Cas fréquent : "Pipeline handler in use by another process"
            # = ce service tourne déjà dans un autre terminal
            logger.error(f"❌ Erreur Picamera2 : {e}")
            if "use by another process" in str(e):
                logger.warning("💡 La caméra est déjà utilisée par un autre process.")
                logger.warning("   Ne pas lancer vision_service deux fois en parallèle.")
                logger.warning("   Vérifier : ps aux | grep vision_service")
            logger.warning("⚠️ Tentative fallback cv2.VideoCapture(0)...")
            self._init_opencv_fallback()
        
        # === STATE ===
        self.output_frame = None
        self.frame_lock = threading.Lock()
        self.is_running = True
        self.frame_count = 0
        self.last_hands = None
        self.last_pose = None
        
        # === MQTT ===
        self.mqtt_client = self._init_mqtt()
    
    def _init_opencv_fallback(self):
        """Initialise OpenCV comme source caméra de secours."""
        try:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.use_picamera2 = False
                logger.info("✅ Fallback OpenCV initialisé (index=0)")
            else:
                self.cap = None
                logger.error("❌ OpenCV : aucune caméra disponible (index=0)")
                logger.error("   Le service continuera sans flux vidéo.")
        except Exception as e:
            self.cap = None
            logger.error(f"❌ Fallback OpenCV échoué : {e}")

    def _init_mqtt(self):
        """Initialise client MQTT"""
        try:
            try:
                from paho.mqtt.enums import CallbackAPIVersion
                client = mqtt.Client(CallbackAPIVersion.VERSION2, "NOVA_VISION")
            except:
                client = mqtt.Client("NOVA_VISION")
            
            client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
            client.loop_start()
            logger.info("✅ MQTT client connecté")
            return client
        except Exception as e:
            logger.warning(f"⚠️ MQTT non disponible: {e}")
            return None
    
    def process_vision(self):
        """Boucle principale de traitement vidéo"""
        logger.info("🚀 Traitement vision démarré...")

        # Vérification caméra disponible
        if not self.use_picamera2 and self.cap is None:
            logger.error("❌ Aucune caméra disponible — boucle vision annulée.")
            logger.error("   Causes possibles :")
            logger.error("   1. vision_service tourne déjà dans un autre process")
            logger.error("   2. Nappe caméra mal connectée")
            logger.error("   3. Picamera2 pas installée et webcam USB absente")
            return

        # Warm-up (laisser la caméra se stabiliser)
        for _ in range(10):
            if self.use_picamera2:
                try:
                    self.picam2.capture_array()
                except Exception:
                    pass
            elif self.cap is not None:
                self.cap.read()
            time.sleep(0.05)

        logger.info("👁️ Analyse MediaPipe active")

        # Variables pour calcul FPS réel
        fps_counter = 0
        fps_timer = time.time()
        current_fps = 0.0
        
        # === BOUCLE PRINCIPALE ===
        while self.is_running:
            try:
                # --- CAPTURE FRAME ---
                if self.use_picamera2:
                    try:
                        frame = self.picam2.capture_array()
                    except Exception as e:
                        logger.warning(f"⚠️ Picamera2 read error: {e}")
                        time.sleep(0.1)
                        continue
                elif self.cap is not None:
                    success, frame = self.cap.read()
                    if not success or frame is None:
                        time.sleep(0.1)
                        continue
                else:
                    # Aucune caméra — on sort proprement
                    logger.error("❌ Aucune source vidéo, arrêt de la boucle.")
                    break
                
                # --- VALIDATION FRAME ---
                if frame is None or frame.size == 0:
                    continue
                
                if len(frame.shape) < 2 or frame.shape[0] < 10 or frame.shape[1] < 10:
                    continue
                
                # --- FORMAT FIX ---
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                
                if len(frame.shape) != 3 or frame.shape[2] != 3:
                    try:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    except:
                        continue
                
                # Ensure contiguous memory
                frame = np.ascontiguousarray(frame)
                
                # --- TRAITEMENT IA ---
                try:
                    # Conversion RGB pour MediaPipe
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Détection mains
                    hand_results = self.hands.process(image_rgb)
                    if hand_results.multi_hand_landmarks:
                        self.last_hands = hand_results.multi_hand_landmarks
                        for hand_landmarks in hand_results.multi_hand_landmarks:
                            self.mp_draw.draw_landmarks(
                                frame, 
                                hand_landmarks, 
                                self.mp_hands.HAND_CONNECTIONS,
                                self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
                            )
                    
                    # Détection pose (corps complet)
                    pose_results = self.pose.process(image_rgb)
                    if pose_results.pose_landmarks:
                        self.last_pose = pose_results.pose_landmarks
                        self.mp_draw.draw_landmarks(
                            frame,
                            pose_results.pose_landmarks,
                            self.mp_pose.POSE_CONNECTIONS,
                            self.mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                            self.mp_draw.DrawingSpec(color=(255, 255, 0), thickness=2)
                        )
                    
                    # Ajouter info texte
                    cv2.putText(frame, f"Hands: {len(hand_results.multi_hand_landmarks or [])}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"FPS: {current_fps:.1f}", 
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                except cv2.error as e:
                    logger.debug(f"⚠️ OpenCV error (ignored): {e}")
                    continue
                except Exception as e:
                    logger.debug(f"⚠️ MediaPipe error: {e}")
                    continue
                
                # --- ENCODE & PUBLISH ---
                try:
                    # Compression JPEG
                    ret, buffer = cv2.imencode(
                        '.jpg', frame, 
                        [cv2.IMWRITE_JPEG_QUALITY, 60]
                    )
                    
                    if not ret:
                        continue
                    
                    frame_bytes = buffer.tobytes()
                    
                    # Mise à jour frame locale
                    with self.frame_lock:
                        self.output_frame = frame_bytes
                        self.frame_count += 1

                    # Calcul FPS réel sur fenêtre 1 seconde
                    fps_counter += 1
                    now = time.time()
                    if now - fps_timer >= 1.0:
                        current_fps = round(fps_counter / (now - fps_timer), 1)
                        fps_counter = 0
                        fps_timer = now

                    # Publish MQTT — payload enrichi pour le benchmark
                    # Le champ "timestamp" est obligatoire pour mesurer la latence E2E
                    if self.mqtt_client:
                        try:
                            metadata = {
                                "timestamp": time.time(),        # ← latence E2E benchmark
                                "frame_id": self.frame_count,
                                "fps": current_fps,              # ← FPS réel pipeline
                                "hands_detected": len(hand_results.multi_hand_landmarks or []),
                                "pose_detected": bool(pose_results.pose_landmarks),
                                "camera": "picamera2" if self.use_picamera2 else "opencv",
                            }
                            self.mqtt_client.publish(
                                MQTT_TOPIC,
                                json.dumps(metadata),
                                qos=0
                            )
                        except Exception:
                            pass
                
                except Exception as e:
                    logger.debug(f"⚠️ Encode error: {e}")
                    continue
                
                # Sleep minimal
                time.sleep(0.01)
            
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"❌ Erreur boucle vision: {e}")
                time.sleep(0.1)
    
    def generate_frames(self):
        """Générateur MJPEG pour Flask"""
        logger.info("📡 Stream vidéo démarré")
        
        while self.is_running:
            with self.frame_lock:
                if self.output_frame is None:
                    time.sleep(0.05)
                    continue
                frame_to_send = self.output_frame
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_to_send + b'\r\n')
            time.sleep(0.02)
    
    def stop(self):
        """Arrête les threads"""
        self.is_running = False
        if self.use_picamera2:
            try:
                self.picam2.stop()
                self.picam2.close()
            except:
                pass
        else:
            try:
                self.cap.release()
            except:
                pass
        
        if self.mqtt_client:
            try:
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()
            except:
                pass

# ============================================================================
# INSTANCE GLOBALE
# ============================================================================
nova_vision = NovaVision()

# ============================================================================
# FLASK ROUTES
# ============================================================================
@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>NOVA Vision Stream</title>
        <style>
            body { background: #1a1a1a; color: #fff; font-family: monospace; text-align: center; }
            img { max-width: 90%; border: 2px solid #00ff00; border-radius: 10px; margin: 20px; }
            h1 { color: #00ff00; }
        </style>
    </head>
    <body>
        <h1>👁️ NOVA VISION STREAM</h1>
        <img src="/video_feed" alt="Vision Stream">
        <p>MediaPipe Hand + Pose Detection</p>
    </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    """Stream MJPEG"""
    return Response(
        nova_vision.generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/status')
def status():
    """Status JSON"""
    return {
        "status": "online",
        "frames": nova_vision.frame_count,
        "hands_detected": bool(nova_vision.last_hands),
        "pose_detected": bool(nova_vision.last_pose),
        "camera": "Picamera2" if nova_vision.use_picamera2 else "OpenCV"
    }

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    try:
        # Démarrer thread vision
        vision_thread = threading.Thread(
            target=nova_vision.process_vision, 
            daemon=True
        )
        vision_thread.start()
        
        logger.info(f"🚀 Serveur VISION sur http://0.0.0.0:{PORT_WEB}")
        logger.info(f"📡 Stream: http://0.0.0.0:{PORT_WEB}/video_feed")
        logger.info(f"📊 Status: http://0.0.0.0:{PORT_WEB}/status")
        
        app.run(
            host='0.0.0.0',
            port=PORT_WEB,
            debug=False,
            threaded=True,
            use_reloader=False
        )
    
    except KeyboardInterrupt:
        logger.info("⛔ Arrêt...")
        nova_vision.stop()
    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        nova_vision.stop()
