import cv2
import numpy as np
from paho.mqtt import client as mqtt_client
import json
import os
from ultralytics import YOLO

# --- CONFIGURATION DES TOPICS ---
TOPIC_CAMERA = "shos/camera/raw"
TOPIC_CONTROL = "shos/plugins/vision_objet/control"
TOPIC_GLOBAL = "shos/plugins/control"
TOPIC_RESULTS = "shos/plugins/vision_objet/data"

# Chemin vers ton modèle (vérifie bien que le dossier modeles est au bon endroit)
MODEL_PATH = "../../modeles/yolov8n.pt"

class VisionPlugin:
    def __init__(self):
        print("🧠 [VISION] Chargement du modèle YOLOv8...")
        self.model = YOLO(MODEL_PATH)
        
        # 🚩 État initial : désactivé pour économiser le Raspberry Pi
        self.active = False 
        
        # Setup MQTT
        self.client = mqtt_client.Client(mqtt_client.CallbackAPIVersion.VERSION2, "PLUGIN_VISION")
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        
        try:
            self.client.connect("localhost", 1883, 60)
            print("📡 [VISION] Tentative de connexion au Backbone MQTT...")
        except Exception as e:
            print(f"❌ [VISION] Erreur de connexion MQTT : {e}")

    def on_connect(self, client, userdata, flags, rc, properties):
        if rc == 0:
            print("✅ [VISION] Connecté au Backbone. En attente d'activation par profil...")
            # On s'abonne aux flux d'images ET aux deux canaux de contrôle
            client.subscribe([
                (TOPIC_CAMERA, 0),
                (TOPIC_CONTROL, 0),
                (TOPIC_GLOBAL, 0)
            ])
        else:
            print(f"❌ [VISION] Échec de connexion, code : {rc}")

    def on_message(self, client, userdata, msg):
        try:
            # 1. GESTION DES COMMANDES (Start / Stop)
            if msg.topic in [TOPIC_CONTROL, TOPIC_GLOBAL]:
                command = json.loads(msg.payload.decode())
                action = command.get("action")
                
                if action == "start":
                    self.active = True
                    print("🚀 [VISION] Module ACTIVÉ")
                elif action in ["stop", "stop_all"]:
                    self.active = False
                    print("😴 [VISION] Module mis en PAUSE")
                return

            # 2. TRAITEMENT DE L'IMAGE (Seulement si actif)
            if self.active and msg.topic == TOPIC_CAMERA:
                # Vérification de la taille du flux pour éviter les erreurs de décodage
                if len(msg.payload) < 500:
                    return

                # Décodage de l'image reçue par MQTT
                nparr = np.frombuffer(msg.payload, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is not None and frame.size > 0:
                    # Inférence IA (imgsz=160 pour la rapidité sur Pi)
                    results = self.model(frame, stream=True, conf=0.4, verbose=False, imgsz=160)
                    
                    detections = []
                    for r in results:
                        for box in r.boxes:
                            label = self.model.names[int(box.cls[0])]
                            detections.append(label)

                    # Envoi des résultats si on a trouvé quelque chose
                    if detections:
                        print(f"🎯 [VISION] Détecté : {detections}")
                        output = {
                            "plugin": "vision_objet",
                            "found": detections,
                            "count": len(detections)
                        }
                        self.client.publish(TOPIC_RESULTS, json.dumps(output))

        except Exception as e:
            # On ne print pas l'erreur pour ne pas polluer le terminal en cas de flux saccadé
            pass

    def run(self):
        # Boucle infinie pour maintenir le plugin en vie
        self.client.loop_forever()

if __name__ == "__main__":
    plugin = VisionPlugin()
    plugin.run()