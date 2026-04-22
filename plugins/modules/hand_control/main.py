import cv2
import mediapipe as mp
import time
import json
import numpy as np
from paho.mqtt import client as mqtt_client

# --- CONFIGURATION ---
TOPIC_CAMERA_IN = "shos/camera/raw"
TOPIC_CAMERA_OUT = "shos/camera/processed" # Topic pour l'image avec les points
TOPIC_RESULTS = "shos/plugins/hand_control/data"
BROKER = "localhost"

class GesturePlugin:
    def __init__(self):
        print("🧠 [GESTURE] Initialisation MediaPipe avec Visualisation...")
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        # Styles de dessin (points et lignes)
        self.mp_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        self.last_gesture = "NONE"
        self.gesture_start_time = 0
        self.confirm_threshold = 0.8

        self.client = mqtt_client.Client(mqtt_client.CallbackAPIVersion.VERSION2, "PLUGIN_GESTURE")
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

    def on_connect(self, client, userdata, flags, rc, properties=None):
        print(f"📡 [GESTURE] Connecté au Backbone NOVA")
        client.subscribe(TOPIC_CAMERA_IN)

    def on_message(self, client, userdata, msg):
        try:
            # 1. Décodage de l'image
            nparr = np.frombuffer(msg.payload, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None: return

            # 2. Traitement MediaPipe
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)
            current_gesture = "NONE"

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Dessin du squelette (pour vérifier visuellement)
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    lms = hand_landmarks.landmark
                    fingers = []

                    # 1. POUCE : Logique plus souple (distance horizontale entre le bout et la base)
                    # On compare le bout du pouce (4) avec l'articulation (2)
                    if lms[4].x > lms[2].x + 0.02: # Ajuste le +0.02 si besoin
                        fingers.append(True)
                    else:
                        fingers.append(False)

                    # 2. LES 4 AUTRES DOIGTS (Index, Majeur, Annulaire, Auriculaire)
                    # Un doigt est levé si son TIP (bout) est plus haut que son PIP (deuxième articulation)
                    for tip in [8, 12, 16, 20]:
                        if lms[tip].y < lms[tip-2].y:
                            fingers.append(True)
                        else:
                            fingers.append(False)

                    # --- NOUVEAU MAPPING PLUS PRÉCIS ---
                    # [Pouce, Index, Majeur, Annulaire, Auriculaire]
                    
                    # 🖐️ TOUT LEVÉ = START
                    if fingers == [True, True, True, True, True]:
                        current_gesture = "START_SYSTEM"
                    
                    # ✊ TOUT FERMÉ (Poing) = STOP
                    # Parfois le pouce reste un peu "ouvert" sur un poing, on check surtout les 4 doigts
                    elif fingers[1:] == [False, False, False, False]:
                        current_gesture = "STOP_SYSTEM"
                    
                    # ☝️ SEUL L'INDEX EST LEVÉ = READ_TEXT
                    elif fingers == [False, True, False, False, False]:
                        current_gesture = "READ_TEXT"
                    
                    # 👍 SEUL LE POUCE EST LEVÉ = TRIGGER_FLORENCE
                    elif fingers == [True, False, False, False, False]:
                        current_gesture = "TRIGGER_FLORENCE"

            # Le reste du code (publish et confirm) reste identique...

            # 3. Envoi de l'image dessinée vers le serveur NOVA
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 35])
            self.client.publish(TOPIC_CAMERA_OUT, buffer.tobytes())

            # 4. Gestion de la confirmation du geste
            if current_gesture != self.last_gesture:
                if current_gesture != "NONE": self.gesture_start_time = time.time()
                self.last_gesture = current_gesture
            elif current_gesture != "NONE":
                if (time.time() - self.gesture_start_time) > self.confirm_threshold:
                    output = {"plugin": "hand_control", "gesture": current_gesture, "status": "confirmed"}
                    self.client.publish(TOPIC_RESULTS, json.dumps(output))
                    print(f"🎯 Geste Validé : {current_gesture}")
                    self.gesture_start_time = time.time() + 1.5

        except Exception as e:
            print(f"Erreur main.py: {e}")

    def run(self):
        self.client.connect(BROKER, 1883, 60)
        self.client.loop_forever()

if __name__ == "__main__":
    GesturePlugin().run()