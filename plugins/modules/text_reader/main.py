import cv2
import pytesseract
import numpy as np
import json
import time
from paho.mqtt import client as mqtt_client

# --- CONFIGURATION ---
TOPIC_CAMERA_IN = "shos/camera/raw"  # On écoute le flux de NOVA
TOPIC_RESULTS = "shos/plugins/text_reader/data"
BROKER = "localhost"

# Configuration Tesseract Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text(frame):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Amélioration pour Tesseract
        processed_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        text = pytesseract.image_to_string(processed_img, lang='fra', config='--oem 3 --psm 3')
        return text.strip()
    except Exception as e:
        return ""

class TextReaderPlugin:
    def __init__(self):
        # Utilisation de la version 2 de l'API Paho pour correspondre à ton nova.py
        self.client = mqtt_client.Client(mqtt_client.CallbackAPIVersion.VERSION2, "PLUGIN_TEXT_READER")
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.last_ocr_time = 0

    def on_connect(self, client, userdata, flags, rc, properties=None):
        print("📡 [OCR] Connecté au Backbone NOVA. En attente d'images...")
        client.subscribe(TOPIC_CAMERA_IN)

    def on_message(self, client, userdata, msg):
        current_time = time.time()
        # On ne traite une image que toutes les 1.5 secondes (pour le CPU)
        if current_time - self.last_ocr_time > 1.5:
            try:
                # 1. Décodage de l'image reçue via MQTT
                nparr = np.frombuffer(msg.payload, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None: return

                # 2. OCR
                result_text = extract_text(frame)

                if result_text and len(result_text) > 2:
                    print(f"📝 [OCR] Texte : {result_text[:40]}...")
                    payload = {
                        "plugin": "text_reader",
                        "text": result_text
                    }
                    self.client.publish(TOPIC_RESULTS, json.dumps(payload))
                
                self.last_ocr_time = current_time
            except Exception as e:
                print(f"❌ Erreur traitement : {e}")

    def run(self):
        try:
            self.client.connect(BROKER, 1883, 60)
            self.client.loop_forever()
        except KeyboardInterrupt:
            print("🛑 Arrêt du lecteur.")

if __name__ == "__main__":
    plugin = TextReaderPlugin()
    plugin.run()