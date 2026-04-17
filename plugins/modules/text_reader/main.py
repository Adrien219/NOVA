import cv2
import pytesseract
import numpy as np
import json
import time
from paho.mqtt import client as mqtt_client

# --- CONFIGURATION ---
TOPIC_RESULTS = "shos/plugins/text_reader/data"
# Sur Windows, si Tesseract n'est pas dans ton PATH, décommente la ligne suivante :
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text(frame):
    """
    Fonction optimisée pour extraire le texte d'une image OpenCV
    """
    try:
        # 1. Conversion en niveaux de gris
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. Amélioration du contraste (Seuillage d'Otsu)
        # On peut aussi ajouter un flou gaussien pour réduire le bruit si besoin
        processed_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # 3. OCR avec Tesseract (langue française forcée)
        config = r'--oem 3 --psm 3'
        text = pytesseract.image_to_string(processed_img, lang='fra', config=config)

        return text.strip()
    except Exception as e:
        print(f"❌ Erreur OCR : {e}")
        return ""

class TextReaderPlugin:
    def __init__(self):
        self.client = mqtt_client.Client(mqtt_client.CallbackAPIVersion.VERSION2, "PLUGIN_TEXT_READER")
        try:
            self.client.connect("localhost", 1883, 60)
            print("📡 [OCR] Connecté au Backbone MQTT")
        except Exception as e:
            print(f"❌ [OCR] Erreur connexion MQTT : {e}")

    def run(self):
        # On utilise l'index 0 pour la caméra par défaut (ou l'URL de ton flux Pi)
        cap = cv2.VideoCapture(0)
        
        print("🚀 [OCR] Lecteur démarré. Prêt à scanner...")
        
        last_send_time = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # On ne fait l'OCR que toutes les 1.5 secondes pour ne pas saturer le processeur
            current_time = time.time()
            if current_time - last_send_time > 1.5:
                # Analyse du texte
                result_text = extract_text(frame)

                if result_text:
                    print(f"📝 [OCR] Texte détecté : {result_text[:30]}...")
                    
                    # Envoi des données vers l'interface
                    payload = {
                        "plugin": "text_reader",
                        "text": result_text
                    }
                    self.client.publish(TOPIC_RESULTS, json.dumps(payload))
                
                last_send_time = current_time

            # Optionnel : Affichage local pour debug (à commenter sur le Pi final)
            # cv2.imshow("Debug OCR", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    plugin = TextReaderPlugin()
    plugin.run()