import cv2
import numpy as np
import json
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from paho.mqtt import client as mqtt_client

# Topics
TOPIC_CAMERA = "shos/camera/raw"
TOPIC_RESULTS = "shos/plugins/vision_objet/data"
MODEL_PATH = "../../modeles/efficientdet_lite0.tflite"

class VisionPlugin:
    def __init__(self):
        print("🧠 [VISION] Chargement MediaPipe...")
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.4)
        self.detector = vision.ObjectDetector.create_from_options(options)
        
        self.active = True # Activé par défaut pour tes tests
        self.client = mqtt_client.Client(mqtt_client.CallbackAPIVersion.VERSION2, "PLUGIN_VISION")
        self.client.on_connect = lambda c, u, f, rc, p: c.subscribe(TOPIC_CAMERA)
        self.client.on_message = self.on_message
        self.client.connect("localhost", 1883, 60)

    def on_message(self, client, userdata, msg):
        if self.active and msg.topic == TOPIC_CAMERA:
            # Conversion de l'image MQTT
            nparr = np.frombuffer(msg.payload, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None: return

            # Inférence MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = self.detector.detect(mp_image)

            found = [det.categories[0].category_name for det in result.detections]
            
            if found:
                print(f"🎯 Détecté : {found}")
                # ENVOI VERS L'INTERFACE
                output = {"plugin": "vision_objet", "found": found}
                self.client.publish(TOPIC_RESULTS, json.dumps(output))

    def run(self):
        self.client.loop_forever()

if __name__ == "__main__":
    VisionPlugin().run()