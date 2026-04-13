import paho.mqtt.client as mqtt
import json
import time
import random

# CONFIGURATION DU TEST
NAME = "capteur_test"
PRIORITY = 2  # 1=Haut, 3=Bas

def on_connect(client, userdata, flags, rc):
    print(f"📡 Module {NAME} connecté. Inscription...")
    # S'enregistrer auprès du Backbone
    client.publish("nova/modules/register", json.dumps({"name": NAME, "priority": PRIORITY}))
    # Écouter les ordres du Backbone
    client.subscribe(f"nova/modules/{NAME}/control")

def on_message(client, userdata, msg):
    command = msg.payload.decode()
    if command == "START":
        print("🚀 AUTORISATION REÇUE ! Envoi de données...")
        userdata['running'] = True
    elif command == "STOP":
        print("🛑 ORDRE D'ARRÊT ! Fermeture...")
        exit(0)

state = {'running': False}
client = mqtt.Client()
client.user_data_set(state)
client.on_connect = on_connect
client.on_message = on_message
client.connect("localhost", 1883)
client.loop_start()

try:
    while True:
        if state['running']:
            # Simuler l'envoi de données (ex: une distance)
            val = random.randint(10, 100)
            client.publish(f"nova/modules/{NAME}/data", json.dumps({"valeur": val}))
            print(f"📤 Donnée envoyée : {val}")
        time.sleep(2)
except KeyboardInterrupt:
    pass
