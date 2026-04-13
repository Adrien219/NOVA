import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion
import time
import json
import statistics

# --- CONFIGURATION (SYNCHRONISÉE AVEC NOVA.PY) ---
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
TOPIC_PING = "shos/benchmark/ping"
TOPIC_PONG = "shos/benchmark/pong"
NB_TESTS = 20  # <--- CETTE LIGNE DOIT EXISTER

latencies = []

def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print("✅ Connecté au Broker MQTT")
        client.subscribe(TOPIC_PONG)
        print(f"📡 Écoute sur {TOPIC_PONG}...")
    else:
        print(f"❌ Échec de connexion (Code {rc})")

def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload)
        sent_time = data['timestamp']
        latency = (time.time() - sent_time) * 1000 
        latencies.append(latency)
        print(f"Test {data['iter']+1}/{NB_TESTS} : {latency:.2f} ms")
    except Exception as e:
        print(f"⚠️ Erreur de lecture : {e}")

# --- INITIALISATION ---
client = mqtt.Client(callback_api_version=CallbackAPIVersion.VERSION1, client_id="NOVA_Bench")
client.on_connect = on_connect
client.on_message = on_message

print(f"🚀 Démarrage du Benchmark NOVA...")
try:
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()
    time.sleep(1) # Attente connexion

    for i in range(NB_TESTS):
        payload = json.dumps({"timestamp": time.time(), "iter": i})
        client.publish(TOPIC_PING, payload)
        time.sleep(0.2) 

    time.sleep(1) # Attente des dernières réponses

finally:
    client.loop_stop()
    client.disconnect()

    if latencies:
        print("\n" + "="*30)
        print(f"📊 MOYENNE : {statistics.mean(latencies):.2f} ms")
        print("="*30)
    else:
        print("\n❌ ÉCHEC : Aucun message reçu.")
        print(f"Vérifie que nova.py répond bien sur le topic: {TOPIC_PING}")
