import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion
import time
import json
import statistics

# --- CONFIGURATION ---
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
TOPIC_PING = "nova/benchmark/ping"
TOPIC_PONG = "nova/benchmark/pong"
NB_TESTS = 20  # Nombre d'itérations pour la moyenne

latencies = []

# --- CALLBACKS MQTT ---
def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print("✅ Connecté au Broker MQTT")
        client.subscribe(TOPIC_PONG)
    else:
        print(f"❌ Échec de connexion (Code {rc})")

def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload)
        sent_time = data['timestamp']
        # Calcul du Round Trip Time (RTT)
        latency = (time.time() - sent_time) * 1000 
        latencies.append(latency)
        print(f"Test {data['iter']+1}/{NB_TESTS} : {latency:.2f} ms")
    except Exception as e:
        print(f"⚠️ Erreur de lecture : {e}")

# --- INITIALISATION DU CLIENT (PAHO 2.0) ---
# Utilisation de CallbackAPIVersion.VERSION1 pour la compatibilité
client = mqtt.Client(callback_api_version=CallbackAPIVersion.VERSION1, client_id="NOVA_Bench")
client.on_connect = on_connect
client.on_message = on_message

# --- BOUCLE PRINCIPALE ---
print(f"🚀 Démarrage du Benchmark NOVA sur {MQTT_BROKER}...")
try:
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()

    # Attendre un peu que la connexion s'établisse
    time.sleep(1)

    for i in range(NB_TESTS):
        payload = json.dumps({
            "timestamp": time.time(),
            "iter": i
        })
        client.publish(TOPIC_PING, payload)
        time.sleep(0.5) # Pause entre les pings pour ne pas saturer

    # Attendre la dernière réponse
    time.sleep(1)

finally:
    client.loop_stop()
    client.disconnect()

    # --- RAPPORT FINAL ---
    if latencies:
        print("\n" + "="*30)
        print("📊 RÉSULTATS DU BENCHMARK NOVA")
        print("="*30)
        print(f"Moyenne    : {statistics.mean(latencies):.2f} ms")
        print(f"Médiane    : {statistics.median(latencies):.2f} ms")
        print(f"Minimum    : {min(latencies):.2f} ms")
        print(f"Maximum    : {max(latencies):.2f} ms")
        print(f"Écart-type : {statistics.stdev(latencies):.2f} ms")
        print("="*30)
        
        if statistics.mean(latencies) < 100:
            print("💡 Résultat : EXCELLENT (Temps réel fluide)")
        elif statistics.mean(latencies) < 300:
            print("💡 Résultat : ACCEPTABLE (Légère latence perçue)")
        else:
            print("💡 Résultat : CRITIQUE (Trop lent pour l'assistance)")
    else:
        print("\n❌ Échec : Aucun message reçu. Vérifie que NOVA_Backbone écoute bien le topic 'nova/benchmark/ping' et renvoie le message sur 'nova/benchmark/pong'.")
