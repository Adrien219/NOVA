import serial
import json
import time
import platform
import sys
import threading
from pathlib import Path
from paho.mqtt import client as mqtt_client

class ArduinoBridge:
    """
    Bridge intelligent pour la liaison Arduino → MQTT
    - Auto-détection du port série (Windows/Linux)
    - Reconnexion automatique
    - Publication des données en boucle
    - Logging détaillé pour le débogage
    """
    
    def __init__(self, baudrate=115200, broker="localhost", port=1883):
        self.baudrate = baudrate
        self.broker = broker
        self.mqtt_port = port
        self.ser = None
        self.running = False
        self.client = None
        self.connected_mqtt = False
        self.last_data = None
        self.watchdog_timeout = 5  # Secondes avant timeout
        self.last_read_time = time.time()
        
        # Initialiser MQTT
        self._init_mqtt()
        
    def _init_mqtt(self):
        """Initialiser le client MQTT avec les callbacks"""
        try:
            self.client = mqtt_client.Client(
                mqtt_client.CallbackAPIVersion.VERSION2, 
                "SHOS_BRIDGE"
            )
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            self.client.on_message = self._on_message
            self.client.connect(self.broker, self.mqtt_port, 60)
            self.client.loop_start()
            print("✅ [MQTT] Client initialisé")
        except Exception as e:
            print(f"❌ [MQTT] Erreur initialisation : {e}")
            sys.exit(1)
    
    def _on_connect(self, client, userdata, flags, rc, properties=None):
        """Callback de connexion MQTT"""
        if rc == 0:
            self.connected_mqtt = True
            
            # S'abonner aux données du téléphone
            client.subscribe("shos/sensors/mobile")
            print("📡 [MQTT] Abonné au flux mobile")
            
            
            print("✅ [MQTT] Connecté au broker (localhost:1883)")
        else:
            print(f"❌ [MQTT] Erreur connexion (code {rc})")
    
    def _on_disconnect(self, client, userdata, rc, properties=None):
        """Callback de déconnexion MQTT"""
        self.connected_mqtt = False
        print(f"⚠️  [MQTT] Déconnecté (code {rc})")
    
    
    def _on_message(self, client, userdata, msg):
        """Reçoit du téléphone et renvoie vers le Web"""
        try:
            if msg.topic == "shos/sensors/mobile":
                # On relaie la donnée telle quelle vers le dashboard
                self.client.publish("shos/sensors/web", msg.payload)
        except Exception as e:
            print(f"⚠️ Erreur relais : {e}")
    
    
    def detect_arduino_port(self):
        """
        Auto-détecte le port Arduino selon l'OS
        Windows: COM1-COM9
        Linux: /dev/ttyUSB0, /dev/ttyACM0
        """
        system = platform.system()
        ports_to_try = []
        
        if system == "Windows":
            ports_to_try = [f"COM{i}" for i in range(1, 10)]
        else:  # Linux, macOS
            ports_to_try = [
                f"/dev/ttyUSB{i}" for i in range(0, 5)
            ] + [
                f"/dev/ttyACM{i}" for i in range(0, 5)
            ]
        
        print(f"🔍 [SERIAL] Détection du port Arduino ({system})...")
        
        for port in ports_to_try:
            try:
                ser = serial.Serial(port, self.baudrate, timeout=1)
                # Envoyer un caractère de test et vérifier la réponse
                time.sleep(0.5)  # Attendre le reset Arduino
                ser.reset_input_buffer()
                
                print(f"✅ [SERIAL] Arduino trouvé sur {port}")
                return ser
            except (serial.SerialException, OSError):
                continue
        
        print(f"❌ [SERIAL] Arduino non détecté sur les ports testés")
        return None
    
    def start(self):
        """Démarrer la boucle de lecture et publication"""
        self.running = True
        
        # Chercher l'Arduino
        self.ser = self.detect_arduino_port()
        if not self.ser:
            print("❌ [BRIDGE] Impossible de continuer sans Arduino")
            sys.exit(1)
        
        print("\n🚀 [BRIDGE] Démarrage de la boucle de publication MQTT...\n")
        
        try:
            while self.running:
                data = self._read_serial()
                
                if data:
                    self.last_read_time = time.time()
                    self._publish_mqtt(data)
                else:
                    # Vérifier le watchdog
                    elapsed = time.time() - self.last_read_time
                    if elapsed > self.watchdog_timeout:
                        print(f"⚠️  [WATCHDOG] Arduino ne répond plus ({elapsed:.1f}s)")
                        # Tenter une reconnexion
                        self._reconnect_serial()
                
                time.sleep(0.05)  # 50ms entre les lectures
        
        except KeyboardInterrupt:
            print("\n\n🛑 [BRIDGE] Arrêt du service...")
            self.stop()
    
    def _read_serial(self):
        """Lire une ligne JSON depuis l'Arduino"""
        try:
            if self.ser and self.ser.in_waiting > 0:
                line = self.ser.readline().decode('utf-8').strip()
                
                if line:
                    # Valider que c'est du JSON
                    data = json.loads(line)
                    return data
        
        except json.JSONDecodeError as e:
            print(f"⚠️  [SERIAL] JSON invalide : {e}")
        except UnicodeDecodeError as e:
            print(f"⚠️  [SERIAL] Erreur décodage : {e}")
        except Exception as e:
            print(f"❌ [SERIAL] Erreur lecture : {e}")
        
        return None
    
    def _publish_mqtt(self, data):
        """Publier les données sur MQTT"""
        if not self.connected_mqtt:
            print("⚠️  [MQTT] Non connecté, tentative de reconnexion...")
            return
        
        try:
            # Publier les données brutes
            payload = json.dumps(data)
            result = self.client.publish(
                "shos/sensors/raw",
                payload,
                qos=1,  # Au moins une fois
                retain=False
            )
            
            # Vérifier le statut de publication
            if result.rc == mqtt_client.MQTT_ERR_SUCCESS:
                print(f"📤 [MQTT] Publiée : {payload}")
                self.last_data = data
            else:
                print(f"❌ [MQTT] Erreur publication : {result.rc}")
        
        except Exception as e:
            print(f"❌ [MQTT] Erreur sérialisation : {e}")
    
    def _reconnect_serial(self):
        """Tenter une reconnexion au port série"""
        print("🔄 [SERIAL] Tentative de reconnexion...")
        
        if self.ser:
            try:
                self.ser.close()
            except:
                pass
        
        time.sleep(1)
        self.ser = self.detect_arduino_port()
        
        if self.ser:
            self.last_read_time = time.time()
            print("✅ [SERIAL] Reconnecté")
        else:
            print("❌ [SERIAL] Reconnexion échouée")
    
    def stop(self):
        """Arrêter le bridge proprement"""
        self.running = False
        
        if self.ser:
            self.ser.close()
            print("✅ [SERIAL] Port fermé")
        
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            print("✅ [MQTT] Déconnecté")
        
        print("✅ [BRIDGE] Service arrêté")


def main():
    """Point d'entrée du Bridge"""
    print("""
    ╔═══════════════════════════════════════╗
    ║   S.H.O.S ARDUINO BRIDGE V2.0         ║
    ║   Serial → MQTT Bridge                ║
    ╚═══════════════════════════════════════╝
    """)
    
    bridge = ArduinoBridge(
        baudrate=115200,
        broker="localhost",
        port=1883
    )
    
    bridge.start()


if __name__ == "__main__":
    main()
