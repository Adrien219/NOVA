import subprocess
import json
import time
import threading
import sys
import os
import logging
import psutil
from datetime import datetime
from paho.mqtt import client as mqtt_client

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger("BACKBONE")


class SensorDataNormalizer:
    """
    Normalise les données Arduino brutes vers un format standard
    
    FORMAT REÇU (Arduino):
    {"dist": 189.48, "gas": 57, "temp_ext": 24.8, "hum": 90.0, "son": 213, "lum": 1, "obs": 0, "joy_x": 494, "joy_y": 497, "btn": 0, "time": "159:125"}
    
    FORMAT NORMALIZE:
    {"temperature": 24.8, "humidity": 90.0, "gas": 57, "distance": 189.48, "sound": 213, "light": 1, ...}
    """
    
    @staticmethod
    def normalize(raw_data):
        """Normaliser les données Arduino brutes"""
        try:
            normalized = {
                'timestamp': datetime.now().isoformat(),
                # Capteurs principaux (correspondance)
                'temperature': raw_data.get('temp_ext') or raw_data.get('t'),
                'humidity': raw_data.get('hum') or raw_data.get('h'),
                'gas': raw_data.get('gas') or raw_data.get('g'),
                'flame': raw_data.get('f', 0),
                'distance': raw_data.get('dist') or raw_data.get('d'),
                
                # Capteurs additionnels
                'sound': raw_data.get('son'),
                'light': raw_data.get('lum'),
                'obstacle': raw_data.get('obs'),
                'joystick_x': raw_data.get('joy_x'),
                'joystick_y': raw_data.get('joy_y'),
                'button': raw_data.get('btn'),
                'sensor_time': raw_data.get('time'),
                
                'valid': True
            }
            
            # Validation basique
            if normalized['temperature'] is not None:
                if not (-50 <= normalized['temperature'] <= 100):
                    logger.warning(f"⚠️  Température hors limites: {normalized['temperature']}")
            
            if normalized['humidity'] is not None:
                if not (0 <= normalized['humidity'] <= 100):
                    logger.warning(f"⚠️  Humidité hors limites: {normalized['humidity']}")
            
            if normalized['gas'] is not None:
                if normalized['gas'] < 0:
                    logger.warning(f"⚠️  Valeur gaz invalide: {normalized['gas']}")
            
            return normalized
        
        except Exception as e:
            logger.error(f"❌ Erreur normalisation: {e}")
            return None


class SHOS_Backbone:
    """
    Kernel du système S.H.O.S - VERSION FINALE ULTIME
    - Écoute les données MQTT brutes du Bridge
    - Normalise et valide les données (CORRECTED)
    - Détecte les alertes
    - Gère ESP32 et capteurs téléphone
    - Compatible paho-mqtt 1.x et 2.x
    """
    
    def __init__(self):
        self.running = True
        self.mqtt_client = None
        self.last_sensor_data = None
        self.last_heartbeat = time.time()
        self.hardware_connected = True
        self.watchdog_timeout = 10
        self.max_cpu_percent = 85.0
        
        # Initialiser MQTT
        self._init_mqtt()
        logger.info("🚀 Backbone initialisé")
    
    def _init_mqtt(self):
        """Initialiser MQTT avec support paho-mqtt 1.x et 2.x"""
        try:
            # Essayer avec CallbackAPIVersion (2.x), sinon fallback (1.x)
            try:
                self.mqtt_client = mqtt_client.Client(
                    mqtt_client.CallbackAPIVersion.VERSION2,
                    "SHOS_BACKBONE"
                )
            except AttributeError:
                # Fallback pour paho-mqtt 1.x
                self.mqtt_client = mqtt_client.Client("SHOS_BACKBONE")
            
            self.mqtt_client.on_connect = self._on_mqtt_connect
            self.mqtt_client.on_message = self._on_mqtt_message
            self.mqtt_client.on_disconnect = self._on_mqtt_disconnect
            self.mqtt_client.max_inflight_messages_set(20)
            self.mqtt_client.connect("localhost", 1883, 60)
            self.mqtt_client.loop_start()
            
            logger.info("✅ MQTT Client démarré")
        
        except Exception as e:
            logger.error(f"❌ Erreur MQTT init: {e}")
            sys.exit(1)
    
    def _on_mqtt_connect(self, client, userdata, flags, rc, properties=None):
        """Callback connexion MQTT"""
        if rc == 0:
            logger.info("📡 Connecté au broker MQTT")
            client.subscribe([
                ("shos/sensors/raw", 1),         # Arduino Bridge
                ("shos/esp32/telemetry", 1),    # ESP32
                ("shos/mobile/sensors", 1),     # Téléphone
                ("shos/system/reset", 0),
                ("nova/benchmark/ping", 1),
            ])
            client.publish("shos/system/status", "backbone_online", qos=1, retain=True)
        else:
            logger.error(f"❌ Erreur connexion MQTT (code {rc})")
    
    def _on_mqtt_disconnect(self, client, userdata, rc, properties=None):
        """Callback déconnexion"""
        if rc != 0:
            logger.warning(f"⚠️  Déconnexion MQTT (code {rc})")
    
    def _on_mqtt_message(self, client, userdata, msg):
        """Traiter les messages MQTT"""
        topic = msg.topic
        payload = msg.payload.decode('utf-8')
        
        try:
            
            if topic == "nova/benchmark/ping":
                # Effet miroir immédiat
                self.mqtt_client.publish("nova/benchmark/pong", payload)
                return
            
            
            # ✅ DONNÉES ARDUINO (avec normalisation correcte!)
            if topic == "shos/sensors/raw":
                raw_data = json.loads(payload)
                self._process_sensor_data(raw_data)
            
            # ESP32 Telemetry
            elif topic == "shos/esp32/telemetry":
                esp32_data = json.loads(payload)
                logger.info(f"📡 ESP32: {esp32_data}")
                self.mqtt_client.publish("shos/sensors/esp32", payload, qos=1)
            
            # Mobile Sensors
            elif topic == "shos/mobile/sensors":
                mobile_data = json.loads(payload)
                logger.info(f"📱 Téléphone: {mobile_data}")
                self.mqtt_client.publish("shos/sensors/mobile", payload, qos=1)
            
            # System Reset
            elif topic == "shos/system/reset":
                logger.warning("🔄 Reset système")
                self._system_reset()
        
        except json.JSONDecodeError:
            logger.error(f"❌ JSON invalide de {topic}")
        except Exception as e:
            logger.error(f"❌ Erreur traitement: {e}")
    
    def _process_sensor_data(self, raw_data):
        """Traiter et normaliser les données"""
        logger.debug(f"📥 Données brutes: {raw_data}")
        
        # ✅ NORMALISATION CORRECTE
        normalized = SensorDataNormalizer.normalize(raw_data)
        
        if not normalized:
            return
        
        self.last_sensor_data = normalized
        self.last_heartbeat = time.time()
        self.hardware_connected = True
        
        # Publier les données normalisées
        self.mqtt_client.publish(
            "shos/sensors/normalized",
            json.dumps(normalized),
            qos=1
        )
        
        # 🎉 AFFICHAGE CORRECT (plus de None!)
        logger.info(f"✅ T={normalized['temperature']}°C | "
                   f"H={normalized['humidity']}% | "
                   f"G={normalized['gas']}ppm | "
                   f"DIST={normalized['distance']}cm")
        
        self._check_alerts(normalized)
    
    def _check_alerts(self, data):
        """Analyser les alertes"""
        
        # Alerte gaz
        if data.get('gas') and data['gas'] > 400:
            logger.critical(f"🔥 ALERTE: GAZ = {data['gas']}ppm")
            self.mqtt_client.publish(
                "shos/alert/critical",
                json.dumps({'type': 'gas', 'value': data['gas']}),
                qos=2
            )
        
        # Alerte flamme
        if data.get('flame') == 1:
            logger.critical("🔥 ALERTE: FLAMME DÉTECTÉE!")
            self.mqtt_client.publish(
                "shos/alert/critical",
                json.dumps({'type': 'flame'}),
                qos=2
            )
    
    def _system_reset(self):
        """Réinitialiser le système"""
        self.last_sensor_data = None
        self.hardware_connected = False
        time.sleep(2)
        logger.info("✅ Système réinitialisé")
    
    # ... tes autres fonctions ...

    def watchdog_loop(self):
        # Cette ligne doit être décalée par rapport au "def"
        logger.info("👁️ Watchdog & Resource Monitor activé")
        
        while self.running:
            # 1. Vérification Hardware (ton code existant)
            elapsed = time.time() - self.last_heartbeat
            
            # 2. Vérification Ressources CPU/RAM
            cpu_usage = psutil.cpu_percent(interval=None)
            ram_usage = psutil.virtual_memory().percent
            
            if cpu_usage > self.max_cpu_percent:
                logger.warning(f"⚠️ SURCHARGE CPU: {cpu_usage}%")
                self.mqtt_client.publish("nova/system/overload", json.dumps({
                    "cpu": cpu_usage,
                    "ram": ram_usage,
                    "action": "limit_modules"
                }))

            # Publication de la télémétrie
            self.mqtt_client.publish("nova/system/telemetry", json.dumps({
                "cpu": cpu_usage,
                "ram": ram_usage,
                "temp": self._get_pi_temp()
            }))
            
            time.sleep(2)

    def _get_pi_temp(self):
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                return int(f.read()) / 1000
        except:
            return 0
    
    def run(self):
        """Démarrer le système"""
        logger.info("""
        ╔═══════════════════════════════════════╗
        ║  S.H.O.S BACKBONE - VERSION ULTIME    ║
        ║  ✅ Bugs fixés                        ║
        ║  ✅ Compatible Windows + Raspberry    ║
        ║  ✅ Support ESP32 + Téléphone         ║
        ╚═══════════════════════════════════════╝
        """)
        
        try:
            threading.Thread(target=self.watchdog_loop, daemon=True).start()
            logger.info("✅ Tous les services lancés")
            
            while self.running:
                time.sleep(1)
        
        except KeyboardInterrupt:
            logger.info("\n🛑 Arrêt du backbone...")
            self.running = False
            time.sleep(1)
            
            if self.mqtt_client:
                self.mqtt_client.publish(
                    "shos/system/status",
                    "offline",
                    qos=1,
                    retain=True
                )
                self.mqtt_client.loop_stop()


if __name__ == "__main__":
    backbone = SHOS_Backbone()
    backbone.run()
