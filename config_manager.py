import json
import os
import paho.mqtt.client as mqtt

class ConfigManager:
    def __init__(self, config_path=None):
        if config_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.config_path = os.path.join(base_dir, 'config.json')
        else:
            self.config_path = config_path
            
        # Initialisation MQTT compatible Version 2 (Raspberry Pi)
        try:
            from paho.mqtt.enums import CallbackAPIVersion
            self.mqtt_client = mqtt.Client(CallbackAPIVersion.VERSION1, "NOVA_CONFIG_MANAGER")
        except:
            self.mqtt_client = mqtt.Client("NOVA_CONFIG_MANAGER")
        
        try:
            self.mqtt_client.connect("localhost", 1883, 60)
            self.mqtt_client.loop_start()
        except:
            print("⚠️ ConfigManager : Broker MQTT injoignable.")

    def load_config(self):
        """Charge le fichier de configuration JSON"""
        if not os.path.exists(self.config_path):
            return None
        with open(self.config_path, 'r') as f:
            return json.load(f)

    def save_config(self, data):
        """Sauvegarde les modifications"""
        with open(self.config_path, 'w') as f:
            json.dump(data, f, indent=4)

    def activate_profile(self, profile_id):
        """Diffuse l'intention de changement de profil au Backbone"""
        config = self.load_config()
        if not config or profile_id not in config['profiles']:
            print(f"❌ Profil {profile_id} inconnu dans config.json.")
            return False

        profile = config['profiles'][profile_id]
        print(f"🚀 N.O.V.A : Préparation du profil '{profile['name']}'")

        # --- LOGIQUE HARMONISÉE NOVA ---
       
        payload = {
            "action": "SWITCH_PROFILE",
            "profile_id": profile_id,
            "profile_name": profile.get('name'),
            "allowed_modules": profile.get('modules', {}) # Format: {"nom": priorité}
        }

        # On publie sur le topic système de NOVA
        self.mqtt_client.publish(
            "nova/system/profile/switch", 
            json.dumps(payload), 
            qos=1, 
            retain=True
        )

        # Mise à jour du JSON local
        config['current_active_profile'] = profile_id
        self.save_config(config)
        
        print(f"✅ Ordre de switch envoyé pour : {profile['name']}")
        return True

if __name__ == "__main__":
    cm = ConfigManager()
   
    cm.activate_profile("default")