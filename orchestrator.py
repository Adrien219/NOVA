import paho.mqtt.client as mqtt
import json

class NovaOrchestrator:
    def __init__(self):
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.current_mode = "idle"
        
        # Définition des modules à activer par mode
        self.modes_config = {
            "blind_assist": ["danger_monitor", "voice_assistant", "text_reader"],
            "remote_expert": ["camera_stream", "voice_assistant"],
            "diagnostic": ["sensor_deep_scan", "voice_assistant"]
        }

    def on_connect(self, client, userdata, flags, rc, properties=None):
        print("🧠 Orchestrateur NOVA connecté")
        client.subscribe("shos/system/set_mode")

    def on_message(self, client, userdata, msg):
        if msg.topic == "shos/system/set_mode":
            new_mode = msg.payload.decode()
            self.switch_mode(new_mode)

    def switch_mode(self, mode_name):
        if mode_name in self.modes_config:
            print(f"🔄 Basculement vers le mode : {mode_name}")
            self.current_mode = mode_name
            
            # On informe tous les modules de leur nouvel état
            for plugin, active_plugins in self.modes_config.items():
                status = "start" if plugin in active_plugins else "stop"
                self.client.publish(f"shos/plugins/{plugin}/control", status)
            
            # Feedback au HUD
            self.client.publish("shos/system/status", json.dumps({"active_mode": mode_name}))

    def run(self):
        self.client.connect("localhost", 1883)
        self.client.loop_forever()

if __name__ == "__main__":
    orchestrator = NovaOrchestrator()
    orchestrator.run()