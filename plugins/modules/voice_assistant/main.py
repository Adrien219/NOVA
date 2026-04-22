
import os
import sys
import json
import time
import datetime
import subprocess
import psutil
import threading
import paho.mqtt.client as mqtt
from vosk import Model, KaldiRecognizer
import pyaudio
from pathlib import Path
import re
import unicodedata
from flask_socketio import emit

# ══════════════════════════════════════════════════════════════
#  CONFIGURATION & CONSTANTES
# ══════════════════════════════════════════════════════════════
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPICS = [
    ("shos/sensors/normalized", 1), 
    ("shos/alert/critical", 2),
    ("shos/sensors/vision", 1),
    ("helmet/plugins/vision/data", 1)
]

MODEL_PATH = "model"
USER_NAME = "Adrien"
ASSISTANT_NAME = "NOVA"
STORAGE_DIR = os.path.expanduser("~/NOVA_FILES")

# ══════════════════════════════════════════════════════════════
#  GRAMMAIRE NORMALISÉE (SANS ACCENTS POUR VOSK)
# ══════════════════════════════════════════════════════════════
# Vosk a du mal avec les accents dans la grammaire JSON.
# On utilise des mots simplifiés pour la reconnaissance.
GRAMMAR_WORDS = [
    "temperature", "degre", "chaud", "froid", "humidite", "humide",
    "obstacle", "distance", "devant", "proche", "gaz", "air", "danger", "alerte",
    "fichier", "note", "dossier", "memoire", "cree", "fais", "nouveau", "liste",
    "heure", "temps", "statut", "systeme", "ram", "cpu", "etat",
    "volume", "son", "augmente", "baisse", "fort", "moins",
    "eteins", "quitter", "stop", "revoir", "nova", "adrien",
    "vois", "regarde", "vision", "objet", "quoi", "est", "ce", "que", "tu",
    "[unk]"
]

# ══════════════════════════════════════════════════════════════
#  UTILITAIRES
# ══════════════════════════════════════════════════════════════
def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def auto_detect_audio():
    p = pyaudio.PyAudio()
    input_idx = None
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if dev.get('maxInputChannels') > 0 and input_idx is None:
            input_idx = i
            break
    p.terminate()
    return input_idx

# ══════════════════════════════════════════════════════════════
#  CLASSE PRINCIPALE : NOVA OMNISCIENT V2
# ══════════════════════════════════════════════════════════════
class NovaOmniscientV2:
    def __init__(self, gui_callback=None):
        self.gui_callback = gui_callback
        self.running = True  # Correction : Initialisation de l'état de la boucle
        
        os.makedirs(STORAGE_DIR, exist_ok=True)
        
        self.memory = {
            "sensors": {},
            "vision": [],
            "last_update": 0
        }
        
        self.input_idx = auto_detect_audio()
        self.setup_mqtt()

        if not os.path.exists(MODEL_PATH):
            print(f"❌ Erreur : Dossier '{MODEL_PATH}' introuvable.")
            sys.exit(1)
            
        self.model = Model(MODEL_PATH)
        # On passe la grammaire simplifiée (sans accents)
        self.rec = KaldiRecognizer(self.model, 16000, json.dumps(GRAMMAR_WORDS))

    def setup_mqtt(self):
        try:
            self.client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
            self.client.on_connect = self._on_connect
            self.client.on_message = self._on_message
            self.client.connect(MQTT_BROKER, MQTT_PORT, 60)
            self.client.loop_start()
        except Exception as e:
            print(f"⚠️ Erreur MQTT : {e}")

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        print(f"📡 [MQTT] Connecté au réseau NOVA (Code: {rc})")
        client.subscribe(MQTT_TOPICS)

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            topic = msg.topic
            if "sensors" in topic:
                self.memory["sensors"].update(payload)
                self.memory["last_update"] = time.time()
            elif "vision" in topic:
                if isinstance(payload, list): self.memory["vision"] = payload
                elif isinstance(payload, dict) and "objects" in payload:
                    self.memory["vision"] = [obj["class"] for obj in payload["objects"]]
                elif isinstance(payload, dict) and "found" in payload:
                    self.memory["vision"] = payload["found"]
            elif "alert" in topic:
                self._handle_alert(payload)
        except: pass

    def _handle_alert(self, alert):
        msg = f"ALERTE ! {alert.get('type', 'Danger')} détecté."
        self.parler(msg)

    def parler(self, texte):
        print(f"[{ASSISTANT_NAME}]: {texte}")
        if sys.platform == "win32":
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.say(texte); engine.runAndWait()
            except: pass
        else:
            subprocess.run(["espeak-ng", "-v", "fr", "-s", "160", texte], stderr=subprocess.DEVNULL)

    def _fuzzy_match(self, texte, mots_cles):
        # On compare le texte normalisé (sans accents) aux mots-clés
        t = remove_accents(texte.lower())
        return any(mot in t for mot in mots_cles)

    def executer_commande(self, texte):
        t = remove_accents(texte.lower())
        
        # 1. VISION
        if self._fuzzy_match(t, ["vois", "regarde", "vision", "objet", "quoi"]):
            objets = self.memory["vision"]
            if objets:
                counts = {}
                for o in objets: counts[o] = counts.get(o, 0) + 1
                reponse = "Je vois : " + ", ".join([f"{v} {k}" for k, v in counts.items()])
                self.parler(reponse)
            else:
                self.parler("Je ne vois rien de particulier.")

        # 2. CAPTEURS DYNAMIQUES
        elif self._fuzzy_match(t, ["temperature", "humidite", "gaz", "distance", "obstacle", "air"]):
            found = False
            mapping = {
                "temperature": ["temperature", "temp", "t"],
                "humidite": ["humidity", "hum", "h"],
                "gaz": ["gas", "g", "smoke"],
                "distance": ["distance", "dist", "d"],
                "obstacle": ["distance", "dist", "obstacle"]
            }
            for key, aliases in mapping.items():
                if key in t:
                    for alias in aliases:
                        if alias in self.memory["sensors"]:
                            val = self.memory["sensors"][alias]
                            unit = "degres" if "temp" in key else "pour cent" if "hum" in key else "centimetres" if "dist" in key else "ppm"
                            self.parler(f"La valeur de {key} est de {val} {unit}.")
                            found = True; break
            if not found: self.parler("Donnée indisponible.")

        # 3. FICHIERS
        elif self._fuzzy_match(t, ["fichier", "note", "dossier"]):
            if self._fuzzy_match(t, ["cree", "fais", "nouveau"]):
                nom = f"note_{datetime.datetime.now().strftime('%H%M_%S')}.txt"
                with open(os.path.join(STORAGE_DIR, nom), "w", encoding="utf-8") as f:
                    f.write(f"Note NOVA : {texte}")
                self.parler(f"Fichier {nom} cree.")
            elif self._fuzzy_match(t, ["liste", "voir"]):
                nb = len(os.listdir(STORAGE_DIR))
                self.parler(f"Il y a {nb} fichiers.")

        # 4. SYSTÈME
        elif self._fuzzy_match(t, ["heure", "temps"]):
            self.parler(datetime.datetime.now().strftime("Il est %H heures %M."))
        elif self._fuzzy_match(t, ["statut", "systeme", "ram", "cpu"]):
            ram, cpu = psutil.virtual_memory().percent, psutil.cpu_percent()
            self.parler(f"RAM {ram}%, CPU {cpu}%.")
        elif self._fuzzy_match(t, ["volume", "son"]):
            if self._fuzzy_match(t, ["augmente", "fort"]):
                self.parler("Volume augmente.")
                if sys.platform != "win32": os.system("amixer set Master 10%+")
            elif self._fuzzy_match(t, ["baisse", "moins"]):
                self.parler("Volume baisse.")
                if sys.platform != "win32": os.system("amixer set Master 10%-")
        elif self._fuzzy_match(t, ["eteins", "quitter", "stop"]):
            self.parler(f"Arret du systeme. Au revoir {USER_NAME}.")
            self.running = False; sys.exit()

    def run(self):
        p = pyaudio.PyAudio()
        try:
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, 
                            input_device_index=self.input_idx, frames_per_buffer=8000)
            stream.start_stream()
            
            # Envoi du statut initial au HUD via MQTT
            self.client.publish("shos/plugins/voice_assistant/data", json.dumps({
                "plugin": "voice_assistant", 
                "status": "ÉCOUTE ACTIVE"
            }))
            
            self.parler(f"Systeme NOVA V2 active. Je vous ecoute {USER_NAME}.")
            
            while self.running:
                data = stream.read(4000, exception_on_overflow=False)
                if self.rec.AcceptWaveform(data):
                    res = json.loads(self.rec.Result())
                    phrase = res.get("text", "")
                    if phrase:
                        # Publication de la transcription pour l'interface HUD
                        self.client.publish("shos/plugins/voice_assistant/data", json.dumps({
                            "plugin": "voice_assistant",
                            "type": "transcription",
                            "value": phrase
                        }))
                        
                        if self.gui_callback:
                            self.gui_callback("transcription", phrase)
                        
                        print(f"🎙️ Entendu : {phrase}")
                        self.executer_commande(phrase)
        except Exception as e: print(f"❌ Erreur : {e}")
        finally: p.terminate()

# ══════════════════════════════════════════════════════════════
#  INTERFACE DE PLUGIN (À AJOUTER ICI)
# ══════════════════════════════════════════════════════════════

class ModuleApp:
    def __init__(self, name):
        self.name = name
        # Remplace 'TonDetecteur' par le nom de ta classe principale (ex: GestureDetector)
        self.logic = GestureDetector() 

    def send_to_ui(self, type, message):
        """Envoie les données au dashboard via MQTT"""
        import paho.mqtt.publish as publish
        import json
        # Modifie "mon_module" par l'ID du module (ex: "hand")
        payload = {"module": "mon_module", "type": type, "value": message}
        publish.single("shos/modules/mon_module/data", json.dumps(payload), hostname="localhost")

    def run(self):
        """Boucle de fonctionnement du module"""
        # Appelle ici la méthode de démarrage de ton module
        self.logic.run()



if __name__ == "__main__":
    NovaOmniscientV2().run()
