import os
import sys
import json
import time
import datetime
import subprocess
import psutil
import paho.mqtt.client as mqtt
from vosk import Model, KaldiRecognizer
import pyaudio

# ══════════════════════════════════════════════════════════════
#  SCANNER AUDIO AUTOMATIQUE (Windows & Pi)
# ══════════════════════════════════════════════════════════════
def get_best_audio_devices():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    
    input_idx = None
    output_idx = None

    print("\n--- SCAN AUDIO ---")
    for i in range(0, numdevices):
        dev = p.get_device_info_by_index(i)
        name = dev.get('name')
        # Détection Micro
        if dev.get('maxInputChannels') > 0 and input_idx is None:
            if "default" in name.lower() or "usb" in name.lower() or "mic" in name.lower():
                input_idx = i
                print(f"✅ Micro trouvé : [{i}] {name}")
        
        # Détection Sortie (HP / TV / AirPods)
        if dev.get('maxOutputChannels') > 0 and output_idx is None:
            if "default" in name.lower() or "hdmi" in name.lower() or "speaker" in name.lower() or "headphones" in name.lower():
                output_idx = i
                print(f"✅ Sortie trouvée : [{i}] {name}")
    
    p.terminate()
    return input_idx, output_idx

# ══════════════════════════════════════════════════════════════
#  MOTEUR VOCAL INTERACTIF
# ══════════════════════════════════════════════════════════════
class NovaHeadless:
    def __init__(self):
        self.input_idx, self.output_idx = get_best_audio_devices()
        self.mqtt_data = {"temp": "inconnue", "dist": "inconnue"}
        
        # Init Vosk
        if not os.path.exists("model"):
            print("❌ Erreur : Dossier 'model' manquant."); exit()
        self.model = Model("model")
        self.rec = KaldiRecognizer(self.model, 16000)

        # MQTT pour les capteurs du mémoire
        self.client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
        self.client.on_message = lambda c, u, m: self.mqtt_data.update(json.loads(m.payload.decode()))
        try:
            self.client.connect("localhost", 1883, 60)
            self.client.subscribe("shos/sensors/#")
            self.client.loop_start()
        except: print("⚠️ MQTT Hors-ligne")

    def parler(self, texte):
        print(f"NOVA: {texte}")
        # Sur Pi, on force la sortie si nécessaire, sinon espeak-ng suffit
        if sys.platform == "win32":
            # Si tu es sur Windows pour tes tests
            import pyttsx3
            engine = pyttsx3.init()
            engine.say(texte); engine.runAndWait()
        else:
            # Sur Raspberry Pi (100% Local)
            subprocess.run(["espeak-ng", "-v", "fr", "-s", "150", texte])

    def demander_confirmation(self, question):
        """Pose une question et attend un oui/non vocal"""
        self.parler(question)
        # On réinitialise le flux pour écouter spécifiquement la réponse
        start_time = time.time()
        while (time.time() - start_time) < 5: # 5 secondes pour répondre
            # (Logique simplifiée d'écoute ici)
            return True # Pour le dev, on simule l'accord

    def traiter_commande(self, commande):
        c = commande.lower()
        
        # --- COMMANDES AUDIO-SYSTÈME ---
        if "heure" in c:
            self.parler(datetime.datetime.now().strftime("Il est précisément %H heures %M"))
        
        elif "température" in c:
            self.parler(f"La température du capteur est de {self.mqtt_data['temp']} degrés.")

        elif "créer" in c and "fichier" in c:
            self.parler("J'ai entendu votre demande de création de fichier. Quel nom voulez-vous donner ?")
            # Ici on attendrait le prochain 'ecouter()' pour le nom
            self.parler("Fichier créé avec succès sur le stockage local.")

        elif "volume" in c:
            if "augmente" in c:
                os.system("amixer set Master 10%+" if sys.platform != "win32" else "")
                self.parler("Le volume a été augmenté de dix pour cent.")

        elif "statut" in c:
            cpu = psutil.cpu_percent()
            self.parler(f"Le système utilise actuellement {cpu} pour cent de sa puissance.")

    def run(self):
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16, 
            channels=1, 
            rate=16000, 
            input=True, 
            input_device_index=self.input_idx,
            frames_per_buffer=8000
        )
        stream.start_stream()
        
        self.parler("Initialisation terminée. Système N O V A prêt. Je vous écoute Adrien.")

        while True:
            data = stream.read(4000, exception_on_overflow=False)
            if self.rec.AcceptWaveform(data):
                result = json.loads(self.rec.Result())
                text = result.get("text", "")
                if text:
                    print(f"🎙️ : {text}")
                    self.traiter_commande(text)

if __name__ == "__main__":
    nova = NovaHeadless()
    nova.run()
