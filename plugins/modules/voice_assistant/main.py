#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════════════
 S.H.O.S · VOICE ASSISTANT V3 — Pilotable + Lance modules
═══════════════════════════════════════════════════════════════════════════════
 Améliorations vs V2 :
   • Détection automatique micro / haut-parleur + publication sur MQTT
   • Pilotable par d'autres modules via 'shos/modules/+/cmd'
     (hand_control, hardware_controls, etc. peuvent activer/pauser/lancer)
   • Lance d'autres modules vocalement via 'shos/system/launch_module'
   • Pause/Resume thread-safe (l'orchestrateur peut couper l'écoute)
   • Publication enrichie sur shos/plugins/voice_assistant/data

 Topics ÉCOUTÉS :
   shos/plugins/voice_assistant/cmd  → commandes directes (start/stop/speak)
   shos/modules/+/cmd                 → commandes inter-modules
   shos/sensors/normalized            → mémoire capteurs (pour répondre)
   shos/camera/vision                 → mémoire vision (pour répondre)
   shos/alert/critical                → alertes à annoncer

 Topics PUBLIÉS :
   shos/plugins/voice_assistant/data  → transcription + status pour UI
   shos/plugins/voice_assistant/audio → device micro/HP actuel
   shos/system/launch_module          → demande lancement module
   shos/system/kill_module            → demande arrêt module
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import time
import datetime
import subprocess
import threading
import unicodedata
import logging
from pathlib import Path

import psutil
import paho.mqtt.client as mqtt
from vosk import Model, KaldiRecognizer

# Backend audio adaptatif : pyaudio (Pi/Linux) ou sounddevice (Windows)
_AUDIO_BACKEND = None
try:
    import pyaudio as _pyaudio
    _AUDIO_BACKEND = "pyaudio"
except ImportError:
    try:
        import sounddevice as _sd
        import numpy as _np
        _AUDIO_BACKEND = "sounddevice"
    except ImportError:
        pass

if _AUDIO_BACKEND is None:
    print("Aucun backend audio. Installez pyaudio (Pi) ou sounddevice (Windows)")
    import sys; sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MQTT_BROKER = "localhost"
MQTT_PORT   = 1883

# Topics écoutés
SUBSCRIPTIONS = [
    ("shos/plugins/voice_assistant/cmd", 1),
    ("shos/modules/+/cmd",                1),
    ("shos/sensors/normalized",           1),
    ("shos/camera/vision",                1),
    ("shos/alert/critical",               2),
]

# Topics publiés
T_DATA   = "shos/plugins/voice_assistant/data"
T_AUDIO  = "shos/plugins/voice_assistant/audio"
T_LAUNCH = "shos/system/launch_module"
T_KILL   = "shos/system/kill_module"

# Chemins
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH   = PROJECT_ROOT / "model"
STORAGE_DIR  = Path(os.path.expanduser("~/NOVA_FILES"))

USER_NAME      = "Adrien"
ASSISTANT_NAME = "NOVA"

# Modules connus (pour le lancement vocal)
KNOWN_MODULES = {
    "vision":         "vision_objet",
    "detection":      "vision_objet",
    "objets":         "vision_objet",
    "camera":         "vision_objet",
    "lecture":        "text_reader",
    "texte":          "text_reader",
    "lire":           "text_reader",
    "danger":         "danger_monitor",
    "alerte":         "danger_monitor",
    "surveillance":   "danger_monitor",
    "geste":          "hand_control",
    "gestes":         "hand_control",
    "main":           "hand_control",
    "stream":         "video_streaming",
    "streaming":      "video_streaming",
    "video":          "video_streaming",
    "diffusion":      "video_streaming",
}

# Grammaire Vosk (sans accents pour fiabilité)
GRAMMAR_WORDS = [
    "temperature", "degre", "chaud", "froid", "humidite", "humide",
    "obstacle", "distance", "devant", "proche", "gaz", "air", "danger", "alerte",
    "fichier", "note", "dossier", "memoire", "cree", "fais", "nouveau", "liste",
    "heure", "temps", "statut", "systeme", "ram", "cpu", "etat",
    "volume", "son", "augmente", "baisse", "fort", "moins",
    "eteins", "quitter", "stop", "revoir", "nova", "adrien",
    "vois", "regarde", "vision", "objet", "quoi", "est", "ce", "que", "tu",
    "active", "lance", "demarre", "arrete", "ferme", "detection",
    "lecture", "texte", "surveillance", "geste", "gestes", "main",
    "stream", "streaming", "video", "diffusion", "camera",
    "pause", "reprends", "continue",
    "[unk]",
    "nova"
]

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] VoiceV3 - %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger("VoiceV3")


# ─────────────────────────────────────────────────────────────────────────────
# UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────
def remove_accents(text: str) -> str:
    nfkd = unicodedata.normalize('NFKD', text)
    return "".join([c for c in nfkd if not unicodedata.combining(c)])


def detect_audio_devices() -> dict:
    """
    Détecte les périphériques audio — pyaudio (Pi) ou sounddevice (Windows).
    Retourne { mic: {...}, speaker: {...}, all_inputs: [...], all_outputs: [...] }
    """
    inputs, outputs = [], []
    mic = None
    speaker = None

    if _AUDIO_BACKEND == "pyaudio":
        p = _pyaudio.PyAudio()
        default_in = default_out = None
        try:
            default_in  = p.get_default_input_device_info().get("index")
            default_out = p.get_default_output_device_info().get("index")
        except Exception:
            pass
        for i in range(p.get_device_count()):
            try:
                info = p.get_device_info_by_index(i)
                entry = {
                    "index": i,
                    "name":  info.get("name", f"device_{i}"),
                    "channels_in":  int(info.get("maxInputChannels", 0)),
                    "channels_out": int(info.get("maxOutputChannels", 0)),
                    "rate":         int(info.get("defaultSampleRate", 0)),
                }
                if entry["channels_in"] > 0:  inputs.append(entry)
                if entry["channels_out"] > 0: outputs.append(entry)
            except Exception:
                continue
        p.terminate()
        mic     = next((e for e in inputs  if e["index"] == default_in),  None) or (inputs[0]  if inputs  else None)
        speaker = next((e for e in outputs if e["index"] == default_out), None) or (outputs[0] if outputs else None)

    elif _AUDIO_BACKEND == "sounddevice":
        try:
            devices = _sd.query_devices()
            default_in  = _sd.default.device[0]
            default_out = _sd.default.device[1]
            for i, d in enumerate(devices):
                entry = {
                    "index": i,
                    "name":  d["name"],
                    "channels_in":  int(d.get("max_input_channels",  0)),
                    "channels_out": int(d.get("max_output_channels", 0)),
                    "rate":         int(d.get("default_samplerate",  0)),
                }
                if entry["channels_in"] > 0:  inputs.append(entry)
                if entry["channels_out"] > 0: outputs.append(entry)
            # Sur Windows, préférer WDM-KS (plus fiable que MME pour Vosk)
            wdm_in  = [e for e in inputs  if "WDM-KS" in e["name"] and e["channels_in"] > 0]
            wdm_out = [e for e in outputs if "WDM-KS" in e["name"] and e["channels_out"] > 0]
            mic     = wdm_in[0]  if wdm_in  else (next((e for e in inputs  if e["index"] == default_in),  None) or (inputs[0]  if inputs  else None))
            speaker = wdm_out[0] if wdm_out else (next((e for e in outputs if e["index"] == default_out), None) or (outputs[0] if outputs else None))
        except Exception as e:
            logger.warning("sounddevice query : %s", e)

    return {
        "mic":         mic,
        "speaker":     speaker,
        "all_inputs":  inputs,
        "all_outputs": outputs,
    }


# ═════════════════════════════════════════════════════════════════════════════
# CLASSE PRINCIPALE — VOICE ASSISTANT V3
# ═════════════════════════════════════════════════════════════════════════════
class VoiceAssistantV3:
    def __init__(self):
        STORAGE_DIR.mkdir(exist_ok=True)

        # État
        self.running = True
        self.listening = True   # peut être mis en pause via cmd
        self.pause_lock = threading.Lock()

        # Mémoire pour les réponses contextuelles
        self.memory = {
            "sensors":     {},
            "vision":      [],
            "last_update": 0,
        }

        # État "question en attente" (NOVA attend une réponse à sa propre question)
        self._pending_question = None   # None ou str décrivant la question posée
        self._verbose = True            # True = réponses complètes / False = bref

        # Mot d'éveil (mode Jarvis/Alexa)
        # États : "sleeping" (écoute mot éveil) | "awake" (attend commande)
        self.wake_mode = True          # True = mode éveil activé
        self.wake_state = "sleeping"   # état courant
        self.wake_timer = None         # timer retour sommeil après inactivité

        # Périphériques audio
        self.audio_devices = detect_audio_devices()
        if not self.audio_devices["mic"]:
            logger.error("❌ Aucun micro détecté")
            sys.exit(1)

        # Init TTS persistant (Windows)
        self._tts_engine = None
        self._tts_queue = __import__('queue').Queue()
        self._speaking = False  # True pendant que NOVA parle → ignore le micro
        self._init_tts()
        # Thread TTS dédié — résout le problème Windows (pyttsx3 thread-unsafe)
        tts_thread = __import__('threading').Thread(target=self._tts_worker, daemon=True)
        tts_thread.start()
        logger.info("🎤 Micro : %s", self.audio_devices["mic"]["name"])
        logger.info("🔈 HP   : %s",
                    self.audio_devices["speaker"]["name"] if self.audio_devices["speaker"] else "—")

        # MQTT
        self.client = None
        self._setup_mqtt()

        # Vosk
        if not MODEL_PATH.exists():
            logger.error("❌ Dossier modèle Vosk introuvable : %s", MODEL_PATH)
            sys.exit(1)
        logger.info("📥 Chargement Vosk : %s", MODEL_PATH)
        self.model = Model(str(MODEL_PATH))
        self.rec   = KaldiRecognizer(self.model, 16000, json.dumps(GRAMMAR_WORDS))

        # Publication initiale du device audio
        self._publish_audio_info()

    # ── MQTT ────────────────────────────────────────────────────────────────
    def _setup_mqtt(self):
        try:
            try:
                from paho.mqtt.enums import CallbackAPIVersion
                self.client = mqtt.Client(CallbackAPIVersion.VERSION2)
            except ImportError:
                self.client = mqtt.Client()
            self.client.on_connect = self._on_connect
            self.client.on_message = self._on_message
            self.client.connect(MQTT_BROKER, MQTT_PORT, 60)
            self.client.loop_start()
        except Exception as e:
            logger.warning("⚠ MQTT : %s", e)

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        logger.info("📡 MQTT connecté (rc=%s)", rc)
        client.subscribe(SUBSCRIPTIONS)
        # Publish status initial
        self._publish_data({"type": "status", "value": "ready", "status": "EN ATTENTE"})

    def _on_message(self, client, userdata, msg):
        topic = msg.topic
        try:
            payload = json.loads(msg.payload.decode())
        except Exception:
            payload = {"raw": msg.payload.decode(errors='replace')}

        # Capteurs / vision mémoire
        if "sensors" in topic:
            if isinstance(payload, dict):
                self.memory["sensors"].update(payload)
                self.memory["last_update"] = time.time()
        elif "camera/vision" in topic:
            if isinstance(payload, dict):
                if "detections" in payload:
                    self.memory["vision"] = [d.get("class","obj") for d in payload["detections"]]
                elif "objects" in payload:
                    self.memory["vision"] = [o.get("class","obj") for o in payload["objects"]]
        elif "alert/critical" in topic:
            self._handle_alert(payload)

        # Commandes directes (voice_assistant/cmd)
        elif topic == "shos/plugins/voice_assistant/cmd":
            self._handle_cmd(payload)

        # Commandes inter-modules (shos/modules/+/cmd)
        elif topic.startswith("shos/modules/") and topic.endswith("/cmd"):
            source_module = topic.split("/")[2]
            # On accepte seulement si le payload est destiné à voice_assistant
            if not isinstance(payload, dict):
                return
            target = payload.get("target", payload.get("module", ""))
            if target and target != "voice_assistant" and target != "voice":
                return
            self._handle_cmd(payload, source=source_module)

    def _handle_cmd(self, payload, source="direct"):
        """Gère les commandes reçues sur MQTT."""
        if not isinstance(payload, dict):
            return
        cmd = str(payload.get("cmd", "")).lower().strip()

        if cmd in ("start", "resume", "listen"):
            with self.pause_lock:
                self.listening = True
            logger.info("▶ Écoute activée (par %s)", source)
            self._publish_data({"type": "status", "value": "listening", "status": "ÉCOUTE ACTIVE"})

        elif cmd in ("stop", "pause"):
            with self.pause_lock:
                self.listening = False
            logger.info("⏸ Écoute en pause (par %s)", source)
            self._publish_data({"type": "status", "value": "paused", "status": "EN PAUSE"})

        elif cmd == "speak":
            text = str(payload.get("text", "")).strip()
            if text:
                self.parler(text)

        elif cmd == "list_files":
            try:
                nb = len(list(STORAGE_DIR.iterdir()))
                self.parler(f"Il y a {nb} fichiers")
            except Exception:
                self.parler("Erreur lecture fichiers")

        elif cmd == "audio_info":
            self._publish_audio_info()

        else:
            logger.debug("Commande inconnue : %s (par %s)", cmd, source)

    # ── Publications ────────────────────────────────────────────────────────
    def _publish_data(self, data: dict):
        if not self.client:
            return
        try:
            full = {"plugin": "voice_assistant", **data, "ts": time.time()}
            self.client.publish(T_DATA, json.dumps(full), qos=0)
        except Exception as e:
            logger.debug("publish data : %s", e)

    def _publish_audio_info(self):
        if not self.client:
            return
        try:
            mic = self.audio_devices["mic"]
            spk = self.audio_devices["speaker"]
            payload = {
                "mic":     mic["name"] if mic else None,
                "mic_idx": mic["index"] if mic else None,
                "speaker": spk["name"] if spk else None,
                "speaker_idx": spk["index"] if spk else None,
                "inputs_count":  len(self.audio_devices["all_inputs"]),
                "outputs_count": len(self.audio_devices["all_outputs"]),
                "ts": time.time(),
            }
            self.client.publish(T_AUDIO, json.dumps(payload), qos=1, retain=True)
            # Aussi dans data pour que l'UI le voie au chargement
            self._publish_data({"type": "audio", "audio": payload})
        except Exception as e:
            logger.debug("publish audio : %s", e)

    def _launch_module(self, module_id: str):
        """Demande à l'orchestrateur de lancer un module."""
        if not self.client:
            return
        try:
            self.client.publish(T_LAUNCH, json.dumps({
                "module": module_id, "source": "voice", "ts": time.time()
            }))
            logger.info("🚀 Demande lancement → %s", module_id)
        except Exception as e:
            logger.debug("launch_module : %s", e)

    def _kill_module(self, module_id: str):
        if not self.client:
            return
        try:
            self.client.publish(T_KILL, json.dumps({
                "module": module_id, "source": "voice", "ts": time.time()
            }))
            logger.info("⏹ Demande arrêt → %s", module_id)
        except Exception as e:
            logger.debug("kill_module : %s", e)

    # ── Alertes ─────────────────────────────────────────────────────────────
    def _handle_alert(self, alert):
        if not isinstance(alert, dict):
            return
        msg = f"ALERTE ! {alert.get('type', 'Danger')} détecté."
        self.parler(msg)

    # ── TTS ─────────────────────────────────────────────────────────────────
    def _init_tts(self):
        """Initialise le moteur TTS persistant (Windows uniquement)."""
        if sys.platform != "win32":
            return
        try:
            import pyttsx3
            self._tts_engine = pyttsx3.init()
            # Voix française si dispo
            voices = self._tts_engine.getProperty('voices')
            for v in voices:
                if 'fr' in v.id.lower() or 'french' in v.name.lower():
                    self._tts_engine.setProperty('voice', v.id)
                    break
            self._tts_engine.setProperty('rate', 165)
            logger.info("TTS pyttsx3 initialisé")
        except Exception as e:
            logger.warning("pyttsx3 non disponible : %s", e)
            self._tts_engine = None

    def _tts_worker(self):
        """Thread TTS dedié — multi-backend Windows/Pi.
        Ordre priorité Windows :
          1. win32com SAPI.SpVoice  (natif, pip install pywin32)
          2. winsound beep          (fallback sonore)
        Pi/Linux : espeak-ng
        """
        backend = None

        if sys.platform == "win32":
            try:
                import win32com.client as _wc
                sapi = _wc.Dispatch("SAPI.SpVoice")
                voices = sapi.GetVoices()
                for i in range(voices.Count):
                    v = voices.Item(i)
                    desc = v.GetDescription()
                    if any(x in desc.lower() for x in ["fr", "french", "hortense"]):
                        sapi.Voice = v
                        break
                sapi.Rate = 1
                backend = ("sapi", sapi)
                logger.info("Thread TTS pret (SAPI win32com)")
            except Exception as e:
                logger.warning("win32com SAPI : %s", e)
                backend = ("beep", None)
                logger.info("Thread TTS : mode beep (pip install pywin32 pour la voix)")
        else:
            backend = ("espeak", None)
            logger.info("Thread TTS pret (espeak-ng)")

        while True:
            try:
                texte = self._tts_queue.get(timeout=1.0)
                if texte is None:
                    break

                self._speaking = True
                try:
                    kind, engine = backend
                    if kind == "sapi":
                        engine.Speak(texte)
                    elif kind == "espeak":
                        subprocess.run(
                            ["espeak-ng", "-v", "fr", "-s", "160", texte],
                            stderr=subprocess.DEVNULL, timeout=15,
                        )
                    else:
                        import winsound
                        winsound.Beep(880, 200)
                        logger.info("(beep) %s", texte)
                except Exception as ex:
                    logger.warning("TTS speak : %s", ex)
                finally:
                    time.sleep(0.35)
                    self._speaking = False
            except Exception:
                continue


    def parler(self, texte: str):
        """Envoie le texte au thread TTS dédié (non-bloquant)."""
        logger.info("🔊 %s", texte)
        self._publish_data({"type": "speaking", "value": texte})
        try:
            self._tts_queue.put_nowait(texte)
        except Exception as ex:
            logger.warning("TTS queue : %s", ex)

    # ── Logique de commande ─────────────────────────────────────────────────
    def _fuzzy_match(self, texte, mots_cles):
        t = remove_accents(texte.lower())
        return any(mot in t for mot in mots_cles)

    def _parler_bref(self, complet, bref):
        """Parle en version complète ou brève selon self._verbose."""
        self.parler(bref if not self._verbose else complet)

    def _recherche_duckduckgo(self, query: str) -> str:
        """Recherche DuckDuckGo — retourne le résumé (Instant Answer API)."""
        try:
            import urllib.request, urllib.parse
            url = "https://api.duckduckgo.com/?q=" + urllib.parse.quote(query) + "&format=json&no_html=1&skip_disambig=1"
            req = urllib.request.Request(url, headers={"User-Agent": "NOVA-SHOS/3.0"})
            with urllib.request.urlopen(req, timeout=5) as r:
                data = json.loads(r.read().decode())
            abstract = data.get("AbstractText", "").strip()
            if abstract:
                return abstract[:300]
            # Fallback : premiers related topics
            topics = data.get("RelatedTopics", [])
            for t in topics:
                if isinstance(t, dict) and t.get("Text"):
                    return t["Text"][:200]
            return ""
        except Exception as e:
            logger.debug("DuckDuckGo : %s", e)
            return ""

    def _recherche_wikipedia(self, query: str) -> str:
        """Recherche Wikipedia FR — retourne le premier paragraphe."""
        try:
            import urllib.request, urllib.parse
            url = ("https://fr.wikipedia.org/w/api.php?action=query&prop=extracts"
                   "&exintro=1&explaintext=1&titles=" + urllib.parse.quote(query)
                   + "&format=json&redirects=1")
            req = urllib.request.Request(url, headers={"User-Agent": "NOVA-SHOS/3.0"})
            with urllib.request.urlopen(req, timeout=5) as r:
                data = json.loads(r.read().decode())
            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                extract = page.get("extract", "")
                if extract:
                    # Premier paragraphe seulement
                    para = extract.split("\n")[0].strip()
                    return para[:300] if para else ""
            return ""
        except Exception as e:
            logger.debug("Wikipedia : %s", e)
            return ""

    def _sauvegarder_note(self, texte: str) -> str:
        """Sauvegarde une note dans le fichier Markdown mémoire."""
        try:
            STORAGE_DIR.mkdir(exist_ok=True)
            note_file = STORAGE_DIR / "memoire_nova.md"
            now = datetime.datetime.now()
            header = f"\n## {now.strftime('%d/%m/%Y')}\n" if not note_file.exists() else ""
            entry = f"- **{now.strftime('%H:%M')}** — {texte}\n"
            # Ajouter header de date si nouveau jour
            if note_file.exists():
                existing = note_file.read_text(encoding="utf-8")
                today_header = f"## {now.strftime('%d/%m/%Y')}"
                if today_header not in existing:
                    header = f"\n## {now.strftime('%d/%m/%Y')}\n"
            with open(note_file, "a", encoding="utf-8") as f:
                if header:
                    f.write(header)
                f.write(entry)
            return note_file.name
        except Exception as e:
            logger.warning("Note : %s", e)
            return "erreur"

    def executer_commande(self, texte: str):
        t = remove_accents(texte.lower())

        # ── RÉPONSE À UNE QUESTION EN ATTENTE ─────────────────────────────
        if self._pending_question:
            q = self._pending_question
            self._pending_question = None

            if q == "module_a_activer":
                # L'utilisateur répond à "quel module activer ?"
                for mot_cle, module_id in KNOWN_MODULES.items():
                    if mot_cle in t:
                        self._parler_bref(
                            f"Activation du module {mot_cle}",
                            f"Module {mot_cle}"
                        )
                        self._launch_module(module_id)
                        return
                self.parler("Je n'ai pas reconnu ce module")
                return

            elif q == "note_contenu":
                # L'utilisateur dicte le contenu à mémoriser
                fname = self._sauvegarder_note(texte)
                self._parler_bref(f"Noté dans {fname}", "Noté")
                return

        # ── TAIRE NOVA ────────────────────────────────────────────────────
        # Mots-clés précis pour éviter les faux positifs
        if self._fuzzy_match(t, ["tais toi", "silence", "chut"]):
            try:
                while not self._tts_queue.empty():
                    self._tts_queue.get_nowait()
            except Exception:
                pass
            self._speaking = False
            return

        # ── MODE BREF ─────────────────────────────────────────────────────
        if self._fuzzy_match(t, ["sois bref", "plus court", "moins long", "reponds court"]):
            self._verbose = False
            self.parler("Compris")
            return

        # ── VERBOSE ON ────────────────────────────────────────────────────
        if self._fuzzy_match(t, ["sois detail", "plus complet", "explique mieux"]):
            self._verbose = True
            self.parler("Bien")
            return

        # ── SALUTATIONS ───────────────────────────────────────────────────
        if self._fuzzy_match(t, ["bonjour", "bonsoir", "salut", "hello", "hi"]):
            now_h = datetime.datetime.now().hour
            if now_h < 12:
                self._parler_bref(f"Bonjour {USER_NAME}, comment puis-je vous aider ?", f"Bonjour {USER_NAME}")
            else:
                self._parler_bref(f"Bonsoir {USER_NAME}, à votre service", f"Bonsoir")
            return

        # ── PRÉSENTATION ─────────────────────────────────────────────────
        if self._fuzzy_match(t, ["qui es", "presente", "comment tu", "c est quoi", "que fais"]):
            self._parler_bref(
                f"Je suis NOVA, l'assistant vocal du système S.H.O.S. "
                f"Je suis intégré aux lunettes intelligentes et je peux contrôler "
                f"les modules, répondre à vos questions et surveiller vos capteurs. "
                f"Je suis conçu spécialement pour ce système.",
                "Je suis NOVA, l'assistant du système S.H.O.S"
            )
            return

        # ── MÉMORISER ─────────────────────────────────────────────────────
        if self._fuzzy_match(t, ["retiens", "memorise", "note", "rappelle", "souviens"]):
            # Extraire ce qu'il faut noter après "retiens que / note que"
            for trigger in ["retiens que", "memorise que", "note que", "rappelle que", "souviens que",
                            "retiens", "memorise", "note", "rappelle"]:
                tr = remove_accents(trigger)
                if tr in t:
                    contenu = t.split(tr, 1)[-1].strip()
                    if len(contenu) > 3:
                        fname = self._sauvegarder_note(contenu)
                        self._parler_bref(f"Mémorisé dans {fname}", "Mémorisé")
                        return
            # Pas de contenu direct → demander
            self._pending_question = "note_contenu"
            self.parler("Que dois-je mémoriser ?")
            self._reset_wake_timer()
            return

        # ── RECHERCHE EN LIGNE ────────────────────────────────────────────
        if self._fuzzy_match(t, ["cherche", "recherche", "trouve", "qu est ce que", "c est quoi", "definition", "qui est"]):
            # Extraire la requête
            for trigger in ["cherche", "recherche", "trouve", "qu est ce que", "c est quoi", "definition de", "qui est"]:
                tr = remove_accents(trigger)
                if tr in t:
                    query = t.split(tr, 1)[-1].strip()
                    if len(query) < 2:
                        query = texte
                    break
            else:
                query = texte

            self.parler("Je cherche")
            # Wikipedia d'abord pour les définitions
            result = ""
            if self._fuzzy_match(t, ["definition", "c est quoi", "qu est ce que"]):
                result = self._recherche_wikipedia(query)
            if not result:
                result = self._recherche_duckduckgo(query)
            if not result:
                result = self._recherche_wikipedia(query)

            if result:
                # Tronquer à 2 phrases pour la voix
                sentences = result.replace(".", ".||").split("||")
                short = " ".join(s.strip() for s in sentences[:2] if s.strip())
                self._parler_bref(short or result[:200], result[:100])
            else:
                self.parler("Je n'ai pas trouvé de résultat")
            return

        # ── ACTIVER / LANCER MODULES ─────────────────────────────────────
        if self._fuzzy_match(t, ["active", "lance", "demarre", "ouvre"]):
            for mot_cle, module_id in KNOWN_MODULES.items():
                if mot_cle in t:
                    self._parler_bref(
                        f"Activation du module {mot_cle}",
                        f"Module {mot_cle}"
                    )
                    self._launch_module(module_id)
                    return
            # Pas de module reconnu → demander (question en attente)
            self._pending_question = "module_a_activer"
            self.parler("Quel module dois-je activer ?")
            self._reset_wake_timer()
            return

        # ── ARRÊTER MODULES ──────────────────────────────────────────────
        if self._fuzzy_match(t, ["arrete", "ferme"]) and not self._fuzzy_match(t, ["eteins", "quitter"]):
            for mot_cle, module_id in KNOWN_MODULES.items():
                if mot_cle in t:
                    self.parler(f"Arrêt du module {mot_cle}")
                    self._kill_module(module_id)
                    return

        # ── VISION ────────────────────────────────────────────────────────
        if self._fuzzy_match(t, ["vois", "regarde", "objet", "quoi"]):
            objets = self.memory["vision"]
            if objets:
                counts = {}
                for o in objets:
                    counts[o] = counts.get(o, 0) + 1
                reponse = "Je vois : " + ", ".join([f"{v} {k}" for k, v in counts.items()])
                self.parler(reponse)
            else:
                self.parler("Je ne vois rien de particulier")
            return

        # ── CAPTEURS ──────────────────────────────────────────────────────
        if self._fuzzy_match(t, ["temperature", "humidite", "gaz", "distance", "obstacle", "air"]):
            mapping = {
                "temperature": (["temperature", "temp", "t"], "degrés"),
                "humidite":    (["humidity", "hum", "h"],     "pour cent"),
                "gaz":         (["gas", "g", "smoke"],         "ppm"),
                "distance":    (["distance", "dist", "d"],     "centimètres"),
            }
            for key, (aliases, unit) in mapping.items():
                if key in t:
                    for alias in aliases:
                        if alias in self.memory["sensors"]:
                            val = self.memory["sensors"][alias]
                            self.parler(f"La valeur de {key} est de {val} {unit}")
                            return
            self.parler("Donnée capteur indisponible")
            return

        # ── FICHIERS ──────────────────────────────────────────────────────
        if self._fuzzy_match(t, ["fichier", "note", "dossier"]):
            if self._fuzzy_match(t, ["cree", "fais", "nouveau"]):
                nom = f"note_{datetime.datetime.now().strftime('%H%M_%S')}.txt"
                with open(STORAGE_DIR / nom, "w", encoding="utf-8") as f:
                    f.write(f"Note NOVA : {texte}")
                self.parler(f"Fichier {nom} créé")
            elif self._fuzzy_match(t, ["liste", "voir"]):
                nb = len(list(STORAGE_DIR.iterdir()))
                self.parler(f"Il y a {nb} fichiers")
            return

        # ── SYSTÈME ───────────────────────────────────────────────────────
        if self._fuzzy_match(t, ["heure", "temps"]):
            self.parler(datetime.datetime.now().strftime("Il est %H heures %M"))
            return
        if self._fuzzy_match(t, ["statut", "systeme", "ram", "cpu"]):
            ram = psutil.virtual_memory().percent
            cpu = psutil.cpu_percent()
            self.parler(f"RAM {ram:.0f} pour cent, CPU {cpu:.0f} pour cent")
            return

        # ── AIDE ──────────────────────────────────────────────────────────
        if self._fuzzy_match(t, ["aide", "help", "commandes", "que peux"]):
            self._parler_bref(
                "Voici mes commandes : heure, statut système, lance un module, "
                "arrête un module, mémorise quelque chose, cherche sur internet, "
                "volume, tais-toi, sois bref, bonjour.",
                "heure, statut, lance, mémorise, cherche, volume"
            )
            return

        # ── VOLUME ────────────────────────────────────────────────────────
        if self._fuzzy_match(t, ["volume", "son"]):
            if self._fuzzy_match(t, ["augmente", "fort"]):
                self.parler("Volume augmenté")
                if sys.platform != "win32":
                    os.system("amixer set Master 10%+ > /dev/null 2>&1")
            elif self._fuzzy_match(t, ["baisse", "moins"]):
                self.parler("Volume baissé")
                if sys.platform != "win32":
                    os.system("amixer set Master 10%- > /dev/null 2>&1")
            return

        # ── ARRÊT SYSTÈME ─────────────────────────────────────────────────
        if self._fuzzy_match(t, ["eteins", "quitter", "revoir"]):
            self.parler(f"Arrêt du système. Au revoir {USER_NAME}")
            self.running = False
            sys.exit()

        # ── COMMANDE NON RECONNUE ─────────────────────────────────────────
        self._parler_bref(
            f"Je n'ai pas compris cette commande. "
            f"Je suis NOVA, conçu pour le système S.H.O.S. "
            f"Je réponds aux commandes comme : heure, statut, lance un module, "
            f"mémorise, cherche, ou volume.",
            "Commande non reconnue. Dites aide pour la liste"
        )

    # ── Boucle principale ───────────────────────────────────────────────────
    def run(self):
        if _AUDIO_BACKEND == "pyaudio":
            self._run_pyaudio()
        elif _AUDIO_BACKEND == "sounddevice":
            self._run_sounddevice()
        else:
            logger.error("Aucun backend audio disponible")

    def _on_audio_data(self, phrase: str):
        """
        Appelé avec le texte reconnu par Vosk.
        Mode Jarvis :
          - sleeping  : attend "nova" → répond "oui Adrien" → passe à awake
          - awake     : exécute la commande → retour sleeping après 8s
        """
        # Ignorer si NOVA est en train de parler (évite l'écho)
        if self._speaking:
            return

        # Nettoyer les [unk] du texte
        phrase_clean = phrase.replace("[unk]", "").strip()
        if not phrase_clean:
            return

        logger.info("🎙 Entendu : %s", phrase_clean)

        if not self.wake_mode:
            # Mode sans éveil — exécute directement
            self._publish_data({"type": "transcription", "value": phrase_clean})
            self.executer_commande(phrase_clean)
            return

        # ── MODE ÉVEIL ─────────────────────────────────────────────────────
        t = remove_accents(phrase_clean.lower())

        if self.wake_state == "sleeping":
            # Cherche le mot d'éveil "nova"
            if any(w in t for w in ["nova", "no va", "novak", "noval"]):
                self.wake_state = "awake"
                self._publish_data({"type": "transcription", "value": phrase_clean})
                self._publish_data({"type": "status", "value": "listening", "status": "EN ÉCOUTE"})
                self.parler(f"Oui {USER_NAME}")
                # Retour sommeil si pas de commande dans 20s
                self._reset_wake_timer()
            # Sinon — mot éveil pas détecté, ignorer silencieusement
            return

        elif self.wake_state == "awake":
            # On a déjà été activé — exécuter la commande
            self._publish_data({"type": "transcription", "value": phrase_clean})
            if self.wake_timer:
                self.wake_timer.cancel()
            self.executer_commande(phrase_clean)
            # Si NOVA a posé une question → rester awake pour attendre la réponse
            if self._pending_question:
                self._reset_wake_timer()  # reset le timer 20s
            else:
                self._go_to_sleep()

    def _reset_wake_timer(self):
        """Lance un timer : retour en sommeil après 8s sans commande."""
        if self.wake_timer:
            self.wake_timer.cancel()
        self.wake_timer = threading.Timer(20.0, self._go_to_sleep)
        self.wake_timer.daemon = True
        self.wake_timer.start()

    def _go_to_sleep(self):
        """Retour en mode sommeil (écoute passive du mot éveil)."""
        self.wake_state = "sleeping"
        self.wake_timer = None
        self._publish_data({"type": "status", "value": "sleeping", "status": "EN ATTENTE"})
        logger.info("💤 Retour en veille — en attente de 'Nova'")

    def _run_pyaudio(self):
        """Boucle audio via pyaudio (Pi/Linux)."""
        p = _pyaudio.PyAudio()
        mic_idx = self.audio_devices["mic"]["index"]
        try:
            stream = p.open(
                format=_pyaudio.paInt16, channels=1, rate=16000, input=True,
                input_device_index=mic_idx, frames_per_buffer=8000,
            )
            stream.start_stream()
            self._publish_data({"type": "status", "value": "listening", "status": "ECOUTE ACTIVE"})
            self.parler(f"Systeme NOVA V3 activer. Je vous ecoute {USER_NAME}")

            while self.running:
                with self.pause_lock:
                    is_listening = self.listening
                if not is_listening:
                    time.sleep(0.2); continue

                data = stream.read(4000, exception_on_overflow=False)
                if self.rec.AcceptWaveform(data):
                    res = json.loads(self.rec.Result())
                    phrase = res.get("text", "").strip()
                    if phrase and phrase != "[unk]":
                        self._on_audio_data(phrase)

        except Exception as e:
            logger.error("Erreur pyaudio : %s", e)
        finally:
            try: stream.stop_stream(); stream.close()
            except Exception: pass
            p.terminate()

    def _run_sounddevice(self):
        """Boucle audio via sounddevice (Windows) avec fallback multi-device."""
        import queue
        RATE      = 16000
        BLOCKSIZE = 8000

        # Candidats : WDM-KS en priorité, puis tous les inputs
        all_inputs = self.audio_devices.get("all_inputs", [])
        wdm = [e for e in all_inputs if "WDM-KS" in e["name"] and e["channels_in"] > 0]
        candidates = wdm if wdm else all_inputs
        # Ajouter None (défaut système) comme dernier recours
        candidate_indices = [e["index"] for e in candidates] + [None]

        q = queue.Queue()
        working_idx = None

        def _callback(indata, frames, time_info, status):
            q.put(bytes(indata))

        # Tester chaque device jusqu'à ce qu'un fonctionne
        for idx in candidate_indices:
            try:
                # Test d'ouverture rapide
                with _sd.RawInputStream(
                    samplerate=RATE, blocksize=BLOCKSIZE,
                    device=idx, dtype='int16', channels=1,
                    callback=_callback,
                ) as test_stream:
                    working_idx = idx
                    name = f"index {idx}" if idx is not None else "défaut système"
                    logger.info("Micro sélectionné : %s", name)
                    # Mettre à jour l'affichage dans l'UI
                    if idx is not None:
                        mic = next((e for e in all_inputs if e["index"] == idx), None)
                        if mic:
                            self.audio_devices["mic"] = mic
                            self._publish_audio_info()
                    break
            except Exception as e:
                logger.debug("Device %s KO : %s", idx, e)
                continue

        if working_idx is None and None not in candidate_indices:
            working_idx = None  # dernier recours

        self._publish_data({"type": "status", "value": "listening", "status": "ECOUTE ACTIVE"})
        self.parler(f"Systeme NOVA V3 activer. Je vous ecoute {USER_NAME}")

        try:
            with _sd.RawInputStream(
                samplerate=RATE, blocksize=BLOCKSIZE,
                device=working_idx, dtype='int16', channels=1,
                callback=_callback,
            ):
                while self.running:
                    with self.pause_lock:
                        is_listening = self.listening
                    if not is_listening:
                        try: q.get(timeout=0.2)
                        except Exception: pass
                        continue
                    try:
                        data = q.get(timeout=1.0)
                    except Exception:
                        continue
                    if self.rec.AcceptWaveform(data):
                        res = json.loads(self.rec.Result())
                        phrase = res.get("text", "").strip()
                        if phrase and phrase != "[unk]":
                            self._on_audio_data(phrase)

        except Exception as e:
            logger.error("Erreur sounddevice : %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        VoiceAssistantV3().run()
    except KeyboardInterrupt:
        logger.info("⛔ Arrêt utilisateur")
    except Exception as e:
        logger.error("❌ %s", e)