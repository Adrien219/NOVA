#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
 S.H.O.S - ORCHESTRATEUR IA HYBRIDE V2
===============================================================================
 Architecture : regles expertes + 3 modeles d'IA specialises.
 Declenche sur evenement MQTT (pas de boucle continue).

   1. Module Manager   (Decision Tree)  - quels modules activer
   2. Energy Optimizer (3 MLP TFLite)   - CPU freq + FPS + sleep modules
   3. Profile Predictor (k-NN k=5)      - suggerer un profil

 Strategie hybride :
   - Confiance IA >= 60% -> decision IA appliquee
   - Confiance IA <  60% -> fallback sur regles expertes
   - Les regles tournent TOUJOURS en parallele (securite + comparaison)

 Topics MQTT ecoutes (declencheurs) :
   shos/sensors/normalized       - etat capteurs
   nova/system/telemetry         - CPU/RAM/Temp
   shos/system/event             - evenements systeme
   shos/system/set_mode          - forcage manuel
   shos/camera/vision            - meta vision
   shos/system/profile/active    - profil actuellement actif

 Topics MQTT publies (decisions) :
   shos/orchestrator/decision    - decision complete
   shos/orchestrator/modules     - modules a activer/desactiver
   shos/orchestrator/energy      - reglages CPU/FPS/sleep
   shos/orchestrator/profile     - profil suggere
   shos/system/status            - etat orchestrateur
===============================================================================
"""

import os
import sys
import json
import time
import logging
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Tuple, List

import numpy as np
import paho.mqtt.client as mqtt

try:
    from paho.mqtt.enums import CallbackAPIVersion
    _MQTT_V2 = True
except ImportError:
    _MQTT_V2 = False

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR   = PROJECT_ROOT / "shos_models"

MQTT_BROKER  = "localhost"
MQTT_PORT    = 1883

CONFIDENCE_THRESHOLD = 0.60
DEBOUNCE_MS          = 500

SUBSCRIPTIONS = [
    ("shos/sensors/normalized",    1),
    ("nova/system/telemetry",      1),
    ("shos/system/event",          1),
    ("shos/system/set_mode",       1),
    ("shos/camera/vision",         0),
    ("shos/system/profile/active", 1),
]

T_DECISION = "shos/orchestrator/decision"
T_MODULES  = "shos/orchestrator/modules"
T_ENERGY   = "shos/orchestrator/energy"
T_PROFILE  = "shos/orchestrator/profile"
T_STATUS   = "shos/system/status"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger("Orchestrator")


# =============================================================================
# COUCHE 1 - REGLES EXPERTES (fallback toujours actif)
# =============================================================================
class ExpertRules:

    @staticmethod
    def decide_modules(s):
        modules = []
        if s.get("distance", 999) < 50 or s.get("gas", 0) > 350 or s.get("vibration", 0) > 0.7:
            modules.append("danger_monitor")
        if s.get("environment") == "exterieur" and s.get("context") in ("matin", "midi"):
            modules.append("vision_objet")
        if s.get("light", 0) > 150 and s.get("activity") in ("repos", "interaction"):
            modules.append("text_reader")
        if s.get("context") == "nuit" or s.get("battery", 0) > 60:
            modules.append("voice_assistant")
        if s.get("activity") == "transport" and s.get("battery", 0) > 40 and s.get("cpu_pct", 100) < 70:
            modules.append("video_streaming")
        if s.get("activity") == "interaction" and s.get("context") != "nuit":
            modules.append("hand_control")
        if s.get("cpu_pct", 100) < 60 and s.get("temp_cpu", 100) < 70:
            modules.append("mediapipe_gesture")
        if s.get("activity") == "transport":
            modules.append("joystick")
        if "danger_monitor" in modules and s.get("gas", 0) > 500:
            modules.append("remote")
        return modules if modules else ["voice_assistant"]

    @staticmethod
    def decide_energy(s):
        bat  = s.get("battery", 100)
        tcpu = s.get("temp_cpu", 50)
        cpu  = s.get("cpu_pct", 30)
        act  = s.get("activity", "repos")

        if bat < 20 or tcpu > 75:
            cpu_freq = "LOW"
        elif bat < 50 or tcpu > 65 or cpu > 80:
            cpu_freq = "MED"
        else:
            cpu_freq = "HIGH"

        if   bat < 25:               fps = 10
        elif bat < 50 or tcpu > 70:  fps = 15
        elif act == "transport":     fps = 25
        elif act == "repos":         fps = 15
        else:                        fps = 20

        if   bat < 15 or tcpu > 78:  sleep = "vision+streaming"
        elif bat < 30:               sleep = "streaming"
        elif cpu > 85:               sleep = "vision"
        else:                        sleep = "none"
        return cpu_freq, fps, sleep

    @staticmethod
    def decide_profile(s):
        h   = s.get("hour", 12)
        ctx = s.get("context", "midi")
        act = s.get("activity", "repos")
        env = s.get("environment", "interieur")
        wkn = s.get("is_weekend", 0)

        if act == "transport" and not wkn and ctx in ("matin", "soir"):
            return "profile_1"
        if not wkn and ctx == "midi" and env == "interieur" and act == "interaction":
            return "profile_2"
        if env == "exterieur" and ctx in ("matin", "midi") and act != "repos":
            return "profile_3"
        if ctx == "soir" and act in ("repos", "interaction"):
            return "profile_4"
        if ctx == "nuit":
            return "profile_5"
        if wkn and act in ("marche", "interaction"):
            return "profile_6"
        return "profile_7"


# =============================================================================
# COUCHE 2 - MODELES IA
# =============================================================================
class AIModels:

    def __init__(self, models_dir=MODELS_DIR):
        self.dir = models_dir
        self.ready = {"modules": False, "energy": False, "profile": False}
        self._joblib = None
        self._tflite = None
        self.mod_pkg     = None
        self.eng_pre     = None
        self.eng_interps = {}
        self.prof_pkg    = None
        self._load()

    def _import_libs(self):
        try:
            import joblib
            self._joblib = joblib
        except ImportError:
            logger.warning("joblib non disponible - modeles .pkl desactives")

        try:
            import tflite_runtime.interpreter as tflite
            self._tflite = tflite
        except ImportError:
            try:
                from tensorflow.lite.python.interpreter import Interpreter as _Interp
                class _T:
                    Interpreter = _Interp
                self._tflite = _T()
            except ImportError:
                logger.warning("tflite_runtime/tensorflow non disponible - TFLite desactive")

    def _load(self):
        self._import_libs()
        if not self.dir.exists():
            logger.warning("Dossier modeles introuvable : %s", self.dir)
            return

        if self._joblib:
            try:
                p = self.dir / "module_manager.pkl"
                if p.exists():
                    self.mod_pkg = self._joblib.load(p)
                    self.ready["modules"] = True
                    logger.info("Module Manager charge (%d features -> %d sorties)",
                                len(self.mod_pkg["features"]), len(self.mod_pkg["targets"]))
            except Exception as e:
                logger.error("Module Manager : %s", e)

        if self._tflite:
            try:
                pre = self.dir / "energy_preprocessing.pkl"
                if pre.exists() and self._joblib:
                    self.eng_pre = self._joblib.load(pre)
                for target in ["cpu_freq", "fps_target", "sleep_modules"]:
                    tpath = self.dir / ("energy_" + target + ".tflite")
                    if tpath.exists():
                        interp = self._tflite.Interpreter(model_path=str(tpath))
                        interp.allocate_tensors()
                        self.eng_interps[target] = interp
                if len(self.eng_interps) == 3 and self.eng_pre:
                    self.ready["energy"] = True
                    logger.info("Energy Optimizer charge (3 MLP TFLite)")
            except Exception as e:
                logger.error("Energy Optimizer : %s", e)

        if self._joblib:
            try:
                p = self.dir / "profile_predictor.pkl"
                if p.exists():
                    self.prof_pkg = self._joblib.load(p)
                    self.ready["profile"] = True
                    logger.info("Profile Predictor charge (k=%d)",
                                self.prof_pkg["model"].n_neighbors)
            except Exception as e:
                logger.error("Profile Predictor : %s", e)

    def predict_modules(self, s):
        if not self.ready["modules"]:
            return [], 0.0
        try:
            pkg = self.mod_pkg
            features = pkg["features"]
            encoders = pkg["encoders"]
            row = []
            for f in features:
                v = s.get(f, 0)
                if f in encoders:
                    le = encoders[f]
                    try:
                        v = le.transform([str(v)])[0]
                    except ValueError:
                        v = 0
                row.append(float(v))
            X = np.array([row], dtype=np.float32)
            preds_proba = pkg["model"].predict_proba(X)
            preds = pkg["model"].predict(X)[0]
            mods = [pkg["targets"][i].replace("mod_", "")
                    for i, v in enumerate(preds) if v == 1]
            confidences = []
            for proba in preds_proba:
                if proba.shape[1] >= 2:
                    confidences.append(float(proba[0].max()))
            conf = float(np.mean(confidences)) if confidences else 0.0
            return mods, conf
        except Exception as e:
            logger.warning("predict_modules: %s", e)
            return [], 0.0

    def predict_energy(self, s):
        if not self.ready["energy"] or not self.eng_pre:
            return None, None, None, 0.0
        try:
            pre = self.eng_pre
            features = pre["features"]
            scaler = pre["scaler"]
            target_encoders = pre["target_encoders"]
            feat_encoders = pre["feature_encoders"]
            row = []
            for f in features:
                v = s.get(f, 0)
                if f in feat_encoders:
                    le = feat_encoders[f]
                    try:
                        v = le.transform([str(v)])[0]
                    except ValueError:
                        v = 0
                row.append(float(v))
            X = np.array([row], dtype=np.float32)
            X_scaled = scaler.transform(X).astype(np.float32)
            results = {}
            confidences = []
            for target, interp in self.eng_interps.items():
                inp = interp.get_input_details()
                out = interp.get_output_details()
                interp.set_tensor(inp[0]['index'], X_scaled)
                interp.invoke()
                proba = interp.get_tensor(out[0]['index'])[0]
                idx = int(np.argmax(proba))
                confidences.append(float(proba[idx]))
                le = target_encoders[target]
                results[target] = le.inverse_transform([idx])[0]
            cpu_freq = str(results.get("cpu_freq", "MED"))
            fps_str  = str(results.get("fps_target", "20"))
            try:
                fps = int(fps_str)
            except ValueError:
                fps = 20
            sleep = str(results.get("sleep_modules", "none"))
            return cpu_freq, fps, sleep, float(np.mean(confidences))
        except Exception as e:
            logger.warning("predict_energy: %s", e)
            return None, None, None, 0.0

    def predict_profile(self, s):
        if not self.ready["profile"]:
            return None, 0.0
        try:
            pkg = self.prof_pkg
            features = pkg["features"]
            scaler = pkg["scaler"]
            encs = pkg["feature_encoders"]
            le = pkg["profile_encoder"]
            row = []
            for f in features:
                v = s.get(f, 0)
                if f in encs:
                    try:
                        v = encs[f].transform([str(v)])[0]
                    except ValueError:
                        v = 0
                row.append(float(v))
            X = np.array([row], dtype=np.float32)
            X_scaled = scaler.transform(X)
            proba = pkg["model"].predict_proba(X_scaled)[0]
            idx = int(np.argmax(proba))
            conf = float(proba[idx])
            profile = le.inverse_transform([idx])[0]
            return profile, conf
        except Exception as e:
            logger.warning("predict_profile: %s", e)
            return None, 0.0


# =============================================================================
# COUCHE 3 - ETAT SYSTEME AGREGE
# =============================================================================
class SystemState:

    def __init__(self):
        self.lock = threading.Lock()
        self._state = {}
        self.last_decision_ts = 0.0
        self.last_decision_payload = {}

    def update(self, data):
        with self.lock:
            self._state.update(data)
            now = datetime.now()
            h = now.hour
            self._state["hour"] = h
            self._state["weekday_int"] = now.weekday()
            self._state["is_weekend"] = int(now.weekday() >= 5)
            if   5  <= h < 11: self._state["context"] = "matin"
            elif 11 <= h < 17: self._state["context"] = "midi"
            elif 17 <= h < 22: self._state["context"] = "soir"
            else:              self._state["context"] = "nuit"

    def snapshot(self):
        with self.lock:
            return dict(self._state)


# =============================================================================
# COUCHE 4 - ORCHESTRATEUR
# =============================================================================
class NovaOrchestrator:

    def __init__(self, threshold=CONFIDENCE_THRESHOLD):
        self.threshold = threshold
        self.state = SystemState()
        self.rules = ExpertRules()
        self.ai = AIModels()

        if _MQTT_V2:
            self.client = mqtt.Client(CallbackAPIVersion.VERSION2, client_id="NOVA_ORCHESTRATOR")
        else:
            self.client = mqtt.Client(client_id="NOVA_ORCHESTRATOR")
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

        self.stats = {
            "decisions_total":  0,
            "decisions_ai":     0,
            "decisions_rules":  0,
            "started_at":       datetime.now().isoformat(),
        }

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        rc_code = rc if isinstance(rc, int) else rc.value
        if rc_code == 0:
            logger.info("Orchestrateur NOVA connecte a MQTT")
            for topic, qos in SUBSCRIPTIONS:
                client.subscribe(topic, qos=qos)
            logger.info("Abonne a %d topics", len(SUBSCRIPTIONS))
            self._publish_status("online")
        else:
            logger.error("MQTT connexion refusee (rc=%d)", rc_code)

    def _on_message(self, client, userdata, msg):
        topic = msg.topic
        try:
            payload = json.loads(msg.payload.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = {"raw": msg.payload.decode('utf-8', errors='replace')}

        if topic == "shos/sensors/normalized":
            self.state.update(payload)
            self._trigger_decision(reason="sensor_change")

        elif topic == "nova/system/telemetry":
            self.state.update({
                "cpu_pct":  payload.get("cpu",  0),
                "ram_pct":  payload.get("ram",  0),
                "temp_cpu": payload.get("temp", 0),
            })
            if payload.get("cpu", 0) > 80 or payload.get("temp", 0) > 70:
                self._trigger_decision(reason="resource_pressure")

        elif topic == "shos/system/event":
            self._trigger_decision(reason="system_event")

        elif topic == "shos/system/set_mode":
            mode = payload.get("mode") if isinstance(payload, dict) else str(payload)
            logger.info("Forcage manuel mode : %s", mode)
            self._trigger_decision(reason="manual_mode", force_profile=mode)

        elif topic == "shos/camera/vision":
            self.state.update({
                "hands_detected": payload.get("hands_detected", 0),
                "objects_count":  payload.get("objects_count",  0),
            })

        elif topic == "shos/system/profile/active":
            self.state.update({"active_profile": payload.get("profile_id", "")})

    def _trigger_decision(self, reason, force_profile=None):
        now = time.time()
        if (now - self.state.last_decision_ts) * 1000 < DEBOUNCE_MS:
            return

        s = self.state.snapshot()

        rule_mods = self.rules.decide_modules(s)
        ai_mods, ai_mods_conf = self.ai.predict_modules(s)
        if ai_mods_conf >= self.threshold and ai_mods:
            final_mods, mods_source = ai_mods, "ai"
        else:
            final_mods, mods_source = rule_mods, "rules"

        rule_cpu, rule_fps, rule_sleep = self.rules.decide_energy(s)
        ai_cpu, ai_fps, ai_sleep, ai_eng_conf = self.ai.predict_energy(s)
        if ai_eng_conf >= self.threshold and ai_cpu is not None:
            final_cpu, final_fps, final_sleep = ai_cpu, ai_fps, ai_sleep
            energy_source = "ai"
        else:
            final_cpu, final_fps, final_sleep = rule_cpu, rule_fps, rule_sleep
            energy_source = "rules"

        if force_profile:
            final_profile, profile_source, ai_prof_conf = force_profile, "manual", 1.0
        else:
            rule_prof = self.rules.decide_profile(s)
            ai_prof, ai_prof_conf = self.ai.predict_profile(s)
            if ai_prof_conf >= self.threshold and ai_prof:
                final_profile, profile_source = ai_prof, "ai"
            else:
                final_profile, profile_source = rule_prof, "rules"

        self.stats["decisions_total"] += 1
        if "ai" in (mods_source, energy_source, profile_source):
            self.stats["decisions_ai"] += 1
        else:
            self.stats["decisions_rules"] += 1

        decision = {
            "reason":  reason,
            "ts":      now,
            "modules": {
                "activate":   final_mods,
                "source":     mods_source,
                "confidence": round(ai_mods_conf, 3),
            },
            "energy": {
                "cpu_freq":      final_cpu,
                "fps_target":    final_fps,
                "sleep_modules": final_sleep,
                "source":        energy_source,
                "confidence":    round(ai_eng_conf, 3),
            },
            "profile": {
                "suggested":  final_profile,
                "source":     profile_source,
                "confidence": round(ai_prof_conf, 3),
            },
            "state_snapshot": {k: s.get(k) for k in [
                "context", "activity", "battery", "cpu_pct", "temp_cpu",
                "distance", "gas", "light"
            ]},
        }

        self._publish(T_DECISION, decision)
        self._publish(T_MODULES, decision["modules"])
        self._publish(T_ENERGY,  decision["energy"])
        self._publish(T_PROFILE, decision["profile"])

        ai_count = sum(1 for src in [mods_source, energy_source, profile_source] if src == "ai")
        logger.info(
            "Decision [%s] | mods:%d(%s) cpu:%s fps:%s sleep:%s | profil:%s(%s) | IA:%d/3",
            reason, len(final_mods), mods_source, final_cpu, final_fps, final_sleep,
            final_profile, profile_source, ai_count
        )

        self.state.last_decision_ts = now
        self.state.last_decision_payload = decision

    def _publish(self, topic, payload):
        try:
            self.client.publish(topic, json.dumps(payload), qos=0)
        except Exception as e:
            logger.warning("publish %s: %s", topic, e)

    def _publish_status(self, status):
        self._publish(T_STATUS, {
            "component":     "orchestrator",
            "status":        status,
            "models_loaded": self.ai.ready,
            "threshold":     self.threshold,
            "stats":         self.stats,
            "ts":            time.time(),
        })

    def run(self):
        logger.info("=" * 70)
        logger.info(" NOVA Orchestrator V2 - Demarrage")
        logger.info("   Seuil confiance IA : %.0f%%", self.threshold * 100)
        logger.info("   Modeles charges    : %s", self.ai.ready)
        logger.info("   Mode               : Event-driven (MQTT)")
        logger.info("=" * 70)
        try:
            self.client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
            self.client.loop_forever()
        except KeyboardInterrupt:
            logger.info("Arret orchestrateur")
            self._publish_status("offline")
            self.client.disconnect()
        except Exception as e:
            logger.error("Erreur fatale : %s", e)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SHOS Orchestrator V2 - Hybride regles + IA")
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD,
                        help="Seuil de confiance pour decision IA (defaut 0.60)")
    parser.add_argument("--models", type=str, default=str(MODELS_DIR),
                        help="Dossier des modeles")
    args = parser.parse_args()

    orch = NovaOrchestrator(threshold=args.threshold)
    if args.models != str(MODELS_DIR):
        orch.ai = AIModels(models_dir=Path(args.models))
    orch.run()