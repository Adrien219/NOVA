# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════════════
 HARDWARE_CONTROLS · ACTIONS CATALOG
 ─────────────────────────────────────────────────────────────────────────────
 Catalogue des 30 actions exécutables par les commandes physiques :
   • Modules    (8 actions)  : lancer/arrêter les modules SHOS
   • Système    (7 actions)  : reboot, shutdown, screenshot, etc.
   • Réglages   (8 actions)  : volume, luminosité, seuils alertes
   • Scripts    (3 actions)  : exécution fichiers / commandes
   • MQTT       (4 actions)  : publish messages sur le bus

 Chaque action a un ID unique, un label, une catégorie et une fonction execute().
 Les paramètres (ex: nom de fichier) sont stockés dans le binding.
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import time
import json
import shutil
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Callable, Optional

logger = logging.getLogger("SHOS.HwActions")

# ─────────────────────────────────────────────────────────────────────────────
# CONTEXTE PARTAGÉ
# ─────────────────────────────────────────────────────────────────────────────

class ActionContext:
    """
    Contexte partagé entre toutes les actions.
    Configuré une fois par hardware_controls/main.py au démarrage.
    """
    mqtt_client       = None   # paho mqtt client
    socketio          = None   # flask-socketio instance
    project_root      = None   # Path
    config_manager    = None   # ConfigManager instance
    state             = {}     # state runtime (volume, luminosité, etc.)

ctx = ActionContext()

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _publish_mqtt(topic: str, payload: dict, qos: int = 0) -> bool:
    """Publie sur MQTT si client disponible."""
    if not ctx.mqtt_client:
        logger.warning("MQTT non disponible — action ignorée")
        return False
    try:
        ctx.mqtt_client.publish(topic, json.dumps(payload), qos=qos)
        return True
    except Exception as e:
        logger.error("MQTT publish: %s", e)
        return False


def _emit_socket(event: str, data: dict):
    """Émet un event SocketIO si disponible."""
    if ctx.socketio:
        try:
            ctx.socketio.emit(event, data)
        except Exception as e:
            logger.error("SocketIO emit: %s", e)


def _safe_run(cmd: list, timeout: int = 5) -> dict:
    """Exécution sécurisée d'une commande système."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        return {
            "ok":      result.returncode == 0,
            "stdout":  result.stdout[:500],
            "stderr":  result.stderr[:500],
            "code":    result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

# ─────────────────────────────────────────────────────────────────────────────
# CATÉGORIE 1 — MODULES (8 actions)
# ─────────────────────────────────────────────────────────────────────────────

def act_launch_voice(params=None):
    _publish_mqtt("shos/system/launch_module", {"module": "voice_assistant", "ts": time.time()})
    _emit_socket("module_launch", {"module": "voice_assistant"})
    return {"ok": True, "msg": "Module Voice lancé"}

def act_launch_vision(params=None):
    _publish_mqtt("shos/system/launch_module", {"module": "vision_objet", "ts": time.time()})
    _emit_socket("module_launch", {"module": "vision_objet"})
    return {"ok": True, "msg": "Module Vision lancé"}

def act_launch_ocr(params=None):
    _publish_mqtt("shos/system/launch_module", {"module": "text_reader", "ts": time.time()})
    _emit_socket("module_launch", {"module": "text_reader"})
    return {"ok": True, "msg": "Module OCR lancé"}

def act_launch_streaming(params=None):
    _publish_mqtt("shos/system/launch_module", {"module": "video_streaming", "ts": time.time()})
    _emit_socket("module_launch", {"module": "video_streaming"})
    return {"ok": True, "msg": "Module Streaming lancé"}

def act_launch_danger(params=None):
    _publish_mqtt("shos/system/launch_module", {"module": "danger_monitor", "ts": time.time()})
    _emit_socket("module_launch", {"module": "danger_monitor"})
    return {"ok": True, "msg": "Module Danger lancé"}

def act_launch_hand(params=None):
    _publish_mqtt("shos/system/launch_module", {"module": "hand_control", "ts": time.time()})
    _emit_socket("module_launch", {"module": "hand_control"})
    return {"ok": True, "msg": "Module Hand lancé"}

def act_launch_joystick(params=None):
    _publish_mqtt("shos/system/launch_module", {"module": "joystick", "ts": time.time()})
    _emit_socket("module_launch", {"module": "joystick"})
    return {"ok": True, "msg": "Module Joystick lancé"}

def act_kill_module(params=None):
    module = (params or {}).get("module", "")
    _publish_mqtt("shos/system/kill_module", {"module": module, "ts": time.time()})
    _emit_socket("module_kill", {"module": module})
    return {"ok": True, "msg": f"Module {module} arrêté"}

# ─────────────────────────────────────────────────────────────────────────────
# CATÉGORIE 2 — SYSTÈME (7 actions)
# ─────────────────────────────────────────────────────────────────────────────

def act_reboot(params=None):
    logger.warning("⚠ Reboot système demandé")
    _publish_mqtt("shos/system/event", {"type": "reboot", "ts": time.time()})
    if os.name != 'nt':
        subprocess.Popen(["sudo", "reboot"])
    return {"ok": True, "msg": "Redémarrage en cours..."}

def act_shutdown(params=None):
    logger.warning("⚠ Shutdown système demandé")
    _publish_mqtt("shos/system/event", {"type": "shutdown", "ts": time.time()})
    if os.name != 'nt':
        subprocess.Popen(["sudo", "shutdown", "-h", "now"])
    return {"ok": True, "msg": "Arrêt en cours..."}

def act_screenshot(params=None):
    """Capture d'écran système (Pi)."""
    out_dir = ctx.project_root / "data" / "screenshots" if ctx.project_root else Path("/tmp")
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    path = out_dir / filename
    result = _safe_run(["scrot", str(path)], timeout=3)
    if result.get("ok"):
        _publish_mqtt("shos/system/screenshot", {"path": str(path), "ts": time.time()})
        return {"ok": True, "msg": f"Capture : {filename}"}
    return {"ok": False, "msg": "Échec capture (scrot manquant ?)"}

def act_snapshot_video(params=None):
    """Snapshot du flux vidéo en cours."""
    _publish_mqtt("shos/camera/snapshot_request", {"ts": time.time()})
    return {"ok": True, "msg": "Snapshot vidéo demandé"}

def act_restart_nova(params=None):
    """Redémarre le service nova.py via systemd ou kill."""
    _publish_mqtt("shos/system/event", {"type": "restart_nova", "ts": time.time()})
    if ctx.project_root:
        subprocess.Popen([sys.executable, str(ctx.project_root / "nova.py")])
    return {"ok": True, "msg": "Nova redémarré"}

def act_sync_time(params=None):
    """Synchronise l'horloge NTP."""
    result = _safe_run(["sudo", "timedatectl", "set-ntp", "true"], timeout=3)
    return {"ok": result.get("ok", False), "msg": "Horloge synchronisée" if result.get("ok") else "Échec sync"}

def act_mute_system(params=None):
    """Coupe le son système."""
    result = _safe_run(["amixer", "set", "Master", "mute"], timeout=2)
    _publish_mqtt("shos/system/audio", {"muted": True, "ts": time.time()})
    return {"ok": result.get("ok", False), "msg": "Son coupé"}

# ─────────────────────────────────────────────────────────────────────────────
# CATÉGORIE 3 — RÉGLAGES (8 actions)
# ─────────────────────────────────────────────────────────────────────────────

def act_volume_up(params=None):
    step = int((params or {}).get("step", 5))
    ctx.state["volume"] = _clamp(ctx.state.get("volume", 50) + step, 0, 100)
    _safe_run(["amixer", "sset", "Master", f"{step}%+"], timeout=2)
    _publish_mqtt("shos/settings/volume", {"value": ctx.state["volume"]})
    return {"ok": True, "msg": f"Volume : {ctx.state['volume']}%"}

def act_volume_down(params=None):
    step = int((params or {}).get("step", 5))
    ctx.state["volume"] = _clamp(ctx.state.get("volume", 50) - step, 0, 100)
    _safe_run(["amixer", "sset", "Master", f"{step}%-"], timeout=2)
    _publish_mqtt("shos/settings/volume", {"value": ctx.state["volume"]})
    return {"ok": True, "msg": f"Volume : {ctx.state['volume']}%"}

def act_brightness_up(params=None):
    step = int((params or {}).get("step", 10))
    ctx.state["brightness"] = _clamp(ctx.state.get("brightness", 70) + step, 0, 100)
    _publish_mqtt("shos/settings/brightness", {"value": ctx.state["brightness"]})
    _emit_socket("brightness_change", {"value": ctx.state["brightness"]})
    return {"ok": True, "msg": f"Luminosité : {ctx.state['brightness']}%"}

def act_brightness_down(params=None):
    step = int((params or {}).get("step", 10))
    ctx.state["brightness"] = _clamp(ctx.state.get("brightness", 70) - step, 0, 100)
    _publish_mqtt("shos/settings/brightness", {"value": ctx.state["brightness"]})
    _emit_socket("brightness_change", {"value": ctx.state["brightness"]})
    return {"ok": True, "msg": f"Luminosité : {ctx.state['brightness']}%"}

def act_gas_threshold_up(params=None):
    step = int((params or {}).get("step", 50))
    ctx.state["gas_threshold"] = _clamp(ctx.state.get("gas_threshold", 400) + step, 0, 9999)
    _publish_mqtt("shos/settings/thresholds", {"gas": ctx.state["gas_threshold"]})
    return {"ok": True, "msg": f"Seuil gaz : {ctx.state['gas_threshold']} ppm"}

def act_gas_threshold_down(params=None):
    step = int((params or {}).get("step", 50))
    ctx.state["gas_threshold"] = _clamp(ctx.state.get("gas_threshold", 400) - step, 0, 9999)
    _publish_mqtt("shos/settings/thresholds", {"gas": ctx.state["gas_threshold"]})
    return {"ok": True, "msg": f"Seuil gaz : {ctx.state['gas_threshold']} ppm"}

def act_distance_threshold_up(params=None):
    step = int((params or {}).get("step", 10))
    ctx.state["distance_threshold"] = _clamp(ctx.state.get("distance_threshold", 50) + step, 0, 500)
    _publish_mqtt("shos/settings/thresholds", {"distance": ctx.state["distance_threshold"]})
    return {"ok": True, "msg": f"Seuil distance : {ctx.state['distance_threshold']} cm"}

def act_distance_threshold_down(params=None):
    step = int((params or {}).get("step", 10))
    ctx.state["distance_threshold"] = _clamp(ctx.state.get("distance_threshold", 50) - step, 0, 500)
    _publish_mqtt("shos/settings/thresholds", {"distance": ctx.state["distance_threshold"]})
    return {"ok": True, "msg": f"Seuil distance : {ctx.state['distance_threshold']} cm"}

# ─────────────────────────────────────────────────────────────────────────────
# CATÉGORIE 4 — SCRIPTS (3 actions)
# ─────────────────────────────────────────────────────────────────────────────

def act_run_python_file(params=None):
    """
    Exécute un fichier Python du projet.
    params: { "file": "test_vision.py", "args": ["--mode", "fast"] }
    # ASVS V5.1 : path traversal bloqué, whitelist d'extensions
    """
    p = params or {}
    filename = str(p.get("file", "")).strip()
    args     = p.get("args", [])

    if not filename or not filename.endswith(".py"):
        return {"ok": False, "msg": "Fichier .py requis"}

    safe_name = Path(filename).name  # bloque path traversal
    if not ctx.project_root:
        return {"ok": False, "msg": "Project root non défini"}

    full_path = ctx.project_root / safe_name
    if not full_path.exists():
        return {"ok": False, "msg": f"Fichier introuvable : {safe_name}"}

    cmd = [sys.executable, str(full_path)] + [str(a) for a in args if isinstance(a, (str, int, float))]
    result = _safe_run(cmd, timeout=30)
    return {
        "ok":      result.get("ok", False),
        "msg":     f"Exécuté : {safe_name}",
        "output":  result.get("stdout", "")[-300:],
        "error":   result.get("stderr", "")[-200:],
    }

def act_run_shell_cmd(params=None):
    """
    Exécute une commande shell de la whitelist.
    params: { "cmd": "uptime" }
    # ASVS V5.1 : whitelist stricte, pas de shell=True
    """
    p = params or {}
    cmd_str = str(p.get("cmd", "")).strip()

    WHITELIST = {
        "uptime":         ["uptime"],
        "free":           ["free", "-h"],
        "df":             ["df", "-h"],
        "vcgencmd_temp":  ["vcgencmd", "measure_temp"],
        "ip":             ["hostname", "-I"],
        "ps_python":      ["pgrep", "-laf", "python"],
    }
    if cmd_str not in WHITELIST:
        return {"ok": False, "msg": f"Commande non autorisée : {cmd_str}"}

    result = _safe_run(WHITELIST[cmd_str], timeout=5)
    return {
        "ok":     result.get("ok", False),
        "msg":    cmd_str,
        "output": result.get("stdout", ""),
    }

def act_run_benchmark(params=None):
    """Lance benchmark.py avec mode."""
    p = params or {}
    mode = str(p.get("mode", "fps"))
    if mode not in ("fps", "system", "mqtt", "full"):
        return {"ok": False, "msg": "Mode invalide"}
    if not ctx.project_root:
        return {"ok": False, "msg": "Project root non défini"}
    bench = ctx.project_root / "benchmark.py"
    if not bench.exists():
        return {"ok": False, "msg": "benchmark.py introuvable"}
    subprocess.Popen([sys.executable, str(bench), "--mode", mode])
    return {"ok": True, "msg": f"Benchmark {mode} lancé"}

# ─────────────────────────────────────────────────────────────────────────────
# CATÉGORIE 5 — MQTT (4 actions)
# ─────────────────────────────────────────────────────────────────────────────

def act_mqtt_publish(params=None):
    """Publie un message MQTT custom. params: { topic, payload }"""
    p = params or {}
    topic = str(p.get("topic", "")).strip()[:200]
    payload = p.get("payload", {})
    if not topic:
        return {"ok": False, "msg": "Topic requis"}
    ok = _publish_mqtt(topic, payload if isinstance(payload, dict) else {"value": payload})
    return {"ok": ok, "msg": f"Publié sur {topic}"}

def act_mqtt_alert(params=None):
    """Broadcast une alerte sur shos/alerts/broadcast."""
    p = params or {}
    msg = str(p.get("message", "Alerte manuelle"))[:200]
    level = str(p.get("level", "warning"))
    _publish_mqtt("shos/alerts/broadcast", {
        "message": msg, "level": level, "source": "hardware_controls", "ts": time.time()
    })
    _emit_socket("alert_broadcast", {"message": msg, "level": level})
    return {"ok": True, "msg": f"Alerte envoyée : {msg}"}

def act_mqtt_info(params=None):
    """Broadcast info system."""
    p = params or {}
    msg = str(p.get("message", "Information"))[:200]
    _publish_mqtt("shos/info/broadcast", {"message": msg, "source": "hardware_controls", "ts": time.time()})
    return {"ok": True, "msg": "Info diffusée"}

def act_mqtt_ping(params=None):
    """Envoie un ping sur le bus pour test de latence."""
    _publish_mqtt("shos/benchmark/ping", {"ts": time.time(), "source": "hardware_controls"})
    return {"ok": True, "msg": "Ping envoyé"}

# ─────────────────────────────────────────────────────────────────────────────
# REGISTRE DES 30 ACTIONS
# ─────────────────────────────────────────────────────────────────────────────

ACTIONS = {
    # ── Catégorie 1 : MODULES (8) ────────────────────────────────────────────
    "launch_voice":         {"label": "🎙 Lancer Voice",         "category": "modules", "fn": act_launch_voice},
    "launch_vision":        {"label": "👁 Lancer Vision",         "category": "modules", "fn": act_launch_vision},
    "launch_ocr":           {"label": "📖 Lancer OCR",            "category": "modules", "fn": act_launch_ocr},
    "launch_streaming":     {"label": "📡 Lancer Streaming",      "category": "modules", "fn": act_launch_streaming},
    "launch_danger":        {"label": "⚠ Lancer Danger Monitor", "category": "modules", "fn": act_launch_danger},
    "launch_hand":          {"label": "✋ Lancer Hand Control",   "category": "modules", "fn": act_launch_hand},
    "launch_joystick":      {"label": "🕹 Lancer Joystick",       "category": "modules", "fn": act_launch_joystick},
    "kill_module":          {"label": "✖ Arrêter module actif",  "category": "modules", "fn": act_kill_module, "params": ["module"]},

    # ── Catégorie 2 : SYSTÈME (7) ────────────────────────────────────────────
    "reboot":               {"label": "🔄 Redémarrer Pi",         "category": "system",  "fn": act_reboot,      "danger": True},
    "shutdown":             {"label": "⏻ Éteindre Pi",            "category": "system",  "fn": act_shutdown,    "danger": True},
    "screenshot":           {"label": "📸 Capture d'écran",       "category": "system",  "fn": act_screenshot},
    "snapshot_video":       {"label": "🎬 Snapshot vidéo",        "category": "system",  "fn": act_snapshot_video},
    "restart_nova":         {"label": "♻ Redémarrer Nova",       "category": "system",  "fn": act_restart_nova},
    "sync_time":            {"label": "⏰ Synchroniser horloge",  "category": "system",  "fn": act_sync_time},
    "mute_system":          {"label": "🔇 Couper le son",         "category": "system",  "fn": act_mute_system},

    # ── Catégorie 3 : RÉGLAGES (8) ────────────────────────────────────────────
    "volume_up":            {"label": "🔊 Volume +",              "category": "settings", "fn": act_volume_up,        "params": ["step"]},
    "volume_down":          {"label": "🔉 Volume -",              "category": "settings", "fn": act_volume_down,      "params": ["step"]},
    "brightness_up":        {"label": "🔆 Luminosité +",          "category": "settings", "fn": act_brightness_up,    "params": ["step"]},
    "brightness_down":      {"label": "🔅 Luminosité -",          "category": "settings", "fn": act_brightness_down,  "params": ["step"]},
    "gas_threshold_up":     {"label": "⬆ Seuil gaz +",           "category": "settings", "fn": act_gas_threshold_up, "params": ["step"]},
    "gas_threshold_down":   {"label": "⬇ Seuil gaz -",           "category": "settings", "fn": act_gas_threshold_down, "params": ["step"]},
    "distance_threshold_up":   {"label": "⬆ Seuil distance +",   "category": "settings", "fn": act_distance_threshold_up, "params": ["step"]},
    "distance_threshold_down": {"label": "⬇ Seuil distance -",   "category": "settings", "fn": act_distance_threshold_down, "params": ["step"]},

    # ── Catégorie 4 : SCRIPTS (3) ────────────────────────────────────────────
    "run_python_file":      {"label": "▶ Exécuter fichier .py",  "category": "scripts", "fn": act_run_python_file,  "params": ["file", "args"]},
    "run_shell_cmd":        {"label": "⚙ Commande shell",        "category": "scripts", "fn": act_run_shell_cmd,    "params": ["cmd"]},
    "run_benchmark":        {"label": "📊 Lancer benchmark",      "category": "scripts", "fn": act_run_benchmark,    "params": ["mode"]},

    # ── Catégorie 5 : MQTT (4) ───────────────────────────────────────────────
    "mqtt_publish":         {"label": "📤 Publier MQTT",          "category": "mqtt",    "fn": act_mqtt_publish,     "params": ["topic", "payload"]},
    "mqtt_alert":           {"label": "🚨 Alerte broadcast",      "category": "mqtt",    "fn": act_mqtt_alert,       "params": ["message", "level"]},
    "mqtt_info":            {"label": "ℹ Info broadcast",         "category": "mqtt",    "fn": act_mqtt_info,        "params": ["message"]},
    "mqtt_ping":            {"label": "📍 Ping bus",              "category": "mqtt",    "fn": act_mqtt_ping},
}

assert len(ACTIONS) == 30, f"Le catalogue doit contenir 30 actions, pas {len(ACTIONS)}"

# ─────────────────────────────────────────────────────────────────────────────
# DISPATCHER
# ─────────────────────────────────────────────────────────────────────────────

def execute_action(action_id: str, params: Optional[dict] = None) -> dict:
    """
    Exécute une action par son ID avec paramètres optionnels.
    Retourne {"ok": bool, "msg": str, ...}
    """
    if action_id not in ACTIONS:
        return {"ok": False, "msg": f"Action inconnue : {action_id}"}
    try:
        result = ACTIONS[action_id]["fn"](params)
        if isinstance(result, dict):
            result["action_id"] = action_id
            result["ts"]        = time.time()
            return result
        return {"ok": True, "action_id": action_id, "result": str(result)}
    except Exception as e:
        logger.error("Action %s: %s", action_id, e)
        return {"ok": False, "msg": str(e), "action_id": action_id}


def get_catalog() -> list:
    """Retourne le catalogue formaté pour l'API."""
    return [
        {
            "id":       aid,
            "label":    a["label"],
            "category": a["category"],
            "danger":   a.get("danger", False),
            "params":   a.get("params", []),
        }
        for aid, a in ACTIONS.items()
    ]