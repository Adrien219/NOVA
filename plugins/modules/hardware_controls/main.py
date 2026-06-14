#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S.H.O.S - HARDWARE CONTROLS - Point d'entree module
Routes Flask : /execute /config /ir /catalog /status
Initialise ActionContext (MQTT + SocketIO) pour actions.py
"""

import json
import time
import logging
from pathlib import Path

try:
    from flask import Blueprint, request, jsonify
except ImportError:
    Blueprint = None

from .actions import execute_action, get_catalog, ctx as action_ctx

logger = logging.getLogger("SHOS.HardwareControls")

MODULE_DIR    = Path(__file__).resolve().parent
BINDINGS_FILE = MODULE_DIR / "bindings.json"

DEFAULT_BINDINGS = {
    "joy":     {"N": None, "S": None, "E": None, "W": None, "C": None},
    "buttons": [None] * 16,
    "pot":     "volume_up",
    "mic":     "launch_voice",
    "ir":      "snapshot_video",
}

def load_bindings():
    try:
        if BINDINGS_FILE.exists():
            data = json.loads(BINDINGS_FILE.read_text(encoding="utf-8"))
            merged = dict(DEFAULT_BINDINGS)
            merged.update(data)
            return merged
    except Exception as e:
        logger.warning("Chargement bindings : %s", e)
    return dict(DEFAULT_BINDINGS)

def save_bindings(data):
    try:
        MODULE_DIR.mkdir(parents=True, exist_ok=True)
        BINDINGS_FILE.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return True
    except Exception as e:
        logger.error("Sauvegarde bindings : %s", e)
        return False

def create_blueprint(socketio=None, mqtt_client=None, project_root=None):
    action_ctx.mqtt_client  = mqtt_client
    action_ctx.socketio     = socketio
    action_ctx.project_root = project_root
    action_ctx.state        = {
        "volume": 50, "brightness": 70,
        "gas_threshold": 400, "distance_threshold": 50,
    }

    bp = Blueprint("hardware_controls", __name__, url_prefix="/module/hardware_controls")

    @bp.route("/api/catalog", methods=["GET"])
    def api_catalog():
        return jsonify({"ok": True, "actions": get_catalog()})

    @bp.route("/api/execute", methods=["POST"])
    def api_execute():
        body      = request.get_json(silent=True) or {}
        action_id = str(body.get("action_id", "")).strip()
        params    = body.get("params") or {}
        source    = body.get("source", "virt")
        device    = body.get("device", "")
        if not action_id:
            return jsonify({"ok": False, "msg": "action_id requis"}), 400
        logger.info("Execute [%s] %s (via %s)", source, action_id, device)
        result = execute_action(action_id, params)
        if socketio:
            try:
                socketio.emit("hw_action_result", {
                    "action_id": action_id, "source": source,
                    "device": device, "result": result, "ts": time.time(),
                })
            except Exception:
                pass
        return jsonify(result)

    @bp.route("/api/config", methods=["GET"])
    def api_config_get():
        return jsonify(load_bindings())

    @bp.route("/api/config", methods=["POST"])
    def api_config_post():
        body = request.get_json(silent=True) or {}
        if not body:
            return jsonify({"ok": False, "msg": "Body JSON requis"}), 400
        ok = save_bindings(body)
        if ok:
            logger.info("Bindings sauvegardes")
            if socketio:
                try:
                    socketio.emit("hardware_config_updated", body)
                except Exception:
                    pass
            return jsonify({"ok": True, "msg": "Configuration sauvegardee"})
        return jsonify({"ok": False, "msg": "Erreur sauvegarde"}), 500

    @bp.route("/api/ir", methods=["POST"])
    def api_ir():
        body  = request.get_json(silent=True) or {}
        state = str(body.get("state", "off")).lower()
        source= body.get("source", "virt")
        if action_ctx.mqtt_client:
            try:
                action_ctx.mqtt_client.publish(
                    "shos/arduino/gpio",
                    json.dumps({"pin": "IR_LED", "state": state, "source": source}),
                )
            except Exception as e:
                logger.warning("IR MQTT : %s", e)
        logger.info("[IR] %s (source: %s)", state, source)
        return jsonify({"ok": True, "msg": f"IR LED {state.upper()}", "state": state})

    @bp.route("/api/status", methods=["GET"])
    def api_status():
        return jsonify({
            "ok": True, "module": "hardware_controls",
            "state": action_ctx.state,
            "mqtt": action_ctx.mqtt_client is not None,
            "actions": len(get_catalog()), "ts": time.time(),
        })

    logger.info("Blueprint hardware_controls enregistre")
    return bp

def setup(app, socketio=None, mqtt_client=None, project_root=None):
    """Appele par nova.py pour enregistrer le module."""
    bp = create_blueprint(socketio=socketio, mqtt_client=mqtt_client, project_root=project_root)
    app.register_blueprint(bp)
    return bp

if __name__ == "__main__":
    print("Lancer nova.py — ce module est un Blueprint Flask")
    print("30 actions disponibles:")
    for a in get_catalog():
        print(f"  [{a['category']:10s}] {a['id']:35s} {a['label']}")