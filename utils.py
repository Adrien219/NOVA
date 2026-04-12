import json
import logging
from pathlib import Path

logger = logging.getLogger("SHOS_UTILS")

class ConfigManager:
    def __init__(self, config_file="config/config.json"):
        self.config_path = Path(config_file)
        # Crée le dossier 'config' s'il n'existe pas
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        # Crée un fichier de configuration de base si inexistant
        if not self.config_path.exists():
            self.save_config({"profiles": {}, "current_active_profile": ""})

    def load_config(self):
        """Charge les profils depuis le fichier JSON"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {"profiles": {}, "current_active_profile": ""}
        except Exception as e:
            logger.error(f"Erreur chargement config: {e}")
            return {"profiles": {}, "current_active_profile": ""}

    def save_config(self, config_data):
        """Sauvegarde les profils dans le fichier JSON"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=4)
            return True
        except Exception as e:
            logger.error(f"Erreur sauvegarde config: {e}")
            return False

    def activate_profile(self, profile_id):
        """Change le profil actif dans la configuration"""
        config = self.load_config()
        if profile_id in config.get("profiles", {}):
            config["current_active_profile"] = profile_id
            return self.save_config(config)
        return False