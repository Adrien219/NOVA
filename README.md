```markdown
# NOVA – Plateforme d'assistance embarquée modulaire

Application mobile et système embarqué pour l'assistance contextuelle aux personnes malvoyantes et la télé-maintenance industrielle. Nova fonctionne 100 % hors ligne sur Raspberry Pi, avec orchestration dynamique des modules via MQTT.

---

## Fonctionnalités principales

- **Architecture 4 couches** : Physique (Arduino), Middleware (MQTT), Edge AI (Raspberry Pi), Application (Flask)
- **Modules logiciels indépendants** chargés dynamiquement sous forme de greffons (plugins)
- **Détection d'obstacles temps réel** : capteurs ultrasoniques HC-SR04 + retour haptique PWM
- **Commande gestuelle** : MediaPipe Hands (21 repères 3D) avec confirmation temporelle (0,8 s)
- **Reconnaissance vocale hors ligne** : Vosk (modèle fr-0.22, WER 14,3 %)
- **Synthèse vocale** : eSpeak-NG (TTS hors ligne, latence 80–120 ms)
- **Lecture de texte** : Tesseract OCR (87 % de reconnaissance sur 150 images)
- **Orchestrateur IA** : 3 modèles embarqués pour l'activation des modules, l'optimisation énergétique et la suggestion de profils
- **Interface de supervision** : Dashboard web S.H.O.S (Flask + Socket.IO) en réseau local
- **Flux vidéo déporté** : ESP32-CAM (MJPEG Wi-Fi)

---

## Architecture du système

```
┌─────────────────────────────────────────────────────────────┐
│  APPLICATION   : Flask + Socket.IO, profils, supervision   │
├─────────────────────────────────────────────────────────────┤
│  EDGE AI       : nova.py, MediaPipe, Vosk, Tesseract,      │
│                  YOLOv8n, orchestrateur IA                  │
├─────────────────────────────────────────────────────────────┤
│  MIDDLEWARE    : MQTT Mosquitto (port 1883), UART 9600     │
├─────────────────────────────────────────────────────────────┤
│  PHYSIQUE      : Arduino Uno (capteurs + PWM)              │
│                  ESP32-CAM (MJPEG Wi-Fi)                    │
│                  PiCamera2                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Composants matériels

| Composant | Rôle | Spécifications |
|-----------|------|----------------|
| **Raspberry Pi 4B** | Orchestrateur, IA, serveur web | ARM Cortex-A72 1,8 GHz, 4 Go RAM |
| **Arduino Uno Rev3** | Acquisition capteurs, PWM haptique | ATmega328P 16 MHz, 2 Ko SRAM |
| **ESP32-CAM** | Flux vidéo déporté | OV2640, 2 MP, Wi-Fi |
| **HC-SR04 (×2)** | Détection d'obstacles ultrasonique | 2–400 cm, 15° |
| **DHT11** | Température, humidité | ±2 °C, ±5 % RH |
| **MQ-3** | Détection de gaz (alcool) | 0,05–10 mg/L |

---

## Installation

### 1. Raspberry Pi

```bash
# Mise à jour du système
sudo apt update && sudo apt upgrade -y

# Broker MQTT Mosquitto
sudo apt install -y mosquitto mosquitto-clients
sudo systemctl enable mosquitto
sudo systemctl start mosquitto

# Dépendances Python
sudo apt install -y python3-pip python3-venv git
sudo apt install -y tesseract-ocr tesseract-ocr-fra
sudo apt install -y espeak-ng libespeak-ng-dev
sudo apt install -y libatlas-base-dev libhdf5-dev

# Bibliothèques Python
pip install flask flask-socketio paho-mqtt psutil
pip install ultralytics onnxruntime
pip install opencv-python-headless pytesseract
pip install vosk picamera2
```

### 2. Cloner le projet

```bash
git clone https://github.com/Adrien219/NOVA.git
cd NOVA
```

### 3. Configuration du broker MQTT

```bash
mosquitto -d -p 1883
```

### 4. Lancer le système

```bash
python3 nova.py
```

### 5. Accéder à l'interface web

```
http://<ip-du-raspberry-pi>:5000
```

---

## Démarrage du benchmark

Le benchmark SHOS permet de mesurer les performances du système (latence MQTT, FPS, CPU, RAM, température).

```bash
python3 benchmark.py --mode full --duration 60 --output chapitre4
```

### Paramètres disponibles

| Option | Description | Défaut |
|--------|-------------|--------|
| `--mode` | `full` ou `system` | `full` |
| `--duration` | Durée du test (secondes) | 60 |
| `--output` | Préfixe des fichiers de sortie | `benchmark` |
| `--mqtt-msgs` | Nombre de messages MQTT | 100 |
| `--topic` | Topic MQTT à observer | `shos/camera/vision` |

### Résultats

Les résultats sont exportés dans le dossier `results/` au format CSV :
- `*_fps.csv` : données de débit vision
- `*_mqtt.csv` : données de latence MQTT
- `*_system.csv` : données de charge système
- `*_summary.txt` : rapport consolidé

---

## Modules plugins

Nova charge dynamiquement les modules depuis `plugins/modules/`. Chaque module est un dossier autonome contenant :

```
plugins/modules/
├── navigation/
│   ├── config.json
│   ├── main.py
│   └── interface.html
├── vision_objet/
│   ├── config.json
│   ├── main.py
│   └── interface.html
├── assistant_ia/
│   ├── config.json
│   ├── main.py
│   └── interface.html
└── hand_control/
    ├── config.json
    ├── main.py
    └── interface.html
```

### Structure d'un module

```json
{
  "name": "Navigation",
  "description": "Détection d'obstacles ultrasoniques",
  "topics_subscribe": ["nova/sensor/ultrasonic"],
  "topics_publish": ["nova/nav/alert"],
  "template": "navigation/interface.html",
  "autostart": true
}
```

---

## Entraînement de l'orchestrateur IA

L'entraînement des 3 modèles de l'orchestrateur est réalisé dans un notebook Google Colab.

1. Ouvrir le notebook : `shos_orchestrator_training.ipynb`
2. Importer les 3 fichiers CSV : `modules.csv`, `energy.csv`, `profiles.csv`
3. Exécuter les cellules pour entraîner les modèles
4. Exporter les fichiers générés (dossier `shos_models/`)

### Modèles générés

| Modèle | Algorithme | Accuracy | Fichier |
|--------|------------|----------|---------|
| Module Manager | Decision Tree (multi-label) | 98,3 % | `module_manager.pkl` |
| Energy Optimizer (CPU) | MLP 32-16-8 | 94,0 % | `energy_cpu_freq.tflite` |
| Energy Optimizer (FPS) | MLP 32-16-8 | 90,7 % | `energy_fps_target.tflite` |
| Energy Optimizer (Sleep) | MLP 32-16-8 | 98,7 % | `energy_sleep_modules.tflite` |
| Profile Predictor | k-NN (k=5) | 96,0 % | `profile_predictor.pkl` |

### Déploiement sur le Raspberry Pi

```bash
scp shos_models/*.tflite pi@<ip>:/home/pi/nova/models/
scp shos_models/*.pkl pi@<ip>:/home/pi/nova/models/
```

---

## Structure du projet

```
nova/
├── nova.py                # Application Flask + orchestrateur
├── vision_service.py      # Service Picamera2 + MediaPipe
├── utils.py               # ConfigManager (profils JSON)
├── benchmark.py           # Script de benchmark SHOS
├── config.json            # Configuration et profils utilisateurs
├── requirements.txt       # Dépendances Python
├── templates/             # Interfaces web (Jinja2)
│   ├── dashboard.html
│   ├── diagnostic.html
│   ├── profile_manager.html
│   ├── mobile.html
│   └── hud.html
├── plugins/modules/       # Modules fonctionnels indépendants
│   ├── navigation/        # Ultrasons + haptique
│   ├── vision_objet/      # YOLOv8n + OCR
│   ├── assistant_ia/      # STT Vosk + TTS eSpeak
│   └── hand_control/      # Contrôle gestuel MediaPipe
├── models/                # Modèles IA (.onnx, .tflite)
├── results/               # Résultats de benchmark
└── Images/                # Images pour la documentation
```

---

## Formats des messages MQTT

### Alerte obstacle (`nova/nav/alert`)

```json
{
  "side": "left",
  "distance": 85.3,
  "level": "HIGH",
  "vibration_duty": 100,
  "tts_message": "Danger, obstacle très proche",
  "timestamp": "2026-07-14T09:23:11.456"
}
```

### Détection YOLO (`nova/vision/detection`)

```json
{
  "objects": [
    {"class": "person", "confidence": 0.87, "bbox": [120, 80, 350, 460]},
    {"class": "chair", "confidence": 0.72, "bbox": [400, 200, 580, 450]}
  ],
  "count": 2,
  "inference_time_ms": 51.2,
  "frame_id": 1482
}
```

### Heartbeat module (`nova/system/status`)

```json
{
  "module": "vision",
  "status": "running",
  "uptime_s": 3621,
  "fps": 18.4,
  "cpu_pct": 68.2
}
```

---

## Résultats expérimentaux

| Métrique | Valeur |
|----------|--------|
| Latence MQTT médiane | **1,82 ms** (σ = 0,3 ms, IC95 ±0,2 ms) |
| Perte de paquets MQTT | **0 %** (300 messages) |
| Débit vision | **2,0 FPS** |
| CPU moyen | **93,7 %** |
| RAM moyenne | **27,3 %** |
| Température max SoC | **58,4 °C** |
| Throttling | **Aucun** (`vcgencmd get_throttled` → 0x0) |
| Autonomie | **7 h 30** (sur batterie 4 400 mAh) |
| WER (Vosk) | **14,3 %** (10 locuteurs) |
| OCR (Tesseract) | **87 %** (150 images) |

---

## Points d'attention

### Problèmes de connexion MQTT
- Vérifier que le broker Mosquitto est en cours d'exécution :
  ```bash
  systemctl status mosquitto
  ```
- Vérifier le port (1883) et l'adresse (localhost)

### Problèmes de caméra (PiCamera2)
- Vérifier que la caméra est activée :
  ```bash
  sudo raspi-config → Interface Options → Camera → Enable
  ```
- Vérifier les permissions : `sudo usermod -a -G video pi`

### Problèmes de communication UART (Arduino ↔ Pi)
- Vérifier les permissions sur `/dev/ttyUSB0` ou `/dev/ttyACM0`
  ```bash
  sudo chmod 666 /dev/ttyUSB0
  ```
- Utiliser un convertisseur de niveau logique 3,3V ↔ 5V

### Latence E2E non instrumentée
- La latence bout-en-bout de la chaîne de navigation n'a pas encore été mesurée.
- Seul le transport MQTT (1,82 ms) est instrumenté.
- Priorité des travaux futurs.

---

## Performances de l'orchestrateur IA

| Modèle | Accuracy | F1-score | Temps d'inférence |
|--------|----------|----------|-------------------|
| Module Manager | **98,3 %** | **99,6 %** | 0,031 ms |
| Energy Optimizer (CPU) | **94,0 %** | **93,1 %** | 0,170 ms |
| Energy Optimizer (FPS) | **90,7 %** | **89,5 %** | 0,178 ms |
| Energy Optimizer (Sleep) | **98,7 %** | **98,5 %** | 0,389 ms |
| Profile Predictor | **96,0 %** | **96,0 %** | 0,027 ms |

---

## Vérification

```bash
npx expo export --platform web
```

---

## Références

- [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
- [Vosk Offline Speech Recognition](https://alphacephei.com/vosk/)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [Eclipse Mosquitto](https://mosquitto.org/)
- [YOLOv8](https://github.com/ultralytics/ultralytics)
```