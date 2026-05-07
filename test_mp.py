import cv2
import mediapipe as mp
import numpy as np

# Initialisation MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Test de plusieurs index de caméra
cap = cv2.VideoCapture(0)

print("🚀 Test MediaPipe stabilisé...")

try:
    while True:
        success, frame = cap.read()
        
        # SI LA LECTURE ÉCHOUE OU L'IMAGE EST VIDE
        if not success or frame is None or frame.size == 0:
            continue

        try:
            # Conversion RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                print("✋ MAIN DÉTECTÉE !")
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.imshow('NOVA DEBUG', frame)
        except cv2.error as e:
            # Ignore les erreurs de redimensionnement/remap d'OpenCV
            continue

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    pass
finally:
    cap.release()
    cv2.destroyAllWindows()
