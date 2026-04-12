import cv2
import mediapipe as mp
import time

class PiHandController:
    def __init__(self, socketio):
        self.socketio = socketio
        self.cap = cv2.VideoCapture(0)
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        
        self.last_gesture = "NONE"
        self.gesture_start_time = 0
        self.confirm_threshold = 0.8 # Un peu plus long pour être sûr

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret: return None

        frame = cv2.flip(frame, 1) # Effet miroir pour l'interface
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        current_gesture = "NONE"

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                # Dessiner les points sur la main pour l'interface
                self.mp_draw.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)
                
                lms = hand_lms.landmark
                fingers = []
                
                # Pouce (détection simplifiée)
                fingers.append(lms[4].x < lms[3].x if lms[4].x < lms[17].x else lms[4].x > lms[3].x)
                # Autres doigts
                for tip in [8, 12, 16, 20]:
                    fingers.append(lms[tip].y < lms[tip-2].y)

                # Mapping des Gestes
                if fingers == [True, False, False, False, False]: current_gesture = "TRIGGER_FLORENCE"
                elif fingers == [True, True, True, True, True]:   current_gesture = "START_SYSTEM"
                elif fingers == [False, False, False, False, False]: current_gesture = "STOP_SYSTEM"
                elif fingers == [False, True, False, False, False]:  current_gesture = "READ_TEXT"

        # Confirmation du geste
        if current_gesture != "NONE":
            if current_gesture == self.last_gesture:
                if (time.time() - self.gesture_start_time) > self.confirm_threshold:
                    self.socketio.emit('gesture_detected', {'gesture': current_gesture})
            else:
                self.last_gesture = current_gesture
                self.gesture_start_time = time.time()
        
        return frame
