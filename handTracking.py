import mediapipe as mp 
import cv2

class HandGestureControl:
    def __init__(self, detection_confidence=0.7, tracking_confidence=0.7):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.drawing_utils = mp.solutions.drawing_utils
        self.gesture_state = False
        self.toggle_cooldown = 30
        self.cooldown_timer = 0  # ← use this consistently

    def detect_toggle_gesture(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(frame_rgb)

        if result.multi_hand_landmarks and self.cooldown_timer == 0:
            hand_landmarks = result.multi_hand_landmarks[0]

            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]

            distance = ((thumb_tip.y - index_tip.y) ** 2 + (thumb_tip.x - index_tip.x) ** 2) ** 0.5

            if distance < 0.05:
                self.gesture_state = not self.gesture_state
                self.cooldown_timer = self.toggle_cooldown  # ← fixed name

        # Show gesture state
        cv2.putText(frame, f"Gesture State: {self.gesture_state}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Show cooldown
        if self.cooldown_timer > 0:
            cv2.putText(frame, f"Cooldown: {self.cooldown_timer}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            self.cooldown_timer -= 1

        return self.gesture_state


# Test the hand gesture toggle
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    hgc = HandGestureControl()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        hgc.detect_toggle_gesture(frame)

        cv2.imshow("Hand Gesture Toggle Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
