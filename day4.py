import cv2
import mediapipe as mp

# 1. MediaPipe Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils # For drawing lines on hand

# 2. Access Webcam
cap = cv2.VideoCapture(0)

print("Webcam Starting... Show your hand! ✋ Press 'q' to stop.")

while cap.isOpened():
    success, img = cap.read()
    if not success: break

    # Convert to RGB (MediaPipe requirement)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # 3. If hand is detected, draw landmarks
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            # Drawing the 21 points and connections
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
            
            # Optional: Print coordinates of the tip of the thumb (Point 4)
            # for id, lm in enumerate(hand_lms.landmark):
            #     print(id, lm.x, lm.y)

    cv2.imshow("Day 4: AI Hand Tracker", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()