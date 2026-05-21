import cv2
import mediapipe as mp


mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils 


cap = cv2.VideoCapture(0)

print("Webcam Starting... Show your hand! ✋ Press 'q' to stop.")

while cap.isOpened():
    success, img = cap.read()
    if not success: break

    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
    
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
            
    
    cv2.imshow("Day 4: AI Hand Tracker", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()