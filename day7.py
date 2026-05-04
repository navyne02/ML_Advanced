import cv2
import numpy as np
import mediapipe as mp

# 1. Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, detection_confidence=0.85)
mp_draw = mp.solutions.drawing_utils

# Canvas setup (Namma drawing vizhura theredu)
canvas = np.zeros((480, 640, 3), np.uint8)
draw_color = (255, 0, 255) # Purple starting color
px, py = 0, 0 # Previous coordinates

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()
    img = cv2.flip(img, 1) # Mirror effect
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Drawing UI (Color boxes)
    cv2.rectangle(img, (10, 10), (150, 80), (255, 0, 255), cv2.FILLED) # Purple
    cv2.rectangle(img, (160, 10), (300, 80), (0, 255, 0), cv2.FILLED)   # Green
    cv2.rectangle(img, (310, 10), (450, 80), (0, 0, 0), cv2.FILLED)     # Eraser
    cv2.putText(img, "ERASE", (340, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_lms.landmark):
                h, w, c = img.shape
                lm_list.append([int(lm.x * w), int(lm.y * h)])

            # Tip of Index (8) and Middle (12)
            x1, y1 = lm_list[8]
            x2, y2 = lm_list[12]

            # 2. Check which fingers are up
            fingers_up = [1 if lm_list[8][1] < lm_list[6][1] else 0] # Simple logic

            # 3. Selection Mode (Two fingers up)
            if lm_list[8][1] < lm_list[6][1] and lm_list[12][1] < lm_list[10][1]:
                px, py = 0, 0 # Reset drawing
                if y1 < 80:
                    if 10 < x1 < 150: draw_color = (255, 0, 255)
                    elif 160 < x1 < 300: draw_color = (0, 255, 0)
                    elif 310 < x1 < 450: draw_color = (0, 0, 0)
                cv2.rectangle(img, (x1, y1-25), (x2, y2+25), draw_color, cv2.FILLED)

            # 4. Drawing Mode (Only Index finger up)
            elif lm_list[8][1] < lm_list[6][1]:
                cv2.circle(img, (x1, y1), 10, draw_color, cv2.FILLED)
                if px == 0 and py == 0: px, py = x1, y1
                
                if draw_color == (0, 0, 0): # Eraser thickness
                    cv2.line(canvas, (px, py), (x1, y1), draw_color, 50)
                else:
                    cv2.line(canvas, (px, py), (x1, y1), draw_color, 10)
                px, py = x1, y1

    # Merge Canvas and Webcam Feed
    img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, img_inv)
    img = cv2.bitwise_or(img, canvas)

    cv2.imshow("Day 7: Virtual Paint", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()