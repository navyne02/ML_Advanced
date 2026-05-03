import cv2
import mediapipe as mp

# 1. MediaPipe Pose Setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# 2. Access Webcam
cap = cv2.VideoCapture(0)

print("Webcam Starting... Stand back so AI can see your full body! 🤸‍♂️ Press 'q' to stop.")

while cap.isOpened():
    success, img = cap.read()
    if not success: break

    # Convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    # 3. Draw Pose Landmarks
    if results.pose_landmarks:
        # Draw the connections (Skeletal structure)
        mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Logic insight: You can get specific points like this:
        # landmarks = results.pose_landmarks.landmark
        # left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        # print(left_shoulder.x, left_shoulder.y)

    cv2.imshow("Day 5: AI Pose Tracker", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()