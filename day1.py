import cv2

# 1. Load the Face Detection Brain (Haar Cascade)...
face_cascade = cv2.CascadeClassifier('face.xml')

# 2. Access the Webcam (0 is default camera)
cap = cv2.VideoCapture(0)

print("Webcam Starting... Press 'q' to stop!")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # AI works better in Black & White (Gray)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 3. Detect Faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # 4. Draw Rectangle around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, "Face Detected", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the result
    cv2.imshow('Day 1: AI Face Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera
cap.release()
cv2.destroyAllWindows()
