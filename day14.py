import cv2
import numpy as np

# 1. Load Pre-trained Model (SSD with MobileNet)
# Intha files OpenCV library-kullaye irukkum or download link use pannalam
# For simplicity, we are using a common configuration
config_path = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weights_path = 'frozen_inference_graph.pb'

# Note: Intha model 80 types of objects-ah detect pannum (COCO Dataset labels)
classNames = []
classFile = 'coco.names' # List of 80 names like person, bicycle, etc.
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

net = cv2.dnn_DetectionModel(weights_path, config_path)
net.setInputSize(320, 320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

cap = cv2.VideoCapture(0)
print("Webcam Starting... Show some objects (Phone, Bottle, Mug)! 📱 Press 'q' to stop.")

while True:
    success, img = cap.read()
    # 2. Detect Objects
    classIds, confs, bbox = net.detect(img, confThreshold=0.5)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            # 3. Draw Bounding Box and Label
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            cv2.putText(img, classNames[classId-1].upper(), (box[0]+10, box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, str(round(confidence*100, 2)), (box[0]+10, box[1]+60),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Day 14: AI Object Detector", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()