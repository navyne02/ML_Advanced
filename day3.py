from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt

# 1. Unga photo path (sirikira photo)
image_path = "smile.jpg"

print("AI is looking at your face... 👀")

# 2. THE MAGIC LINE: AI-kitta "emotion" mattum kandupidi nu solrom
# enforce_detection=False pota, photo konjam blur-ah irunthalum error varaathu.
result = DeepFace.analyze(img_path=image_path, actions=['emotion'], enforce_detection=False)

# 3. Main emotion-ah veliya edukkurom
dominant_emotion = result[0]['dominant_emotion']

print(f"\n--- 🎭 AI Emotion Result ---")
print(f"AI Says you are feeling: {dominant_emotion.upper()}!")
print("----------------------------\n")

# 4. Photo-va open panni paakalam
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # AI-ku color puriya mathurom

plt.imshow(img)
plt.title(f"AI Detected Mood: {dominant_emotion.upper()}")
plt.axis('off') # Side-la irukkura numbers (axis) thevai illai
plt.show()