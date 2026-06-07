import torch
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

print("--- Step 1: Loading Multimodal AI Brain (CLIP) ---")
# Loading OpenAI's CLIP model and processor from Hugging Face
model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)

print("✅ CLIP Model Loaded Successfully!")

print("\n--- Step 2: Fetching an Image ---")
# Let's download a sample image of a cute dog playing in the grass
image_url = "https://images.unsplash.com/photo-1537151608804-ea6f11840eb3"
image = Image.open(requests.get(image_url, stream=True).raw)

print(f"Image fetched! Dimensions: {image.size}")

print("\n--- Step 3: Defining Text Labels (Zero-Shot) ---")
# We don't train the model on these! We just give it options to choose from.
text_labels = [
    "a photo of a cat",
    "a photo of a dog playing in the grass",
    "a picture of a fast sports car",
    "a beautiful mountain landscape"
]

print(f"Options provided to AI: {text_labels}")

print("\n--- Step 4: AI Cross-Matching (Text vs Image) ---")
# The processor prepares both the image and the text for the neural network
inputs = processor(text=text_labels, images=image, return_tensors="pt", padding=True)

# Forward pass: the model calculates the similarity score between the image and each text
with torch.no_grad():
    outputs = model(**inputs)

# The logits_per_image represents the similarity score
logits_per_image = outputs.logits_per_image 

# Convert the scores into probabilities (percentages)
probs = logits_per_image.softmax(dim=1).numpy()[0]

print("\n--- Final Multimodal Prediction ---")
for label, prob in zip(text_labels, probs):
    print(f"Label: '{label}' -> Probability: {prob * 100:.2f}%")

best_match_idx = probs.argmax()
print(f"\n🏆 AI concludes this image is: '{text_labels[best_match_idx]}'")