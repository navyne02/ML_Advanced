import requests
from PIL import Image
from transformers import ViltProcessor, ViltForQuestionAnswering
import warnings
warnings.filterwarnings("ignore")

print("--- Step 1: Loading Multimodal VQA Brain (ViLT) ---")
# Loading the processor and the pre-trained VQA model
model_id = "dandelin/vilt-b32-finetuned-vqa"
processor = ViltProcessor.from_pretrained(model_id)
model = ViltForQuestionAnswering.from_pretrained(model_id)

print("✅ ViLT Model Loaded Successfully!")

print("\n--- Step 2: Fetching an Image ---")
# Let's fetch an image of some people playing soccer
image_url = "https://images.unsplash.com/photo-1518605368461-1e1e38ce8058"
image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')

print(f"Image fetched! Format: RGB, Dimensions: {image.size}")

print("\n--- Step 3: Asking Questions to the Image ---")
# You can change these questions to test the AI's intelligence!
questions = [
    "What sport is being played?",
    "How many people are in the image?",
    "Is it daytime or nighttime?"
]

for question in questions:
    print(f"\n🗣️ User Asks: '{question}'")
    
    # Prepare the image and the specific question for the AI
    encoding = processor(image, question, return_tensors="pt")

    # Forward pass through the neural network
    outputs = model(**encoding)
    logits = outputs.logits
    
    # Find the answer with the highest probability score
    best_answer_index = logits.argmax(-1).item()
    ai_answer = model.config.id2label[best_answer_index]
    
    print(f"🤖 AI Answers: '{ai_answer.capitalize()}'")

print("\n🎯 Notice how the AI processes the visual context differently for each specific text prompt!")