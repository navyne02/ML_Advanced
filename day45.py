import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import warnings
warnings.filterwarnings("ignore")

print("--- Step 1: Loading Multimodal Image Captioning Brain (BLIP) ---")
# Loading the processor (for image/text prep) and the model itself
model_id = "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(model_id)
model = BlipForConditionalGeneration.from_pretrained(model_id)

print("✅ BLIP Model Loaded Successfully!")

print("\n--- Step 2: Fetching an Image ---")
# Let's fetch a high-quality image of a gaming/coding setup
image_url = "https://images.unsplash.com/photo-1593640408182-31c70c8268f5"
image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')

print(f"Image fetched! Format: RGB, Dimensions: {image.size}")

print("\n--- Step 3: AI Analyzing Image and Writing Text ---")
# 1. Unconditional Captioning (AI writes whatever it sees without hints)
inputs = processor(image, return_tensors="pt")
out = model.generate(**inputs)
unconditional_caption = processor.decode(out[0], skip_special_tokens=True)

print(f"\n🧠 AI Generated Caption: '{unconditional_caption.capitalize()}'")

print("\n--- Step 4: Conditional Image Captioning ---")
# 2. Conditional Captioning (We give the AI a starting phrase to guide it)
starting_text = "a computer"
inputs_cond = processor(image, text=starting_text, return_tensors="pt")
out_cond = model.generate(**inputs_cond)
conditional_caption = processor.decode(out_cond[0], skip_special_tokens=True)

print(f"🎯 AI Guided Caption (Starting with '{starting_text}'): '{conditional_caption.capitalize()}'")