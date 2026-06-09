import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt

print("--- Step 1: Loading the Generative AI Brain (Stable Diffusion) ---")
# Loading the model. We use float16 to heavily save RAM/VRAM on your RTX 3050
model_id = "runwayml/stable-diffusion-v1-5"

print("Downloading/Loading weights... (This might take a minute)")
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

# Move the computation to your Nvidia GPU
pipe = pipe.to("cuda")

# ⚡ VRAM Optimization for 4GB GPUs (Crucial for RTX 3050)
pipe.enable_attention_slicing()

print("✅ Generative AI Model Loaded and Optimized!")

print("\n--- Step 2: Giving the AI an Imagination Prompt ---")
# You can change this text to whatever you want the AI to draw!
prompt = "A futuristic cyberpunk city in Tamil Nadu, flying cars, neon lights, rainy night, highly detailed 4k ultra realistic"

print(f"🎨 User Imagination Prompt: '{prompt}'")

print("\n--- Step 3: Generating the Image from Thin Air (Diffusion Process) ---")
print("Diffusion process started. Removing noise step-by-step... ⏳")

# The AI generates the image based on the text prompt
image = pipe(prompt).images[0]

print("✅ Masterpiece successfully generated!")

print("\n--- Step 4: Saving and Displaying the Output ---")
# Display the generated image in a window
plt.imshow(image)
plt.axis("off")
plt.title("Day 47: AI Generated Art")
plt.show()

# Save it to your ML_Advanced folder
image.save("ai_cyberpunk_city.png")
print("💾 Image saved as 'ai_cyberpunk_city.png' in your project folder.")