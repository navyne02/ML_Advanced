import torch
import torch.nn as nn
from torchvision import models

print("--- Step 1: Loading a Pre-trained AI Brain ---")
# Loading a lightweight image classification model
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model.eval() # Set to evaluation mode

print("--- Step 2: Simulating an Input Image ---")
# We create a dummy image (1 batch, 3 colors, 224x224 pixels)
# In real life, this would be cv2.imread('dog.jpg')
image = torch.rand(1, 3, 224, 224)

# ⚠️ CRITICAL STEP FOR HACKING: We tell PyTorch to track gradients for the IMAGE, not just the model!
image.requires_grad = True

print("\n--- Step 3: Making the Initial Prediction (Before Attack) ---")
output = model(image)
initial_prediction = output.max(1, keepdim=True)[1].item()
print(f"✅ AI Initial Prediction (Class ID): {initial_prediction}")

print("\n--- Step 4: Crafting the FGSM Adversarial Attack 🦹‍♂️ ---")
# Let's say we want to trick the AI. We calculate the loss against the current prediction.
loss_function = nn.CrossEntropyLoss()
target_label = torch.tensor([initial_prediction])

# Forward pass & Loss calculation
loss = loss_function(output, target_label)
model.zero_grad()
loss.backward() # Calculates gradients

# Extract the gradients of the image
image_gradient = image.grad.data

# FGSM Formula: Perturbed Image = Original Image + Epsilon * Sign(Gradient)
# Epsilon is the "Attack Strength" (Keep it small so humans can't see the noise)
epsilon = 0.1
attack_noise = epsilon * image_gradient.sign()
hacked_image = image + attack_noise

# Ensure pixels stay within valid image range [0, 1]
hacked_image = torch.clamp(hacked_image, 0, 1)

print("\n--- Step 5: Testing the Hacked Image (After Attack) ---")
# Feed the poisoned image back into the AI
hacked_output = model(hacked_image)
new_prediction = hacked_output.max(1, keepdim=True)[1].item()

print(f"❌ AI Prediction on Hacked Image (Class ID): {new_prediction}")

if initial_prediction != new_prediction:
    print("\n🚨 ATTACK SUCCESSFUL! The AI was completely fooled by invisible noise!")
else:
    print("\n🛡️ Attack Failed. The AI was robust enough to handle the noise.")