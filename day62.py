import torch

print("--- Step 1: Generating High-Precision Weights (FP32) ---")
# Simulating a small weight matrix of an LLM Layer
torch.manual_seed(42)
original_weights = torch.randn(1000, 1000) * 5.0 # FP32 Weights

fp32_memory = original_weights.element_size() * original_weights.nelement()
print(f"Original FP32 Matrix Shape: {original_weights.shape}")
print(f"Memory Occupied in RAM/VRAM: {fp32_memory / 1024:.2f} KB\n")

print("--- Step 2: Implementing Symmetric INT8 Quantization ---")
# 1. Find the maximum absolute value to determine the scale
max_val = torch.max(torch.abs(original_weights))

# 2. Calculate scale factor for signed 8-bit integer (Range: -128 to 127)
scale = max_val / 127.0

# 3. Quantize: Divide by scale, round, and clip into INT8 boundaries
quantized_weights = torch.round(original_weights / scale).to(torch.int8)

int8_memory = quantized_weights.element_size() * quantized_weights.nelement()
print(f"Quantized INT8 Matrix Memory: {int8_memory / 1024:.2f} KB")
print(f"📉 VRAM Saving Factor: {((fp32_memory - int8_memory) / fp32_memory) * 100:.2f}% Less Space!")

print("\n--- Step 3: Dequantization & Loss Audit (Inference Simulation) ---")
# During inference, the matrix is uncompressed back to floats for calculation
dequantized_weights = quantized_weights.to(torch.float32) * scale

# Calculate Mean Squared Error (MSE) to check information loss
quantization_error = torch.mean((original_weights - dequantized_weights) ** 2)
print(f"⏱️ Reconstruction Mean Squared Error (MSE): {quantization_error.item():.6f}")
print("💡 Notice how the error is extremely close to zero, proving we saved 75% memory with negligible loss!")