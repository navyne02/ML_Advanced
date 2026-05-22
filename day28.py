import numpy as np
import sys

print("--- Step 1: Generating Deep Learning Layer Weights (FP32) ---")
# Simulating a small weight matrix of a neural network layer (e.g., 1000 weight connections)
np.random.seed(42)
fp32_weights = np.random.uniform(-1.5, 1.5, 1000).astype(np.float32)

# Calculate original memory usage
fp32_memory = fp32_weights.nbytes
print(f"Original Weights Sample : {fp32_weights[:3]}")
print(f"Original RAM Footprint  : {fp32_memory} Bytes (32-bit floating point)")

print("\n--- Step 2: Executing Symmetric 8-Bit Quantization Formula ---")
# Core Math Logic: Quantized_value = round(Scale * Real_value)
# Where Scale = (2^(b-1) - 1) / Max_Absolute_Value
max_val = np.max(np.abs(fp32_weights))
quant_scale = 127 / max_val

# Transform to INT8 integers
int8_weights = np.round(fp32_weights * quant_scale).astype(np.int8)
int8_memory = int8_weights.nbytes

print(f"Quantized Weights Sample: {int8_weights[:3]}")
print(f"Quantized RAM Footprint : {int8_memory} Bytes (8-bit integer precision)")

print("\n--- Step 3: De-Quantization Simulation (Inference Layer Lookup) ---")
# Reconstruction back to float for structural calculations
dequantized_weights = int8_weights / quant_scale
print(f"De-quantized Floating Sample: {dequantized_weights[:3]}")

# Calculate Structural Loss Error (Mean Absolute Error)
quantization_loss = np.mean(np.abs(fp32_weights - dequantized_weights))

print("\n--- Final Performance Architecture Metrics ---")
compression_ratio = fp32_memory / int8_memory
print(f"🟢 Storage Compression Ratio : {compression_ratio:.1f}x Smaller!")
print(f"🎯 Quantization Precision Loss: {quantization_loss:.6f} (Extremely Low Error)")