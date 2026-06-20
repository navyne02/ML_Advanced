import torch
import torch.nn as nn
import time
import os

print("--- Step 1: Architecting a Deep Neural Network ---")

class LargeGameAI(nn.Module):
    def __init__(self):
        super(LargeGameAI, self).__init__()
        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 4) 

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

baseline_model = LargeGameAI()
baseline_model.eval()

# Save unoptimized weights to check file size
torch.save(baseline_model.state_dict(), "baseline_model.pt")
baseline_size = os.path.getsize("baseline_model.pt") / (1024 * 1024)
print(f"Standard Model Size: {baseline_size:.2f} MB\n")

print("--- Step 2: Executing Model Optimization (Quantization) ---")

# Simulating TensorRT optimization by converting Float32 weights to Int8
optimized_model = torch.quantization.quantize_dynamic(
    baseline_model, 
    {nn.Linear}, 
    dtype=torch.qint8
)

torch.save(optimized_model.state_dict(), "optimized_model.pt")
optimized_size = os.path.getsize("optimized_model.pt") / (1024 * 1024)
print(f"Optimized Engine Model Size: {optimized_size:.2f} MB")
print(f"📥 Memory Footprint Reduced by: {((baseline_size - optimized_size) / baseline_size) * 100:.1f}%\n")

print("--- Step 3: Benchmarking Latency & Performance (RTX Frame Check) ---")

dummy_game_state = torch.randn(1, 128)

# Benchmark Baseline Model
start_time = time.time()
for _ in range(500):
    with torch.no_grad():
        _ = baseline_model(dummy_game_state)
baseline_latency = (time.time() - start_time) / 500 * 1000

# Benchmark Optimized Model
start_time = time.time()
for _ in range(500):
    with torch.no_grad():
        _ = optimized_model(dummy_game_state)
optimized_latency = (time.time() - start_time) / 500 * 1000

print(f"⏱️ Standard Model Inference Latency : {baseline_latency:.4f} ms")
print(f"⚡ Optimized Engine Inference Latency: {optimized_latency:.4f} ms")
print(f"🚀 Speedup Factor: {baseline_latency / optimized_latency:.2f}x Faster!")

# Cleanup files
os.remove("baseline_model.pt")
os.remove("optimized_model.pt")