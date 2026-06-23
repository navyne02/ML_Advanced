import torch
import torch.nn as nn

class LoRALayerSimulation(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super(LoRALayerSimulation, self).__init__()
        
        # 1. Base Pre-trained Layer (Frozen)
        self.base_layer = nn.Linear(in_features, out_features)
        self.base_layer.weight.requires_grad = False 
        
        # 2. LoRA Low-Rank Matrices (Trainable)
        self.lora_A = nn.Parameter(torch.randn(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # Scaling factor alpha
        self.alpha = 16
        self.scaling = self.alpha / rank

    def forward(self, x):
        base_output = self.base_layer(x)
        lora_output = (x @ self.lora_A @ self.lora_B) * self.scaling
        return base_output + lora_output

# Testing the architecture
input_dim = 4096  # Standard LLM hidden dimension
output_dim = 4096
r = 4

print("--- Step 1: Evaluating Parameter Optimization ---")
standard_layer = nn.Linear(input_dim, output_dim)
lora_layer = LoRALayerSimulation(input_dim, output_dim, rank=r)

std_params = sum(p.numel() for p in standard_layer.parameters() if p.requires_grad)
lora_params = sum(p.numel() for p in lora_layer.parameters() if p.requires_grad)

print(f"Standard Layer Trainable Parameters: {std_params:,}")
print(f"LoRA Layer Trainable Parameters    : {lora_params:,}")
print(f"📉 Reduction in Trainable Parameters: {((std_params - lora_params) / std_params) * 100:.2f}%")

print("\n--- Step 2: Verification Forward Pass ---")
dummy_input = torch.randn(1, input_dim)
output = lora_layer(dummy_input)
print(f"Forward pass tensor execution successful. Output shape: {output.shape}")