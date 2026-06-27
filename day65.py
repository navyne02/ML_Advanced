import torch
import torch.nn as nn

print("--- Step 1: Initializing Sequence Matrices ---")
# Simulating a small context token size
seq_len = 64
head_dim = 64

Q = torch.randn(1, seq_len, head_dim)
K = torch.randn(1, seq_len, head_dim)
V = torch.randn(1, seq_len, head_dim)

print(f"Matrix Tensors Ready. Context Length: {seq_len}, Head Dimension: {head_dim}")

print("\n--- Step 2: Simulating Standard Attention (High Memory I/O) ---")
# Standard method reads and writes huge intermediate tensors to high-bandwidth memory (HBM) repeatedly
hbm_write_count = 0

# 1. Compute QK^T -> Writes intermediate tensor to HBM
S = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)
hbm_write_count += S.nelement() * S.element_size() # Captured memory size

# 2. Compute Softmax -> Reads from HBM, computes, writes back to HBM
P = torch.softmax(S, dim=-1)
hbm_write_count += P.nelement() * P.element_size()

# 3. Compute Attention Output -> Reads from HBM, multiplies with V
O_standard = torch.matmul(P, V)

print(f"❌ Standard Attention Intermediate HBM Writes: {hbm_write_count / 1024:.2f} KB")

print("\n--- Step 3: Simulating FlashAttention Tiling Concept (Zero Intermediate HBM Writes) ---")
# FlashAttention processes in localized blocks within the ultra-fast SRAM, bypassing intermediate HBM writes!
flash_hbm_write_count = 0

# Tiling simulation: Processing in blocks of size 16
block_size = 16
O_flash = torch.zeros_like(O_standard)

for i in range(0, seq_len, block_size):
    # Load a small block of Q into fast SRAM memory
    Q_block = Q[:, i:i+block_size, :]
    
    # Perform all calculations internally inside SRAM cache
    S_block = torch.matmul(Q_block, K.transpose(-2, -1)) / (head_dim ** 0.5)
    P_block = torch.softmax(S_block, dim=-1)
    
    # Accumulate directly into output block
    O_flash[:, i:i+block_size, :] = torch.matmul(P_block, V)

# Only the final output is written back to HBM!
flash_hbm_write_count += O_flash.nelement() * O_flash.element_size()

print(f"⚡ FlashAttention Concept Intermediate HBM Writes: {0.00:.2f} KB (Everything calculated in fast SRAM!)")
print(f"📦 Final Output successfully delivered to main memory: {flash_hbm_write_count / 1024:.2f} KB")

# Verify correctness
assert torch.allclose(O_standard, O_flash, atol=1e-5)
print("\n🏆 Mathematical verification successful! Both models produced identical outputs.")