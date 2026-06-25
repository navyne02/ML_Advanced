import torch
import torch.nn as nn
import time

print("--- Step 1: Initializing LLM Attention Parameters ---")
embed_dim = 256
num_heads = 4
head_dim = embed_dim // num_heads

# Linear projections for Query, Key, and Value
q_proj = nn.Linear(embed_dim, embed_dim)
k_proj = nn.Linear(embed_dim, embed_dim)
v_proj = nn.Linear(embed_dim, embed_dim)

print("✅ Attention Projection Layers Ready.")

print("\n--- Step 2: Simulating Text Generation WITHOUT KV Cache ---")
# Generates tokens 1 by 1. For each token, it recomputes everything from scratch!
seq_len = 100
cached_time_total = 0
uncached_time_total = 0

start_time = time.time()
for t in range(1, seq_len + 1):
    # Simulating the context growing step-by-step
    current_context = torch.randn(1, t, embed_dim) 
    
    with torch.no_grad():
        # Re-computing K and V for ALL tokens in the current sequence length 't'
        Q = q_proj(current_context[:, -1:, :]) # Only need query for the new token
        K = k_proj(current_context)            # Recomputing ALL keys!❌
        V = v_proj(current_context)            # Recomputing ALL values!❌
        
        # Calculate attention matrix
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))
        output = torch.matmul(attn_scores, V)

uncached_time_total = (time.time() - start_time) * 1000
print(f"❌ Without KV Cache: Total Time taken for {seq_len} tokens: {uncached_time_total:.2f} ms")


print("\n--- Step 3: Simulating Text Generation WITH KV Cache ---")
# Stores past K and V values, only projects the single incoming new token!

kv_cache_K = []
kv_cache_V = []

start_time = time.time()
for t in range(1, seq_len + 1):
    # Only the single latest token is passed to the layer!
    new_token = torch.randn(1, 1, embed_dim)
    
    with torch.no_grad():
        Q = q_proj(new_token)
        next_K = k_proj(new_token) # Project ONLY 1 token! ⚡
        next_v = v_proj(new_token) # Project ONLY 1 token! ⚡
        
        # Save to cache memory
        kv_cache_K.append(next_K)
        kv_cache_V.append(next_v)
        
        # Concatenate all past memories from cache
        K_all = torch.cat(kv_cache_K, dim=1)
        V_all = torch.cat(kv_cache_V, dim=1)
        
        # Efficient Attention calculation
        attn_scores = torch.matmul(Q, K_all.transpose(-2, -1))
        output = torch.matmul(attn_scores, V_all)

cached_time_total = (time.time() - start_time) * 1000
print(f"⚡ With KV Cache: Total Time taken for {seq_len} tokens   : {cached_time_total:.2f} ms")

print(f"\n🚀 Efficiency Gain: {uncached_time_total / cached_time_total:.2f}x Faster Inference!")
print("💡 Imagine the speedup when generating 2000 tokens in a real production system!")