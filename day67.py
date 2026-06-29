import math
import random

print("--- Step 1: Initializing Physical VRAM Memory Pool ---")
# Simulating a GPU VRAM divided into 10 fixed-size physical blocks (Pages)
# Each block can hold exactly 4 tokens' KV Cache
BLOCK_SIZE = 4
TOTAL_PHYSICAL_BLOCKS = 10

# Pool of available physical block IDs in VRAM
vram_free_blocks_pool = list(range(TOTAL_PHYSICAL_BLOCKS))
random.shuffle(vram_free_blocks_pool) # Shuffling to simulate non-contiguous memory space

print(f"Total Physical Blocks available in VRAM Pool: {vram_free_blocks_pool}")
print(f"Each block configuration capacity: {BLOCK_SIZE} tokens.\n")

print("--- Step 2: Architecture of PagedAttention Virtual Memory Manager ---")

class PagedAttentionManager:
    def __init__(self, free_pool, block_size):
        self.free_pool = free_pool
        self.block_size = block_size
        self.page_table = {} # Maps Request_ID -> List of allocated Physical Block IDs

    def allocate_space_for_request(self, request_id, token_count):
        # Calculate exactly how many blocks are required for the incoming prompt tokens
        blocks_needed = math.ceil(token_count / self.block_size)
        allocated_physical_blocks = []
        
        print(f"Request [{request_id}] incoming with {token_count} tokens. Needs {blocks_needed} memory blocks.")
        
        for _ in range(blocks_needed):
            if self.free_pool:
                # Pick a non-contiguous free block from the hardware pool
                physical_block = self.free_pool.pop(0)
                allocated_physical_blocks.append(physical_block)
            else:
                print("🚨 VRAM OUT OF MEMORY (OOM)! Cannot allocate request.")
                return False
                
        # Register mapping into the Page Table (Logical to Physical Mapping)
        self.page_table[request_id] = allocated_physical_blocks
        print(f"✅ Mapping Successful -> Request {request_id} Page Table: {self.page_table[request_id]}")
        return True

# Initialize our custom LLM memory scheduler
memory_scheduler = PagedAttentionManager(vram_free_blocks_pool, BLOCK_SIZE)

print("\n--- Step 3: Simulating Concurrent Production Requests ---")
# User 1 sends a prompt with 6 tokens (Needs 2 blocks)
memory_scheduler.allocate_space_for_request("User_Prompt_Alpha", token_count=6)

# User 2 sends a prompt with 11 tokens (Needs 3 blocks)
memory_scheduler.allocate_space_for_request("User_Prompt_Beta", token_count=11)

print(f"\nRemaining Free VRAM Blocks left in pool: {vram_free_blocks_pool}")
print("\n🧠 System Architecture Insight:")
print("Notice how the memory blocks assigned to users are non-contiguous and randomly spread, yet the Page Table links them dynamically, preventing memory fragmentation errors entirely!")