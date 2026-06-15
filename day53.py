import numpy as np
import random

print("--- Step 1: Defining the Network Topology ---")
# Routers: 0 (Source) to 5 (Destination)
# The Reward Matrix (R). 
# -100 means no physical link. 
# -1 means normal link cost. 
# -50 means HEAVY CONGESTION (Traffic Jam).
# 100 means reaching the destination.

R = np.full((6, 6), -100) # Initialize with no links
R[0, 1] = -1; R[0, 2] = -1               # Router 0 connects to 1 and 2
R[1, 0] = -1; R[1, 3] = -50              # Router 1 connects to 0 and 3. (Link to 3 is Congested!)
R[2, 0] = -1; R[2, 3] = -50; R[2, 4] = -1 # Router 2 connects to 0, 3, 4. (Link to 3 is Congested!)
R[3, 1] = -1; R[3, 2] = -1; R[3, 5] = 100 # Router 3 connects to 1, 2, 5
R[4, 2] = -1; R[4, 5] = 100              # Router 4 connects to 2, 5
R[5, 5] = 100                            # Goal state

# Q-Table to store the AI's learned routing table
Q = np.zeros((6, 6))

print("⚠️ Alert: Router 3 is experiencing massive congestion (-50 Penalty)!")
print("Goal: Send packets from Router 0 to Router 5 optimally.\n")

print("--- Step 2: Training the AI Router (Reinforcement Learning) ---")
gamma = 0.9 # Discount factor
epochs = 1000

# Training loop
for episode in range(epochs):
    # Start at a random router (except the destination)
    current_router = np.random.randint(0, 5) 
    
    while current_router != 5:
        # Find physical links (where reward is not -100)
        valid_next_hops = np.where(R[current_router] != -100)[0]
        
        # Explore: pick a random next hop
        next_router = random.choice(valid_next_hops)
        
        # Bellman Equation: Update the routing table
        future_reward = np.max(Q[next_router])
        Q[current_router, next_router] = R[current_router, next_router] + (gamma * future_reward)
        
        current_router = next_router

print("✅ AI Routing Table Updated Successfully!\n")

print("--- Step 3: Testing the Smart Route ---")
current_router = 0
optimal_path = [current_router]

# Follow the best path learned by AI
while current_router != 5:
    # Find the next router with the highest Q-Value
    next_router = np.argmax(Q[current_router])
    optimal_path.append(next_router)
    current_router = next_router

print(f"🚀 AI Calculated Optimal Path: {' -> '.join(map(str, optimal_path))}")
print("🧠 Notice how the AI completely bypassed Router 3 to avoid the traffic jam!")