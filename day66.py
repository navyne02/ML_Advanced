import torch
import torch.nn as nn
import torch.optim as optim

print("--- Step 1: Initializing the Reward Network Architecture ---")
# A neural network that takes text embedding and outputs a single scalar reward score
class MiniRewardModel(nn.Module):
    def __init__(self, embed_dim=128):
        super(MiniRewardModel, self).__init__()
        self.layer1 = nn.Linear(embed_dim, 64)
        self.layer2 = nn.Linear(64, 1) # Output is a single floating point score

    def forward(self, embedding):
        x = torch.relu(self.layer1(embedding))
        return self.layer2(x)

reward_model = MiniRewardModel()
optimizer = optim.Adam(reward_model.parameters(), lr=0.01)

print("✅ Reward Model initialized successfully.")

print("\n--- Step 2: Simulating Human Preference Data (Chosen vs Rejected) ---")
# Prompt: "Give me feedback on my code"
# Chosen Response (Embedding): Polite, detailed, accurate code review
chosen_resp_embedding = torch.randn(1, 128) + 0.5 

# Rejected Response (Embedding): Rude, unhelpful, short response
rejected_resp_embedding = torch.randn(1, 128) - 0.5

print("Human Evaluator setup: Preference alignment datasets compiled.")

print("\n--- Step 3: Training the Reward Model (Preference Loss Optimization) ---")

# Training loop for 5 steps to watch the scores align
for step in range(1, 6):
    optimizer.zero_grad()
    
    # Calculate rewards for both responses
    chosen_reward = reward_model(chosen_resp_embedding)
    rejected_reward = reward_model(rejected_resp_embedding)
    
    # Calculate Bradley-Terry Preference Loss
    # We want chosen_reward to be HIGH and rejected_reward to be LOW
    loss = -torch.log(torch.sigmoid(chosen_reward - rejected_reward))
    
    loss.backward()
    optimizer.step()
    
    print(f"Step {step} -> Loss: {loss.item():.4f} | Chosen Reward: {chosen_reward.item():.2f} | Rejected Reward: {rejected_reward.item():.2f}")

print("\n🏆 Optimization Complete!")
print("Notice how the Reward Model dynamically learned to score the helpful response higher and penalize the bad response automatically!")