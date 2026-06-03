import torch
import torch.nn as nn
import torch.nn.functional as F

print("--- Step 1: Initializing the DQN Brain ---")

# Creating the Neural Network architecture for the RL Agent
class SurvivalDQNAgent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(SurvivalDQNAgent, self).__init__()
        # Input layer receives the environment state
        self.fc1 = nn.Linear(state_dim, 64)
        # Hidden layer processes the complex patterns
        self.fc2 = nn.Linear(64, 64)
        # Output layer gives Q-values for each possible action
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x) # No activation on output, we need raw Q-values

# Define the Environment Dimensions
# State: [Health (%), Ammo Count, Distance to Enemy (m), Distance to Safe Zone (m)]
state_dim = 4

# Actions: [0: Sprint, 1: Shoot, 2: Heal, 3: Take Cover]
action_dim = 4

# Instantiate the AI Agent
agent_brain = SurvivalDQNAgent(state_dim, action_dim)
print("✅ Agent Brain Built Successfully!\n")
print(agent_brain)

print("\n--- Step 2: Simulating a Decision (Forward Pass) ---")
# Scenario: Low Health (20%), Good Ammo (150), Enemy is close (10m), Zone is far (500m)
# We normalize these values for the neural network (0 to 1 scale roughly)
current_state = torch.tensor([0.20, 0.75, 0.10, 0.90], dtype=torch.float32)

print(f"Current Environment State Tensor: {current_state.tolist()}")

# Feed the state into the AI to get action values
# We use torch.no_grad() because we are just predicting, not training yet
with torch.no_grad():
    q_values = agent_brain(current_state)
    best_action_index = torch.argmax(q_values).item()

actions_list = ["SPRINT 🏃", "SHOOT 🔫", "HEAL 🩹", "TAKE COVER 🛡️"]

print(f"\nCalculated Q-Values for Actions: {q_values.tolist()}")
print(f"🎮 AI decides to execute: {actions_list[best_action_index]} (Highest Q-Value)")