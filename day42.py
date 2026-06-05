import torch
import torch.nn as nn
import torch.nn.functional as F

print("--- Step 1: Initializing the Actor-Critic Brain ---")

# Creating a Neural Network with TWO heads (Actor and Critic)
class ActorCriticAgent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCriticAgent, self).__init__()
        
        # Shared Layer: Both Actor and Critic share the same eyes (Base features)
        self.common_layer = nn.Linear(state_dim, 128)
        
        # HEAD 1: The Actor (Outputs probabilities for actions)
        self.actor_head = nn.Linear(128, action_dim)
        
        # HEAD 2: The Critic (Outputs a single score evaluating the state)
        self.critic_head = nn.Linear(128, 1)

    def forward(self, state):
        # 1. Process environment state
        shared_features = F.relu(self.common_layer(state))
        
        # 2. Actor decides: What is the probability of each action?
        # Softmax converts raw numbers into percentages (0.0 to 1.0)
        action_probabilities = F.softmax(self.actor_head(shared_features), dim=-1)
        
        # 3. Critic judges: How good is this current state? (Single Value)
        state_value = self.critic_head(shared_features)
        
        return action_probabilities, state_value

# Dimensions: e.g., 4 State variables, 3 possible Actions
state_dim = 4
action_dim = 3
ai_agent = ActorCriticAgent(state_dim, action_dim)

print("✅ Actor-Critic Neural Network Built Successfully!\n")
print(ai_agent)

print("\n--- Step 2: Simulating the Environment (Forward Pass) ---")
# Simulating a complex game state
current_state = torch.tensor([1.2, -0.5, 0.8, 2.1], dtype=torch.float32)
print(f"Current Environment State Tensor: {current_state.tolist()}")

# Passing the state to our Two-Headed AI
with torch.no_grad():
    action_probs, critic_score = ai_agent(current_state)

print("\n--- Step 3: AI Output Analysis ---")
print("🎭 ACTOR HEAD OUTPUT:")
print(f"Probabilities for Action 0, 1, 2: {action_probs.tolist()}")
chosen_action = torch.argmax(action_probs).item()
print(f"-> Actor confidently chooses Action: {chosen_action}")

print("\n🧐 CRITIC HEAD OUTPUT:")
print(f"State Safety/