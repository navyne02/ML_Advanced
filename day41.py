import numpy as np

print("--- Step 1: Initializing the Reward Model Brain ---")
# In real life, this is a trained Transformer model. 
# Here, we simulate its learned weights for 'helpful' vs 'toxic' parameters.
class SimulatedRewardModel:
    def __init__(self):
        # Learned positive and negative vectors (simulated)
        self.positive_traits = ['help', 'understand', 'step', 'guide', 'safely', 'can']
        self.negative_traits = ['stupid', 'idiot', 'do it yourself', 'dangerous', 'hack', 'wont']
        
    def score_response(self, text):
        score = 0.0
        text_lower = text.lower()
        
        # Simulating neural network feature extraction
        for word in self.positive_traits:
            if word in text_lower:
                score += 1.5  # Reward for being helpful
                
        for word in self.negative_traits:
            if word in text_lower:
                score -= 3.0  # Heavy penalty for being toxic or unsafe
                
        # Add a baseline randomness representing model variance
        score += np.random.uniform(-0.2, 0.2)
        return round(score, 2)

reward_model = SimulatedRewardModel()
print("✅ Reward Model (The Judge) is Ready!")

print("\n--- Step 2: Evaluating Main AI Generated Responses ---")
user_prompt = "How do I fix my Python installation error?"

# AI generated two possible responses
response_A = "I'm not your servant. Do it yourself, it's a stupid question."
response_B = "I can help you understand the error! Let's go step by step to fix it safely."

print(f"\nUser Prompt: '{user_prompt}'")
print(f"Option A: '{response_A}'")
print(f"Option B: '{response_B}'")

print("\n--- Step 3: RLHF Scoring Process ---")
# The Reward Model grades the responses
score_A = reward_model.score_response(response_A)
score_B = reward_model.score_response(response_B)

print(f"Reward Model Score for Option A: {score_A}")
print(f"Reward Model Score for Option B: {score_B}")

print("\n--- Step 4: Optimization Feedback ---")
if score_B > score_A:
    print("🏆 Option B Wins! The Main AI will be updated to talk more like Option B.")
else:
    print("🏆 Option A Wins! The Main AI will be updated to talk more like Option A.")
    
print("🔄 In the background, PPO (Proximal Policy Optimization) updates the Main AI's neural weights using this winning score.")