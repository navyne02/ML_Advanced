import numpy as np

print("--- Step 1: Setting up the Game Environment ---")
# 6 states (0 to 5). The Agent starts at 0, Goal is at 5.
n_states = 6
n_actions = 2  # 0: Move Left, 1: Move Right

# Q-Table initialization (Filled with zeros initially)
# Rows = States, Columns = Actions
q_table = np.zeros((n_states, n_actions))
print(f"Initial Empty Q-Table:\n{q_table}")

print("\n--- Step 2: Defining AI Learning Hyperparameters ---")
gamma = 0.9      # Discount factor (Cares about future rewards)
alpha = 0.1      # Learning rate (How fast it learns)
epsilon = 0.2    # Exploration rate (20% of the time, try a random move)
epochs = 500     # Number of games to play

print("\n--- Step 3: Training the AI (Trial & Error) ---")
for episode in range(epochs):
    state = 0 # Always start at state 0
    
    while state < n_states - 1: # While not reached the goal (state 5)
        
        # 1. Action Selection (Epsilon-Greedy Strategy)
        if np.random.rand() < epsilon:
            action = np.random.choice(n_actions) # Explore: Random move
        else:
            action = np.argmax(q_table[state])   # Exploit: Best known move

        # 2. Take Action & Observe Outcome
        # If moving left (0), state decreases. If right (1), state increases.
        if action == 0:
            next_state = max(0, state - 1) # Don't fall off the left edge
        else:
            next_state = min(n_states - 1, state + 1)
        
        # 3. Reward Logic
        if next_state == n_states - 1:
            reward = 10 # Jackpot! Reached the goal.
        else:
            reward = -1 # Penalty for wasting time in empty rooms

        # 4. The Magic Math: Bellman Equation
        # Update the Q-Table based on the reward and maximum future prediction
        future_optimal_value = np.max(q_table[next_state])
        learned_value = reward + gamma * future_optimal_value
        
        q_table[state, action] = q_table[state, action] + alpha * (learned_value - q_table[state, action])
        
        # Move to next state
        state = next_state

print("✅ Training Complete!")

print("\n--- Step 4: Displaying the Optimized Q-Table ---")
# The AI now knows exactly what to do in every room!
print("Q-Table (Rows: Rooms 0-5, Cols: [Left Score, Right Score])")
print(np.round(q_table, 2))

print("\n🧠 AI Logic Revealed:")
for s in range(n_states - 1):
    best_move = "RIGHT ➡️" if np.argmax(q_table[s]) == 1 else "LEFT ⬅️"
    print(f"If in Room {s}, best action is to go {best_move}")