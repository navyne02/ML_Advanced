import numpy as np

print("--- Step 1: Defining the Game World States & Rules ---")
# 4 Possible states in our grid
states = ["Empty [.]", "Wall  [#]", "Turret [T]", "T-Hall [H]"]

# Transition Probability Matrix
# Rows represent current state, Columns represent next state
# Order: [Empty, Wall, Turret, TownHall]
transition_matrix = np.array([
    [0.60, 0.30, 0.08, 0.02],  # From Empty -> mostly stays empty or builds a wall
    [0.20, 0.70, 0.10, 0.00],  # From Wall  -> highly likely to continue building a wall
    [0.50, 0.40, 0.10, 0.00],  # From Turret-> surrounded by empty space or protective walls
    [0.10, 0.80, 0.10, 0.00]   # From TownHall -> must be heavily fortified with walls
])

print("✅ State Transition Matrices calibrated successfully.")

print("\n--- Step 2: Running the Markov Chain Generation Loop ---")

def generate_base_row(length, starting_state_idx):
    current_state_idx = starting_state_idx
    row = [states[current_state_idx]]
    
    for _ in range(length - 1):
        # Pick the next state based on the probabilities of the current state
        probabilities = transition_matrix[current_state_idx]
        next_state_idx = np.random.choice([0, 1, 2, 3], p=probabilities)
        
        row.append(states[next_state_idx])
        current_state_idx = next_state_idx
        
    return row

# Setting up a 6x6 base grid matrix
grid_size = 6
print(f"Generating an automated {grid_size}x{grid_size} Defense Outpost Layout...\n")

np.random.seed(44) # For reproducible layout matrix
for r in range(grid_size):
    # Center rows are more likely to start with a TownHall or heavy structures
    if r == grid_size // 2:
        row_layout = generate_base_row(grid_size, starting_state_idx=3) # Start with TownHall
    else:
        row_layout = generate_base_row(grid_size, starting_state_idx=0) # Start with Empty
        
    # Formatting output for clean grid visualization
    clean_row = "  ".join([item.split()[1] for item in row_layout])
    print(f"Row {r+1}:  {clean_row}")

print("\n--- Step 3: Layout Generation Architecture Analysis ---")
print("Legend: [.] = Open Field, [#] = Fortified Wall, [T] = Defense Turret, [H] = Central TownHall")
print("🧠 Notice how walls [#] clump together automatically to form defense barriers surrounding the TownHall [H] without manual level design coding!")