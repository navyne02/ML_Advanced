import numpy as np

print("--- Step 1: Initializing Graph Structure (Nodes and Edges) ---")
# Simulating 4 Nodes (Research Papers)
# Node 0: ML Paper, Node 1: Deep Learning Paper, Node 2: SQL DB Paper, Node 3: OS Paper
node_labels = ["Machine Learning", "Deep Learning", "Databases", "Operating Systems"]

# Adjacency Matrix (A): Displays who is connected to whom
# Matrix row/col match indicates an edge link between nodes
A = np.array([
    [0, 1, 0, 0],  # Node 0 is connected to Node 1 (ML references DL)
    [1, 0, 0, 0],  # Node 1 is connected to Node 0
    [0, 0, 0, 1],  # Node 2 is connected to Node 3 (DB references OS)
    [0, 0, 1, 0]   # Node 3 is connected to Node 2
])

# 1. Add Self-Loops (A_hat = A + I) -> Every node references itself too
I = np.eye(4)
A_hat = A + I
print(f"Adjacency Matrix with Self-Loops:\n{A_hat}")

print("\n--- Step 2: Creating Node Feature Vectors (X) ---")
# Each node has a 3-dimensional profile feature (Keywords presence)
# Features match: [has_neural_word, has_data_word, has_kernel_word]
X = np.array([
    [1.0, 0.0, 0.0],  # Node 0 features
    [0.9, 0.1, 0.0],  # Node 1 features
    [0.0, 1.0, 0.0],  # Node 2 features
    [0.0, 0.0, 1.0]   # Node 3 features
])

print("\n--- Step 3: Simulating GCN Message Passing Layer ---")
# Math Rule: New_Features = A_hat X (Aggregating neighbor information)
# Multiplying adjacency with feature matrices forces nodes to blend profiles
aggregated_features = np.dot(A_hat, X)

print("\n--- Final Graph Embedded Node Feature Mapping ---")
for i in range(4):
    print(f"\nPaper {i} [{node_labels[i]}]:")
    print(f"Original Features : {X[i]}")
    print(f"GCN Aggregated Vector: {aggregated_features[i]} (Learned from neighbors!)")v