import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

print("--- Step 1: Simulating Web Server Traffic Data ---")
# Simulating normal user traffic (e.g., 100 to 500 requests per minute)
np.random.seed(42)
normal_traffic = np.random.normal(loc=300, scale=50, size=500)

# Simulating Hacker / Botnet attacks (e.g., sudden spikes of 1000+ requests)
hacker_traffic = np.random.uniform(low=1000, high=1500, size=20)
server_crashes = np.random.uniform(low=10, high=50, size=5) # Server dying

# Combine all data and shuffle
all_traffic = np.concatenate([normal_traffic, hacker_traffic, server_crashes])
np.random.shuffle(all_traffic)

# Convert to a 2D format required by scikit-learn
X = all_traffic.reshape(-1, 1)
print(f"Total Server Logs Collected: {len(X)} minutes of data.")

print("\n--- Step 2: Training the Isolation Forest AI ---")
# contamination=0.05 means we suspect about 5% of our data might be attacks
ai_detector = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
ai_detector.fit(X)

print("\n--- Step 3: Scanning for Anomalies (Intrusion Detection) ---")
# Predict: Returns 1 for Normal, -1 for Anomaly
predictions = ai_detector.predict(X)

# Filter and display the detected anomalies
anomalies = X[predictions == -1]
print(f"🚨 ALERT! Detected {len(anomalies)} suspicious network activities!")
print(f"Sample Hacker/Crash Data Points Detected: {anomalies[:5].flatten()} requests/min")

# Visualization (If you want to see the graph)
plt.figure(figsize=(10, 5))
plt.title("Day 34: AI Server Traffic Intrusion Detection")
plt.plot(all_traffic, color='blue', alpha=0.6, label='Normal Traffic')

# Mark anomalies in RED
anomaly_indices = np.where(predictions == -1)[0]
plt.scatter(anomaly_indices, all_traffic[anomaly_indices], color='red', label='Anomalies / Attacks')

plt.xlabel("Time (Minutes)")
plt.ylabel("API Requests per Minute")
plt.legend()
plt.show()