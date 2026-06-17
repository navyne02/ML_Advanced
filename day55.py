import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

print("--- Step 1: Simulating Encrypted VPN Traffic Metadata ---")
# Features: [Mean Packet Size (Bytes), Packet Size Variance, Mean Inter-Arrival Time (ms), Flow Duration (s)]

# 1. Video Streaming: Continuous steady bursts, uniform sizes, low delays
video_flow = pd.DataFrame({
    'mean_packet_size': np.random.normal(1200, 50, 400),
    'packet_size_var': np.random.normal(5000, 500, 400),
    'mean_iat': np.random.normal(10, 2, 400), # Very consistent arrivals
    'flow_duration': np.random.uniform(60, 1800, 400), # Long duration
    'traffic_type': 0 # 0 = Video Streaming
})

# 2. Bulk File Transfer: Maximum packet sizes, massive variance, spikes in timing
file_flow = pd.DataFrame({
    'mean_packet_size': np.random.normal(1450, 20, 400), # Pushing MTU limits
    'packet_size_var': np.random.normal(2000, 200, 400),
    'mean_iat': np.random.normal(40, 10, 400), # Larger gaps between bursts
    'flow_duration': np.random.uniform(5, 300, 400), # Shorter massive bursts
    'traffic_type': 1 # 1 = File Transfer
})

# Combine encrypted network database
df = pd.concat([video_flow, file_flow], ignore_index=True)
X = df.drop('traffic_type', axis=1)
y = df['traffic_type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Captured metadata profiles for {len(df)} encrypted IPsec tunnel flows.\n")

print("--- Step 2: Training the Fingerprinting Classifier (Gradient Boosting) ---")
# Gradient Boosting is excellent for catching minute differences in continuous metadata metrics
traffic_ai = GradientBoostingClassifier(n_estimators=100, random_state=42)
traffic_ai.fit(X_train, y_train)

# Evaluate on unseen test tunnels
y_pred = traffic_ai.predict(X_test)
print("📊 Traffic Classification Audit Report:")
print(classification_report(y_test, y_pred, target_names=['Video Streaming', 'File Transfer']))

print("--- Step 3: Classifying Live Encrypted VPN Flows ---")
# Scenario: Inspecting 2 active encrypted tunnels right now without opening the payload
# Tunnel Alpha: Consistent mid-size packets arriving fast
# Tunnel Beta: Maximum packet sizes arriving in large staggered bursts
live_vpn_tunnels = pd.DataFrame([
    [1180, 5200, 9.5, 650.0],  # Tunnel Alpha
    [1460, 1950, 45.0, 45.0]   # Tunnel Beta
], columns=['mean_packet_size', 'packet_size_var', 'mean_iat', 'flow_duration'])

predictions = traffic_ai.predict(live_vpn_tunnels)

tunnel_names = ["Tunnel Alpha (Encrypted Node 1)", "Tunnel Beta (Encrypted Node 2)"]
for idx, pred in enumerate(predictions):
    if pred == 0:
        print(f"🟢 [MONITOR] {tunnel_names[idx]} Identified as: VIDEO STREAMING (Optimizing QoS Bandwidth).")
    else:
        print(f"⚡ [ALERT] {tunnel_names[idx]} Identified as: FILE TRANSFER (Auditing for potential Data Exfiltration!).")