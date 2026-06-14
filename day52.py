import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

print("--- Step 1: Simulating Network Traffic Data (Layer 4) ---")
# Features: [Source Port, Destination Port, Packet Size (Bytes), Duration (ms)]
# 1. Normal Traffic (e.g., Browsing HTTP/HTTPS, typical packet sizes)
np.random.seed(42)
normal_traffic = pd.DataFrame({
    'src_port': np.random.randint(1024, 65535, 1000),
    'dst_port': np.random.choice([80, 443, 53], 1000), # HTTP, HTTPS, DNS
    'packet_size': np.random.normal(500, 100, 1000),   # Average 500 bytes
    'duration': np.random.normal(50, 10, 1000)         # ~50ms duration
})

# 2. Anomalous Traffic (e.g., Port Scan, DDoS Payload, weird ports)
hacker_traffic = pd.DataFrame({
    'src_port': np.random.randint(1024, 65535, 20),
    'dst_port': np.random.choice([22, 23, 4444], 20),  # SSH, Telnet, Malware ports
    'packet_size': np.random.uniform(5000, 10000, 20), # Massive payloads
    'duration': np.random.uniform(1000, 5000, 20)      # Super long connections
})

# Combine for training (AI doesn't know which is which!)
network_data = pd.concat([normal_traffic, hacker_traffic], ignore_index=True)

print("✅ Network data captured from the router interface.\n")

print("--- Step 2: Training the AI Security Agent (Isolation Forest) ---")
# Isolation Forest isolates anomalies. It assumes abnormal data points are few and different.
# contamination=0.02 means we expect roughly 2% of traffic to be malicious.
ai_firewall = IsolationForest(contamination=0.02, random_state=42)
ai_firewall.fit(network_data)

print("✅ AI Firewall is now trained and monitoring the network!\n")

print("--- Step 3: Real-Time Packet Inspection ---")
# Let's test it with incoming live packets
incoming_packets = pd.DataFrame({
    'src_port': [54321, 12345, 60000],
    'dst_port': [443, 80, 4444], 
    'packet_size': [450, 510, 8500], # Notice the last packet is huge!
    'duration': [45, 55, 3000]       # And takes too long!
})

print("Inspecting incoming packets...")
# Predict: 1 means Normal, -1 means Anomaly (Hacker)
predictions = ai_firewall.predict(incoming_packets)

for i, pred in enumerate(predictions):
    packet_info = f"Port: {incoming_packets.iloc[i]['dst_port']}, Size: {incoming_packets.iloc[i]['packet_size']}B"
    if pred == 1:
        print(f"🟢 ALLOWED: {packet_info} -> Normal Traffic")
    else:
        print(f"🚨 BLOCKED: {packet_info} -> INTRUSION DETECTED! Dropping packet at Data-Link Layer.")