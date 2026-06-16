import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("--- Step 1: Simulating DNS Query Features ---")
# Features: [Query Length, Number of Digits, Subdomain Count, Entropy Score]

# 1. Normal DNS Traffic (e.g., google.com, github.com)
normal_queries = pd.DataFrame({
    'query_length': np.random.randint(10, 25, 500),
    'num_digits': np.random.randint(0, 3, 500),
    'subdomain_count': np.random.choice([1, 2], 500),
    'entropy': np.random.uniform(2.0, 3.5, 500),
    'label': 0 # 0 = Safe
})

# 2. DNS Tunneling Traffic (Encoded encrypted payloads hidden in subdomains)
tunneling_queries = pd.DataFrame({
    'query_length': np.random.randint(60, 120, 100), # Long encoded strings
    'num_digits': np.random.randint(15, 40, 100),    # High amount of numbers
    'subdomain_count': np.random.randint(4, 8, 100),  # Multiple subdomains chained
    'entropy': np.random.uniform(4.5, 5.8, 100),     # Extremely high structural randomness
    'label': 1 # 1 = Malicious Attack
})

# Merge datasets
df = pd.concat([normal_queries, tunneling_queries], ignore_index=True)
X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_test_split=0.2, random_state=42)
print(f"Captured {len(df)} total active DNS packets for auditing.\n")

print("--- Step 2: Training the Secure Random Forest Classifier ---")
# Random Forest is highly effective for tabular network telemetry logs
dns_guard = RandomForestClassifier(n_estimators=50, random_state=42)
dns_guard.fit(X_train, y_train)

# Evaluate model
y_pred = dns_guard.predict(X_test)
print(f"📊 Firewall AI Accuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")

print("--- Step 3: Inspecting Unseen Production Network Packets ---")
# Real-time packet test
# Packet 1: Standard enterprise lookup
# Packet 2: Suspiciously long encrypted query routing to an external domain
live_traffic = pd.DataFrame([
    [15, 1, 1, 2.8],  # Packet A
    [94, 28, 6, 5.4]  # Packet B
], columns=['query_length', 'num_digits', 'subdomain_count', 'entropy'])

predictions = dns_guard.predict(live_traffic)

traffic_names = ["Packet A (Internal Web Lookup)", "Packet B (External Node Sync)"]
for idx, pred in enumerate(predictions):
    if pred == 0:
        print(f"🟢 [PASSED] {traffic_names[idx]} -> Clean Query. Resolved via Port 53.")
    else:
        print(f"🚨 [BLOCKED] {traffic_names[idx]} -> DNS TUNNELING EXFILTRATION DETECTED! Alerting Network Admins.")