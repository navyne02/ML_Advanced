import numpy as np
import pandas as pd

print("--- Step 1: Simulating Network Connection Timestamps (1 Hour) ---")

# 1. Human Traffic: Random gaps between actions (Poisson distribution)
np.random.seed(42)
human_gaps = np.random.exponential(scale=45, size=100) # Gaps average around 45 seconds
human_timestamps = np.cumsum(human_gaps)

# 2. Malware Beaconing: Strict periodic gaps (e.g., every 30 seconds + tiny 1s jitter)
bot_gaps = 30.0 + np.random.uniform(-1, 1, 100)
bot_timestamps = np.cumsum(bot_gaps)

print("Timestamps captured for Endpoint_A (Human) and Endpoint_B (Suspected Bot).\n")

print("--- Step 2: Feature Engineering (Time Delta Analysis) ---")

def analyze_traffic_periodicity(timestamps):
    # Calculate the difference between consecutive packets (Inter-Arrival Time)
    iats = np.diff(timestamps)
    
    # Statistical features
    mean_iat = np.mean(iats)
    std_iat = np.std(iats) # Standard Deviation (How much the gaps vary)
    
    # Coefficient of Variation (CV) = std / mean
    # Low CV means the timing is highly mechanical and predictable!
    cv = std_iat / mean_iat
    return mean_iat, std_iat, cv

mean_h, std_h, cv_h = analyze_traffic_periodicity(human_timestamps)
mean_b, std_b, cv_b = analyze_traffic_periodicity(bot_timestamps)

print(f"📊 Endpoint_A (Human) -> Mean Gap: {mean_h:.2f}s, StdDev: {std_h:.2f}s, Coeff of Variation: {cv_h:.4f}")
print(f"📊 Endpoint_B (Bot)   -> Mean Gap: {mean_b:.2f}s, StdDev: {std_b:.2f}s, Coeff of Variation: {cv_b:.4f}\n")

print("--- Step 3: AI Threshold Execution (Annihilating Hidden Threat) ---")

# Rule-of-thumb in network telemetry: Automated beaconing typically has a CV < 0.15
def run_beaconing_detector(cv, endpoint_name):
    if cv < 0.15:
        print(f"🚨 [CRITICAL ALERT] {endpoint_name} exhibits strict periodic behavior! MALWARE BEACONING DETECTED.")
        print(f"👉 Action: Isolating IP from the core switch instantly to block C2 access.")
    else:
        print(f"🟢 [SAFE] {endpoint_name} behavior matches standard human burst patterns. Access allowed.")

run_beaconing_detector(cv_h, "Endpoint_A (Developer Laptop)")
print("-" * 60)
run_beaconing_detector(cv_b, "Endpoint_B (Database Server)")