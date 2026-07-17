import numpy as np
import time

print("=" * 70)
print("🧪 DAY 83: DATA POISONING DETECTOR & OUTLIER SANITIZER")
print("=" * 70)

# Simulating clean vector embeddings for "GENUINE" certificates (3D features)
# Clean data cluster tightly around a specific feature space
clean_genuine_data = np.array([
    [0.12, 0.85, 0.11],
    [0.15, 0.82, 0.14],
    [0.10, 0.89, 0.09],
    [0.13, 0.84, 0.12],
    [0.14, 0.86, 0.13]
])

# An attacker secretly injected a poisoned vector labeled as "GENUINE"
# This vector has completely different properties (the backdoor trigger)
poisoned_backdoor_data = np.array([[0.85, 0.12, 0.90]]) 

# Combine into a contaminated training batch
contaminated_dataset = np.vstack([clean_genuine_data, poisoned_backdoor_data])
print(f"Total training vectors loaded for audit: {len(contaminated_dataset)}")
print("⚠️ Batch profile: Contaminated with adversarial backdoor inserts.")

print("\n--- Running Representation Space Defense Audit ---")
time.sleep(0.5)

# 1. Compute the mathematical centroid (mean) of the dataset assuming the majority is clean
batch_centroid = np.mean(contaminated_dataset, axis=0)
print(f"📊 Calculated Batch Centroid Coordinate: {batch_centroid}")

# 2. Set absolute safety threshold for Euclidean distance variance
DISTANCE_THRESHOLD = 0.50
clean_sanitized_batch = []
intercepted_poison_count = 0

# 3. Audit each training vector row
for idx, data_point in enumerate(contaminated_dataset):
    # Calculate Euclidean distance: d = sqrt(sum((x - c)^2))
    distance_from_centroid = np.linalg.norm(data_point - batch_centroid)
    print(f"   Row [{idx}] -> Spatial Distance to Centroid: {distance_from_centroid:.4f}")
    
    if distance_from_centroid > DISTANCE_THRESHOLD:
        print(f"   🚨 [POISON DETECTED] Row [{idx}] deviates violently! Dropping from training set.")
        intercepted_poison_count += 1
    else:
        clean_sanitized_batch.append(data_point)

print("\n" + "="*70)
print("🏆 DEFENSE AUDIT METRICS REPORT")
print("="*70)
print(f"Initial Malicious Contaminants: 1")
print(f"Interception Firewall Deflections: {intercepted_poison_count} Successful Blocks")
print(f"Sanitized Dataset Row Count for Safe Fine-Tuning: {len(clean_sanitized_batch)}")
print("="*70)