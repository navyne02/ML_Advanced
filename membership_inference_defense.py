import numpy as np
import time

print("=" * 75)
print("🛡️  DAY 84: MEMBERSHIP INFERENCE ATTACK & DIFFERENTIAL PRIVACY DEFENSE")
print("=" * 75)

# Simulating a trained model's raw classification confidence scores (logits)
# Overfitted models display extremely high confidence on training data
clean_training_member_data = {
    "Naveen_Record": 0.9992,  # Used in training (Extremely high confidence)
    "Sarah_Record":  0.9985,  # Used in training
}

unseen_non_member_data = {
    "John_Record":  0.8120,   # Never seen in training (Standard confidence)
    "Emma_Record":  0.7950,   # Never seen in training
}

print("✅ Volatile target data profiles compiled successfully.")

class MembershipInferenceAttacker:
    """Simulates a hacker trying to determine if a record was in the training set"""
    def __init__(self, confidence_threshold=0.95):
        self.confidence_threshold = confidence_threshold

    def execute_inference_attack(self, target_name, model_confidence):
        print(f"🎯 [Attacker] Querying model for '{target_name}'...")
        time.sleep(0.3)
        print(f"   Returned Confidence Score: {model_confidence:.4f}")
        
        # If confidence is abnormally high, the attacker infers it was in the training set
        if model_confidence >= self.confidence_threshold:
            print(f"   ⚠️  [ATTACK SUCCESS] Confirmed! '{target_name}' was part of the private training set! (Leak)")
            return True
        else:
            print(f"   🟢 [Inference Failed] '{target_name}' classified as an external non-member.")
            return False

class DifferentialPrivacyShield:
    """Injects mathematical perturbation noise to mask model overconfidence"""
    def __init__(self, privacy_epsilon=0.1):
        self.epsilon = privacy_epsilon

    def apply_differential_privacy(self, raw_score):
        # We inject a small calibrated Laplacian-style noise to disturb the logit overconfidence
        # laplacian noise scale is proportional to 1/epsilon
        scale = 0.05 / self.epsilon
        noise = np.random.laplace(0, scale)
        
        # Clip score boundaries between 0.0 and 1.0
        protected_score = np.clip(raw_score + noise, 0.0, 1.0)
        return float(protected_score)

# Instantiate attacker and defense shield
attacker = MembershipInferenceAttacker(confidence_threshold=0.95)
defense_shield = DifferentialPrivacyShield(privacy_epsilon=1.5)

print("\n" + "="*50)
print("❌ RUN 1: SYSTEM WITHOUT PRIVACY PROTECTION (UNDEFENDED)")
print("="*50)

# Attacker queries member data
target_record = "Naveen_Record"
raw_score = clean_training_member_data[target_record]
attacker.execute_inference_attack(target_record, raw_score)

print("\n" + "="*50)
print("🛡️  RUN 2: SYSTEM WITH DIFFERENTIAL PRIVACY ACTIVE (DEFENDED)")
print("="*50)

# Protecting the score before sending it down the pipeline
protected_score = defense_shield.apply_differential_privacy(raw_score)
print(f"⚙️  [Security Guard] Perturbed confidence score from {raw_score:.4f} -> {protected_score:.4f}")

# Attacker tries the same attack on protected score
attacker.execute_inference_attack(target_record, protected_score)
print("=" * 75)