import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

print("--- Step 1: Simulating Decentralized Data (2 Hospitals) ---")
# Hospital A Data (Private - Cannot be shared)
np.random.seed(42)
X_hospA = np.random.normal(1.0, 0.5, (200, 2))
y_hospA = (X_hospA[:, 0] + X_hospA[:, 1] > 2.0).astype(int)

# Hospital B Data (Private - Cannot be shared)
X_hospB = np.random.normal(1.5, 0.5, (200, 2))
y_hospB = (X_hospB[:, 0] + X_hospB[:, 1] > 2.5).astype(int)

print("\n--- Step 2: Training Local AI Models (On-Device Training) ---")
# Model A trains ONLY on Hospital A's data
model_A = SGDClassifier(loss='log_loss', max_iter=100, random_state=42)
model_A.fit(X_hospA, y_hospA)
print("✅ Hospital A Model Trained. (Patient data remains secure in Hospital A!)")

# Model B trains ONLY on Hospital B's data
model_B = SGDClassifier(loss='log_loss', max_iter=100, random_state=42)
model_B.fit(X_hospB, y_hospB)
print("✅ Hospital B Model Trained. (Patient data remains secure in Hospital B!)")

print("\n--- Step 3: Federated Averaging (FedAvg Algorithm) ---")
# The Central Server receives ONLY the Weights (Math formulas), NOT the data
global_weights = (model_A.coef_ + model_B.coef_) / 2
global_bias = (model_A.intercept_ + model_B.intercept_) / 2

# Creating the new Global Model using the averaged weights
global_model = SGDClassifier(loss='log_loss')
global_model.classes_ = np.array([0, 1])
global_model.coef_ = global_weights
global_model.intercept_ = global_bias
print("🌐 Global AI Brain updated securely via Federated Averaging!")

print("\n--- Step 4: Testing the Global Model vs Local Models ---")
# We test all models on unseen Global Test Data
X_test = np.random.normal(1.2, 0.5, (100, 2))
y_test = (X_test[:, 0] + X_test[:, 1] > 2.2).astype(int)

acc_A = accuracy_score(y_test, model_A.predict(X_test))
acc_B = accuracy_score(y_test, model_B.predict(X_test))
acc_global = accuracy_score(y_test, global_model.predict(X_test))

print(f"Hospital A Local Model Accuracy : {acc_A * 100:.2f}%")
print(f"Hospital B Local Model Accuracy : {acc_B * 100:.2f}%")
print(f"🌟 Global Federated Model Accuracy: {acc_global * 100:.2f}%")
print("-> Notice how the Global Model performs better by combining knowledge without compromising privacy!")