import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow

print("--- Step 1: Simulating 2020 Data (Training) vs 2024 Data (Drifted) ---")
# Baseline Data (What the model learns from)
np.random.seed(42)
X_train = np.random.normal(0, 1, (1000, 2)) 
y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)

# Production Data (Things have changed! Mean shifted to 2)
X_prod = np.random.normal(2, 1.5, (500, 2))
y_prod = (X_prod[:, 0] + X_prod[:, 1] > 0).astype(int)

print("\n--- Step 2: Training & Logging with MLflow ---")
# Setting up the tracking experiment
mlflow.set_experiment("Data_Drift_Monitoring")

with mlflow.start_run():
    # 1. Train the model on old data
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # 2. Test the model on both old and new data
    train_acc = accuracy_score(y_train, model.predict(X_train))
    prod_acc = accuracy_score(y_prod, model.predict(X_prod))
    drop_in_performance = train_acc - prod_acc

    # 3. Log everything to MLflow (This is how real MLOps works!)
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("production_accuracy", prod_acc)
    mlflow.log_metric("accuracy_drop", drop_in_performance)

    print(f"✅ Baseline (Train) Accuracy: {train_acc*100:.2f}%")
    print(f"⚠️ Production (Drifted) Accuracy: {prod_acc*100:.2f}%")
    print(f"📉 Total Accuracy Drop: {drop_in_performance*100:.2f}%\n")

    if drop_in_performance > 0.10: # If accuracy drops by more than 10%
        print("🚨 ALERT: CRITICAL DATA DRIFT DETECTED!")
        print("Action Required: Retrain the model with 2024 data immediately.")

print("\n--- Step 3: View the MLflow Dashboard ---")
print("To see your interactive MLOps dashboard, run this command in your terminal:")
print("👉 mlflow ui")