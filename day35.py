import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt

print("--- Step 1: Simulating Bank Customer Data ---")
# Creating synthetic data for 500 customers
np.random.seed(10)
data_size = 500
df = pd.DataFrame({
    'Credit_Score': np.random.randint(300, 850, data_size),
    'Income': np.random.randint(20000, 150000, data_size),
    'Loan_Amount': np.random.randint(5000, 500000, data_size),
    'Existing_Debt': np.random.randint(0, 50000, data_size)
})

# Secret Rule for approval (AI has to learn this)
score = (df['Credit_Score']/850)*0.4 + (df['Income']/150000)*0.4 - (df['Loan_Amount']/500000)*0.1 - (df['Existing_Debt']/50000)*0.1
y = (score > 0.4).astype(int)

print("\n--- Step 2: Training the 'Black-Box' AI ---")
# Random Forest is highly accurate but very hard to explain
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(df, y)
print("Random Forest Model Trained Successfully! 🌳")

print("\n--- Step 3: SHAP Explainer (Asking the AI 'WHY?') ---")
# Let's pick a specific customer (Customer ID: 42)
customer_id = 42
customer_data = df.iloc[[customer_id]]
prediction = model.predict(customer_data)[0]

print(f"\n[Customer {customer_id} Profile]")
print(customer_data.to_string(index=False))
print(f"\n🤖 AI Decision: {'APPROVED ✅' if prediction == 1 else 'REJECTED ❌'}")

print("\n--- AI Reasoning Breakdown (SHAP Values) ---")
# Initialize SHAP Tree Explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(customer_data)

# Handle different shap output formats based on version
if isinstance(shap_values, list):
    vals = shap_values[1][0] # Focus on Class 1 (Approved)
elif len(shap_values.shape) == 3:
    vals = shap_values[0, :, 1]
else:
    vals = shap_values[0]

# Print the impact of each feature
for feature, val in zip(df.columns, vals):
    impact = "POSITIVE 📈 (Helped get the loan)" if val > 0 else "NEGATIVE 📉 (Pushed towards rejection)"
    print(f"{feature:15} | Impact: {val:+.4f} | {impact}")

print("\nGenerating SHAP Summary Graph... (Close the image window to finish)")
# Generate global explanation for all customers
shap_values_all = explainer.shap_values(df)
vals_all = shap_values_all[1] if isinstance(shap_values_all, list) else shap_values_all[:, :, 1] if len(shap_values_all.shape) == 3 else shap_values_all

shap.summary_plot(vals_all, df, show=False)
plt.title("SHAP Global Feature Importance")
plt.tight_layout()
plt.show()