from transformers import T5Tokenizer, T5ForConditionalGeneration

# 1. Load Model (Athe Flan-T5 use panrom)
model_name = "google/flan-t5-small"
print("Loading AI Brain... 🧠")
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def generate_structured_response(user_email):
    # 2. Crafting the Few-Shot Prompt with Examples
    prompt = f"""
Task: Analyze the email and output ONLY a valid JSON with keys "category" and "urgency".

Example 1:
Email: "My account is locked and I cannot access my dashboard help!"
Output: {{"category": "Account Access", "urgency": "High"}}

Example 2:
Email: "Is there a discount if I buy 50 licenses for my team?"
Output: {{"category": "Sales Inquiry", "urgency": "Medium"}}

Example 3:
Email: "There is a small typo in the spelling on the settings page."
Output: {{"category": "Bug Report", "urgency": "Low"}}

Actual Task:
Email: "{user_email}"
Output:
"""
    
    # 3. Generate Output
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=50)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# 4. Test with a completely new email
new_email = "URGENT: Our production server crashed and everything is down right now!!!"
print(f"\nIncoming Email: {new_email}")

json_result = generate_structured_response(new_email)
print("\n--- AI Structured Output ---")
print(json_result)