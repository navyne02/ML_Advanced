from transformers import T5Tokenizer, T5ForConditionalGeneration

# 1. Load the Model and Tokenizer (Google's Flan-T5)
# Note: 'small' version use panrom so that unga laptop-la fast-ah run aagum
model_name = "google/flan-t5-small"
print(f"Loading {model_name}... 🚀")

tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def ask_ai(prompt):
    # 2. Convert text to numbers
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    
    # 3. Generate response
    outputs = model.generate(input_ids, max_length=50)
    
    # 4. Convert numbers back to text
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# 5. Let's test our AI!
questions = [
    "What is the capital of France?",
    "How does a neural network work?",
    "Give me a step by step guide to make tea."
]

print("\n--- AI Chat Session ---")
for q in questions:
    print(f"\nUser: {q}")
    response = ask_ai(q)
    print(f"AI: {response}")