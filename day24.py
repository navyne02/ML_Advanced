from transformers import T5Tokenizer, T5ForConditionalGeneration

# 1. Load AI Brain
model_name = "google/flan-t5-small"
print("Loading Local LLM for RAG pipeline... 🧠")
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# 2. Our Private Knowledge Base (AI-ku ithu munnadi theriyaathu!)
# This could represent policy documents, project data, or internal guidelines
knowledge_base = {
    "grievance": "Rank Projects Pvt Ltd handles construction worker grievances through a 3-step system: step 1 is reporting to the site supervisor, step 2 is escalation to the HR department, and step 3 is final review by the Project Director.",
    "e-waste": "The company's standard operating procedure for e-waste disposal mandates processing through authorized recycling vendors in Salem by the end of each quarter.",
    "internship": "Navodita Infotech requires continuous documentation of internship task submissions before generating the final experience certificate."
}

# 3. Simple Retrieval Function (Keyword-based matcher)
def retrieve_context(query):
    query_lower = query.lower()
    for key, context in knowledge_base.items():
        if key in query_lower:
            return context
    return "No custom context found in internal documents."

# 4. RAG Execution Pipeline
def run_rag_pipeline(user_query):
    # Step 1: Retrieve private data
    context = retrieve_context(user_query)
    print(f"\n[Retrieved Context]: {context}")
    
    # Step 2: Augment the Prompt
    rag_prompt = f"""
Answer the user query based ONLY on the provided Context.

Context: {context}
User Query: {user_query}

Answer:
"""
    
    # Step 3: Generate Response using LLM
    input_ids = tokenizer(rag_prompt, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# 5. Test the RAG system
query_1 = "How are grievance handling systems set up for construction workers at Rank Projects?"
print(f"\nUser Question: {query_1}")
reply_1 = run_rag_pipeline(query_1)
print(f"AI Final Response: {reply_1}")

query_2 = "What is the policy for e-waste disposal in Salem?"
print(f"\nUser Question: {query_2}")
reply_2 = run_rag_pipeline(query_2)
print(f"AI Final Response: {reply_2}")