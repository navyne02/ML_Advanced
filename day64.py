import numpy as np

print("--- Step 1: Preparing the Knowledge Base (Internal Documents) ---")
# Simulating corporate internal documents
documents = [
    "Rank Projects Pvt Ltd handles heavy construction, employee grievance procedures, and safety protocol logs.",
    "The Fake Certificate Detection project utilizes advanced OCR technology to scan certificates and flag tampering.",
    "Advanced Machine Learning challenge spans 70 days covering MLOps, LLMs, and Network Security paradigms."
]

# Mocking an Embedding Model: Mapping sentences to semantic vector spaces (3 dimensions for simplicity)
# Document 0 is about Construction/Grievance
# Document 1 is about OCR/Certificate Detection
# Document 2 is about ML Challenge
doc_embeddings = np.array([
    [0.9, 0.1, 0.1],  # Doc 0 vector
    [0.1, 0.9, 0.1],  # Doc 1 vector
    [0.2, 0.2, 0.9]   # Doc 2 vector
])

print(f"✅ Indexed {len(documents)} internal enterprise documents into the Vector Memory System.")

print("\n--- Step 2: Processing Live User Query (Similarity Search) ---")
# User is asking a question about certificates
query = "How do you detect a fake academic certificate?"
# Embed the query into the same 3D vector space
query_embedding = np.array([0.15, 0.85, 0.05])

# Calculate Cosine Similarity: Dot product divided by magnitudes
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

scores = [cosine_similarity(query_embedding, doc_vec) for doc_vec in doc_embeddings]
best_doc_idx = np.argmax(scores)

print(f"User Query: '{query}'")
print(f"Vector Database Match Scores: {scores}")
print(f"🏆 Best Match Context Found: '{documents[best_doc_idx]}'")

print("\n--- Step 3: Prompt Augmentation & LLM Generation ---")
# Crafting the Open-Book Exam Prompt for the LLM
augmented_prompt = f"""
Context: {documents[best_doc_idx]}
Question: {query}

Answer the question strictly using the context provided above. If the answer is not present, say 'Information Unavailable'.
AI Response:
"""

print("Augmented Prompt sent to LLM:")
print("-" * 50)
print(augmented_prompt.strip())
print("-" * 50)

# Simulating LLM response based ONLY on the retrieved context
simulated_llm_response = "To detect a fake academic certificate, you must utilize the project setup with advanced OCR technology to scan the files and systematically flag any tampered fields."
print(f"🤖 LLM Generative Output: '{simulated_llm_response}'")