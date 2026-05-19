from sentence_transformers import SentenceTransformer, util

# 1. Load a Lightweight Embedding Model from Hugging Face
print("Loading Embedding Model (MiniLM)... 🧠")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Define Sentences to Compare
sentences = [
    "The grievance handling system manages worker complaints effectively.", # Base Sentence
    "A structured process is used to resolve construction worker disputes.", # Semantically similar
    "Python is an open-source programming language used for machine learning.", # Completely different topic
    "An environmental impact assessment evaluates sustainability and pollution." # Another different topic
]

print("\nConverting sentences into multi-dimensional vectors... 🔢")
# 3. Encode the sentences to get their embeddings
embeddings = model.encode(sentences, convert_to_tensor=True)

# Print the vector shape of the first sentence to see what it looks like
print(f"Shape of sentence 1 embedding vector: {embeddings[0].shape}")
print(f"Sample values from the vector: {embeddings[0][:5]}...")

print("\n--- Calculating Semantic Similarity Scores ---")
# 4. Compute Cosine Similarities between the first sentence and the rest
base_embedding = embeddings[0]

for i in range(1, len(sentences)):
    # Calculate similarity score (Range: -1 to 1, where 1 means identical meaning)
    similarity_score = util.cos_sim(base_embedding, embeddings[i]).item()
    print(f"\nSentence A: '{sentences[0]}'")
    print(f"Sentence B: '{sentences[i]}'")
    print(f"➡️ Semantic Similarity Match: {similarity_score * 100:.2f}%")