import chromadb
from chromadb.utils import embedding_functions

print("Initializing Local ChromaDB Client... 🗄️")
# 1. Initialize persistent chroma client (Saves data to a local folder named 'chroma_storage')
chroma_client = chromadb.PersistentClient(path="./chroma_storage")

# 2. Use a default lightweight embedding function from Chroma
# (Ithu internal-ah sentence-transformers model-ah use பண்ணிக்கும்)
default_ef = embedding_functions.DefaultEmbeddingFunction()

# 3. Create or Get a Collection
collection_name = "ai_challenge_knowledge_base"
print(f"Creating collection: '{collection_name}'... 🎯")
collection = chroma_client.get_or_create_collection(
    name=collection_name, 
    embedding_function=default_ef
)

# 4. Define Documents, Metadata and unique IDs to Insert
documents_data = [
    "Rank Projects Pvt Ltd implements a strict 3-step grievance handling mechanism for site construction workers.",
    "The standard operating procedure for e-waste disposal mandates authorized quarterly recycling in Salem.",
    "Navodita Infotech requires absolute submission of internship tasks before processing experience certification.",
    "Python, PyTorch, and TensorFlow form the core stack for deploying modern deep learning pipelines."
]

metadata_data = [
    {"source": "construction_policy", "location": "head_office"},
    {"source": "environmental_sop", "location": "salem_hub"},
    {"source": "internship_guidelines", "location": "remote"},
    {"source": "tech_stack", "location": "global"}
]

ids_data = ["doc_01", "doc_02", "doc_03", "doc_04"]

print("\nEmbedding and Inserting documents into ChromaDB... 📥")
# 5. Add data to the collection
collection.add(
    documents=documents_data,
    metadatas=metadata_data,
    ids=ids_data
)
print("Data successfully saved to persistent disk storage!")

# 6. Perform a Semantic Query (Concept-based Search)
user_query = "Tell me about the complaints process for construction laborers."
print(f"\nUser Query: '{user_query}'")
print("Searching Vector DB for the closest semantic match... 🔍")

results = collection.query(
    query_texts=[user_query],
    n_results=1 # Bring the top 1 closest match
)

# 7. Print the matched document and its metadata
matched_doc = results['documents'][0][0]
matched_meta = results['metadatas'][0][0]
distance_score = results['distances'][0][0]

print("\n--- ChromaDB Search Result ---")
print(f"Retrieved Document: {matched_doc}")
print(f"Source Metadata   : {matched_meta}")
print(f"Distance Score    : {distance_score:.4f} (Lower means closer match)")