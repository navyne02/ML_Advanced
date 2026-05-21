import chromadb
from chromadb.utils import embedding_functions
from transformers import T5Tokenizer, T5ForConditionalGeneration

print("--- Step 1: Initializing LLM Brain & Vector DB Connection ---")
# 1. Setup Local LLM Pipeline
model_name = "google/flan-t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# 2. Setup Local ChromaDB Storage
chroma_client = chromadb.PersistentClient(path="./chroma_storage")
default_ef = embedding_functions.DefaultEmbeddingFunction()
collection = chroma_client.get_or_create_collection(
    name="production_rag_knowledge", 
    embedding_function=default_ef
)

print("\n--- Step 2: Indexing Enterprise Data Base ---")
# Custom corporate facts (AI does not know this naturally)
internal_docs = [
    "Construction labor grievance handling system at Rank Projects follows a 3-step escalation pathway ending at the Project Director.",
    "Environmental impact assessments in Salem mandate authorized electronic waste processing before the end of each quarter.",
    "Navodita Infotech requires official repository task submissions to trigger the automation of experience certificates."
]
doc_ids = ["policy_01", "policy_02", "policy_03"]

collection.add(documents=internal_docs, ids=doc_ids)
print("Internal documentation safely indexed into local storage vectors.")

# 3. RAG Pipeline Core Function
def execute_production_rag(user_question):
    print(f"\nUser Query: '{user_question}'")
    
    # Step A: Semantic Vector Query Retrieval
    search_results = collection.query(query_texts=[user_question], n_results=1)
    retrieved_context = search_results['documents'][0][0]
    print(f"Vector DB Retrieved Context: '{retrieved_context}'")
    
    # Step B: Augment Prompt Structure
    final_prompt = f"""
Use the Context below to answer the Query accurately. If unsure, say you don't know.

Context: {retrieved_context}
Query: {user_question}

Answer:
"""
    
    # Step C: LLM Text Generation
    input_ids = tokenizer(final_prompt, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=100)
    final_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return final_answer

print("\n--- Step 3: Executing RAG Live Testing ---")
# Let's ask a complex question about worker disputes
test_query = "What is the escalation pathway for construction worker complaints at Rank Projects?"
answer = execute_production_rag(test_query)

print(f"\n🚀 System Final Response:\n{answer}")