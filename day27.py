import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from transformers import T5Tokenizer, T5ForConditionalGeneration

# 1. UI Setup
st.set_page_config(page_title="Corporate AI Assistant", page_icon="🤖", layout="centered")
st.title("🏢 Rank Projects & Navodita AI Assistant")
st.write("Welcome! Ask me anything about our internal company policies.")
st.markdown("---")

# 2. Cache Models & DB (Athigama time edukkum, so cache panrom)
@st.cache_resource
def load_ai_brain():
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
    return tokenizer, model

@st.cache_resource
def load_vector_db():
    chroma_client = chromadb.PersistentClient(path="./chroma_storage")
    default_ef = embedding_functions.DefaultEmbeddingFunction()
    # Nethu create panna athe collection-ah get panrom
    collection = chroma_client.get_collection(
        name="production_rag_knowledge", 
        embedding_function=default_ef
    )
    return collection

tokenizer, model = load_ai_brain()
collection = load_vector_db()

# 3. Chat Interface
user_query = st.text_input("Type your question here (e.g., 'Tell me about e-waste disposal'):")

if st.button("Ask AI"):
    if user_query:
        with st.spinner("Searching internal database and thinking... 🧠"):
            # Step A: Retrieve from ChromaDB
            results = collection.query(query_texts=[user_query], n_results=1)
            retrieved_context = results['documents'][0][0]
            
            # Show the user what the AI found
            with st.expander("🔍 View Retrieved Context (Vector Search)"):
                st.write(retrieved_context)

            # Step B: LLM Generation
            prompt = f"Answer the query using ONLY the context provided.\n\nContext: {retrieved_context}\nQuery: {user_query}\n\nAnswer:"
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            outputs = model.generate(input_ids, max_length=100)
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Step C: Display Result
            st.success("### 🤖 AI Response:")
            st.write(f"**{answer}**")
    else:
        st.warning("Please enter a question first!")