import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from transformers import T5Tokenizer, T5ForConditionalGeneration

# 1. Page Configuration
st.set_page_config(
    page_title="Naveen's Capstone AI Chatbot", 
    page_icon="🎓", 
    layout="wide"
)

st.title("🎓 Advanced 30-Day ML & AI Challenge - Capstone Project")
st.subheader("Your Custom Enterprise Knowledge Conversational Assistant")
st.markdown("---")

# 2. Caching Models and DB connection for Extreme Inference Speed
@st.cache_resource
def initialize_system_cores():
    # Load Core LLM Weights
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
    
    # Load Persistent Vector Database Engine
    chroma_client = chromadb.PersistentClient(path="./chroma_storage")
    default_ef = embedding_functions.DefaultEmbeddingFunction()
    collection = chroma_client.get_or_create_collection(
        name="production_rag_knowledge", 
        embedding_function=default_ef
    )
    return tokenizer, model, collection

with st.spinner("Powering up AI Engine Cores and Vector Indexes... ⚡"):
    tokenizer, model, collection = initialize_system_cores()

# Sidebar Layout for System Information
with st.sidebar:
    st.header("🛠️ System Diagnostics")
    st.success("LLM Status: Connected (Flan-T5)")
    st.success("Vector DB: ChromaDB Connected")
    st.info("Storage Location: `./chroma_storage`")
    st.markdown("---")
    st.markdown("### 🎓 Developer Profile")
    st.markdown("**Name:** Er. Naveen")
    st.markdown("**Status:** Day 30 Graduation Complete! 🎉")

# 3. Maintain Chat History Session States
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior messages in the chat UI
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Accept User Input
if user_prompt := st.chat_input("Ask me about internal corporate guidelines or tech tasks..."):
    # Display user input in real-time
    with st.chat_message("user"):
        st.markdown(user_prompt)
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    # 5. Execute RAG Retrieval Logic
    with st.chat_message("assistant"):
        with st.spinner("Retrieving neural vectors and synthesising... 🧠"):
            # Step A: Hit ChromaDB for Semantic Context
            results = collection.query(query_texts=[user_prompt], n_results=1)
            
            if results['documents'] and len(results['documents'][0]) > 0:
                retrieved_context = results['documents'][0][0]
            else:
                retrieved_context = "No specific internal context available."
            
            # Step B: Craft System Prompt Injecting Private Knowledge
            rag_prompt = f"""
            Answer the user query precisely based on the provided Context.
            Context: {retrieved_context}
            Query: {user_prompt}
            Answer:
            """
            
            # Step C: Model Computation
            input_ids = tokenizer(rag_prompt, return_tensors="pt").input_ids
            outputs = model.generate(input_ids, max_length=120)
            ai_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Display response
            st.markdown(ai_response)
            
            # Informative Expander to show the developer trace
            with st.expander("🛠️ Developer Vector Log Trace"):
                st.write(f"**Retrieved Context Chunk:** {retrieved_context}")
                
        st.session_state.messages.append({"role": "assistant", "content": ai_response})