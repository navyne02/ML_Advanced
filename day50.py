import streamlit as st
import time

# --- Page Configuration ---
st.set_page_config(page_title="AI Assistant UI", page_icon="🤖", layout="centered")

# --- Header Section ---
st.title("🚀 Day 50 Capstone: Advanced AI Assistant")
st.markdown("""
Welcome to the grand finale of the 50-Day ML & AI Challenge! 
This web application demonstrates how to wrap complex AI architectures into a clean, user-friendly interface.
""")
st.divider()

# --- Initialize Chat History ---
# We use Streamlit's session state to remember the conversation
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Vanakkaam Naveen! I am your Day 50 AI Assistant. Ask me anything!"}
    ]

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User Input & AI Processing ---
user_input = st.chat_input("Type your AI prompt here...")

if user_input:
    # 1. Display User Message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 2. Simulate AI Thinking & Response (In production, connect LangChain/LLM here)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Simulating a streaming response from an LLM
        simulated_answer = f"As an advanced AI, I analyzed your prompt: '{user_input}'. Integration with Streamlit makes deploying these AI models incredibly smooth and accessible for end-users!"
        
        for chunk in simulated_answer.split():
            full_response += chunk + " "
            time.sleep(0.05) # Simulating processing time
            message_placeholder.markdown(full_response + "▌")
            
        message_placeholder.markdown(full_response)
    
    # Add AI response to state
    st.session_state.messages.append({"role": "assistant", "content": full_response})

st.divider()
st.caption("Built with ❤️ using Streamlit for the 50-Day ML & AI Challenge.")