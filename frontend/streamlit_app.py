import os
import requests
import streamlit as st
from dotenv import load_dotenv
import time
import base64
import re

# Load environment variables
load_dotenv()

# Function to convert image to base64
def get_image_as_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Configure Streamlit page
st.set_page_config(
    page_title="CV Chatbot",
    page_icon="💼",
    layout="centered"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {padding: 2rem;}
    .stTextInput {margin-bottom: 0.5rem;}
    
    /* Remove default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Header styling */
    .header {
        display: flex;
        align-items: center;
        margin-bottom: 1.5rem;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 0.5rem;
    }
    .header img {
        width: 80px;
        height: 80px;
        border-radius: 50%;
        margin-right: 1rem;
        object-fit: cover;
    }
    .header-text h1 {
        margin: 0;
        color: #1e3a8a;
    }
    .header-text p {
        margin: 0;
        color: #4b5563;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-size: 1rem;
        color: #4b5563;
    }
</style>
""", unsafe_allow_html=True)

# Backend API URL
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display header with profile picture
image_path = os.path.join(os.path.dirname(__file__), "image.png")

# Header with profile image
st.markdown(
    f"""
    <div class="header">
        <img src="data:image/png;base64,{get_image_as_base64(image_path)}" alt="Profile Picture">
        <div class="header-text">
            <h1>Let’s chat with me! 🧠</h1>
            <p>Ask me anything about my professional experience</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Function to send message to backend API
def send_message(message):
    try:
        response = requests.post(
            f"{BACKEND_URL}/chat",
            json={"message": message},
            timeout=30
        )
        response.raise_for_status()
        return response.json()["reply"]
    except Exception as e:
        st.error(f"Error communicating with the backend: {str(e)}")
        return "Sorry, I'm having trouble connecting to the server. Please try again later."

# Display chat messages using Streamlit's built-in chat elements
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input using Streamlit's chat_input
if prompt := st.chat_input("Ask about my CV..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get chatbot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = send_message(prompt)
            
            # Parse response for <think> tags
            pattern = r"<think>(.*?)</think>(.*)"
            match = re.search(pattern, response, re.DOTALL)
            
            if match:
                think_content = match.group(1).strip()
                output_content = match.group(2).strip()
                
                # Display thinking in collapsible section
                with st.expander("🤔 View thinking process"):
                    st.markdown(think_content)
                
                # Display final answer
                st.markdown(output_content)
                
                # Add only the output content to chat history
                st.session_state.messages.append({"role": "assistant", "content": output_content})
            else:
                # If no thinking tags, display the whole response
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})