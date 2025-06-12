import os
import streamlit as st
import time
import base64
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from backend.cv_ingest import extract_cv_content, detect_language
from langchain_core.messages import SystemMessage, HumanMessage

# Load environment variables
load_dotenv()

# Initialize Groq client
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY environment variable is not set")
    st.stop()

# Load CV content at startup
CV_PATH = os.path.join(os.path.dirname(__file__), "cv.docx")
CV_CONTENT = extract_cv_content(CV_PATH)

if not CV_CONTENT:
    st.error("Failed to load CV content. Please check the file path.")
    st.stop()

# Initialize the ChatGroq client
def get_groq_client(model_name="deepseek-r1-distill-llama-70b", temperature=0.7):
    return ChatGroq(
        model_name=model_name,
        temperature=temperature,
    )

# Function to process chat messages
def process_message(message, model="deepseek-r1-distill-llama-70b", temperature=0.7):
    start_time = time.time()
    
    # Detect language of the incoming message
    lang = detect_language(message)
    
    # Prepare system message based on language
    if lang == 'ar':
        system_message = """أنت مساعد مفيد ومحترف. أنت تمثل صاحب السيرة الذاتية وتجيب على أسئلة المستخدمين حول خبراتك ومهاراتك ومؤهلاتك.
        يجب أن تستمد جميع الإجابات فقط من محتوى السيرة الذاتية المرفقة. إذا سُئلت عن معلومات غير موجودة في السيرة الذاتية، فقل بأدب إنك لا تملك هذه المعلومات. واجعله يتواصل معى عن طريق وسائل التواصل الموجوده بال CV
        احتفظ بالردود موجزة ومهنية ومباشرة. حاول أن تكون مفيدًا قدر الإمكان للمستخدمين الذين يستفسرون عن مؤهلاتك.
        
        قم بتضمين تفكيرك في علامات <think></think> قبل إجابتك النهائية. سيتم عرض هذا التفكير في قسم قابل للطي في واجهة المستخدم."""
    else:  # Default to English
        system_message = """You are a helpful and professional assistant. You are representing the CV owner and answering users' questions about your experience, skills, and qualifications.
        All answers must be drawn only from the content of the uploaded CV. If asked about information not in the CV, politely state that you don't have that information, And let him contact me through the means of communication available in the CV.
        Keep replies concise, professional, and on-point. Try to be as helpful as possible to users inquiring about your qualifications.
        
        Include your thinking process within <think></think> tags before your final answer. This thinking will be shown in a collapsible section in the UI."""
    
    try:
        # Initialize Groq client with requested model and temperature
        groq_client = get_groq_client(model_name=model, temperature=temperature)
        
        # Convert messages to the format expected by langchain
        langchain_messages = [
            SystemMessage(content=system_message),
            SystemMessage(content=f"CV Content: {CV_CONTENT}"),
            HumanMessage(content=message)
        ]
        
        # Get response from Groq
        response = groq_client.invoke(langchain_messages)
        
        # Extract the content from the response
        reply = response.content
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        return reply, processing_time
    except Exception as e:
        processing_time = time.time() - start_time
        return f"Error generating response: {str(e)}", processing_time

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

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display header with profile picture
image_path = os.path.join(os.path.dirname(__file__), "frontend", "image.png")

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

# Display chat messages using Streamlit's built-in chat elements
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input using Streamlit's chat_input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get chatbot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response, processing_time = process_message(prompt)
            
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