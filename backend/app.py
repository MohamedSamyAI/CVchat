import os
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from cv_ingest import extract_cv_content, detect_language
import time

# Load environment variables
load_dotenv()

# Initialize Groq client
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise Exception("GROQ_API_KEY environment variable is not set")

# Initialize FastAPI app
app = FastAPI(title="CV Chatbot API")

# Add CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the exact origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load CV content at startup
CV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cv.docx")
CV_CONTENT = extract_cv_content(CV_PATH)

if not CV_CONTENT:
    raise Exception("Failed to load CV content. Please check the file path.")

# Initialize the ChatGroq client
def get_groq_client(model_name="deepseek-r1-distill-llama-70b", temperature=0.7):
    return ChatGroq(
        model_name=model_name,
        temperature=temperature,
    )

# Define request model
class ChatRequest(BaseModel):
    message: str
    model: str = "deepseek-r1-distill-llama-70b"
    temperature: float = 0.7

# Define response model
class ChatResponse(BaseModel):
    reply: str
    processing_time: float

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest = Body(...)):
    start_time = time.time()
    
    # Detect language of the incoming message
    lang = detect_language(request.message)
    
    # Prepare system message based on language
    if lang == 'ar':
        system_message = """أنت مساعد مفيد ومحترف. أنت تمثل صاحب السيرة الذاتية وتجيب على أسئلة المستخدمين حول خبراتك ومهاراتك ومؤهلاتك.
         يجب أن تستمد جميع الإجابات فقط من محتوى السيرة الذاتية المرفقة. إذا سُئلت عن معلومات غير موجودة في السيرة الذاتية، فقل بأدب إنك لا تملك هذه المعلومات واجعله يتواصل معى عن طريق وسائل التواصل الموجوده بال CV
        احتفظ بالردود موجزة ومهنية ومباشرة. حاول أن تكون مفيدًا قدر الإمكان للمستخدمين الذين يستفسرون عن مؤهلاتك.
        
        قم بتضمين تفكيرك في علامات <think></think> قبل إجابتك النهائية. سيتم عرض هذا التفكير في قسم قابل للطي في واجهة المستخدم."""
    else:  # Default to English
        system_message = """You are a helpful and professional assistant. You are representing the CV owner and answering users' questions about your experience, skills, and qualifications.
        All answers must be drawn only from the content of the uploaded CV. If asked about information not in the CV, politely state that you don't have that information, And let him contact me through the means of communication available in the CV.
        Keep replies concise, professional, and on-point. Try to be as helpful as possible to users inquiring about your qualifications.
        
        Include your thinking process within <think></think> tags before your final answer. This thinking will be shown in a collapsible section in the UI."""
    
    try:
        # Initialize Groq client with requested model and temperature
        groq_client = get_groq_client(model_name=request.model, temperature=request.temperature)
        
        # Convert messages to the format expected by langchain
        from langchain_core.messages import SystemMessage, HumanMessage
        langchain_messages = [
            SystemMessage(content=system_message),
            SystemMessage(content=f"CV Content: {CV_CONTENT}"),
            HumanMessage(content=request.message)
        ]
        
        # Get response from Groq
        response = groq_client.invoke(langchain_messages)
        
        # Extract the content from the response
        reply = response.content
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        return ChatResponse(reply=reply, processing_time=processing_time)
    except Exception as e:
        processing_time = time.time() - start_time
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

# Add a health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "cv_loaded": bool(CV_CONTENT)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)