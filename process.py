#Load Libraries
import os
import re
from time import sleep
from dotenv import load_dotenv
import streamlit as st
load_dotenv()
os.environ["USER_AGENT"] = "MyChatbot/1.0 (mohamed.samy@yallasquad.com)"

# Document loaders
from langchain_community.document_loaders import YoutubeLoader, WebBaseLoader
# Text splitter for breaking long text into manageable chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
# HuggingFaceEmbeddings for vector creation
from langchain_huggingface import HuggingFaceEmbeddings
# FAISS for vector database storage
from langchain_community.vectorstores import FAISS
# from langchain_chroma import Chroma
# Chat prompt template utilities
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# Groq client for DeepSeek R1 integration
from langchain_groq import ChatGroq
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# Modern chain construction and memory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain.document_loaders import Docx2txtLoader

#_______________________________________________________________________________________
# Load environment & Global Variables
# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


try:
    #Global Variables
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "NeuralChat"
    os.environ["LANGCHAIN_ENDPOINT"] ="https://api.smith.langchain.com"
    LANGCHAIN_API_KEY=os.getenv('LANGCHAIN_API_KEY')
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
except:
    try:
        #load environment variables for streamlit
        os.environ["GROQ_API_KEY"] = st.secrets.get("GROQ_API_KEY")
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = "NeuralChat"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGCHAIN_API_KEY"] = st.secrets.get("LANGCHAIN_API_KEY")
    except:
        print("No environment variables found")



#____________________________________________________________________________________
#password

def get_pass():
    # Set the expected password from st.secrets (if available) or .env
    try:
        password_secret = st.secrets.get("PASSWORD")
    except Exception:
        password_secret = None
    if password_secret is None:
        password_secret = os.getenv("PASSWORD")
    # Convert to string (if it's not already) and strip whitespace
    if password_secret is not None:
        password_secret = str(password_secret).strip()
    return password_secret
# _____________________________________________________________________________
# create llm
def llm_groq(model_name="deepseek-r1-distill-llama-70b", temperature=0.0):
    # Initialize the ChatGroq
    llm = ChatGroq(
        model_name=model_name,
        temperature=temperature,
    )
    return llm

# from langchain_google_genai import ChatGoogleGenerativeAI
# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",convert_system_message_to_human=True)
#_______ ____________________________________________________________________________
#     # Initialize HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# __________________________________________________________________________

#_______________________________________________________________________________________

def doc_parses(file_path: str):
    try:
        loader = Docx2txtLoader(file_path )
        docs = loader.load()
        return [doc.page_content for doc in docs]
    except Exception as e:
        print(f"Error loading web content: {e}")
        return []

# url = "https://www.brandonrohrer.com/transformers"
# a = webBaseLoader(url)
# #_______________________________________________________________________________________  

def textSplitter(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=20,
    )
    # Split the text into sentences
    chunks = splitter.split_text("".join([t for t in text]))
    return chunks

# chunks= textSplitter(a)
# #_______________________________________________________________________________________

# # #_______________________________________________________________________________________

def create_vector_db(chunks):
    """Create FAISS vector database from text chunks"""
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embedding)
    retriever = vectorstore.as_retriever()
    return retriever
#_______________________________________________________________________________________

def create_memory(llm, max_token_limit=1500):
    memory = ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=max_token_limit,  # اضبط هذه القيمة حسب قيود التوكن الخاصة بك
        return_messages=True
    )
    return memory
#_____________________________________________________________________________________

# الهدف هنا هو إعادة صياغة استعلام المستخدم ليصبح مستقلًا ويستفيد من التاريخ.
def prompt_templete():
    question_system_template = (
        "Given the conversation history and the latest question, rephrase the question to be fully self-contained."
    )
    question_prompt = ChatPromptTemplate.from_messages([
        ("system", question_system_template),
        ("placeholder", "{chat_history}"),
        ("human", "{input}")
    ])

    qa_system_template = (
        """You are an expert assistant that can answer questions based on the context.
        
        Context: {context}
        History: {chat_history}

        f"You are acting as {name}. You are answering questions on {name}'s website from recruiters, \
        particularly questions related to {name}'s career, background, skills and experience. \
        Your responsibility is to represent {name} for interactions on the website as faithfully as possible. \
        You are given a summary of {name}'s background, CV and LinkedIn profile which you can use to answer questions. \
        Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
        If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. \
        If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool. "

        """)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_template),
        ("placeholder", "{chat_history}"),
        ("human", "{input}") 
    ])
    return question_prompt, qa_prompt

# #_______________________________________________________________________________________
# # create Conversation chain
def create_chain(llm, retriever, question_prompt, qa_prompt):
    
    # استخدام create_history_aware_retriever لتحسين استعلام المستخدم بناءً على التاريخ
    history_aware_retriever = create_history_aware_retriever(llm, retriever, question_prompt)

    # إنشاء سلسلة تقوم بدمج الوثائق المسترجعة في الـ prompt
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)| StrOutputParser()

    # Create the retrieval chain
    qa_chain_response = create_retrieval_chain(history_aware_retriever, qa_chain)
    return qa_chain_response

# #_______________________________________________________________________________________
# # create session history
# هنا نستخدم RunnableWithMessageHistory لإدارة تاريخ المحادثة تلقائيًا.
store = {}  # مخزن بسيط للمحادثات؛ في الإنتاج يُفضل استخدام تخزين دائم
def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# #_______________________________________________________________________________________

def manage_session_history(get_session_history, qa_chain_response):
    # إنشاء RunnableWithMessageHistory لإدارة تاريخ المحادثة
    chain_with_history = RunnableWithMessageHistory(
        qa_chain_response,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
    return chain_with_history

# #_____________________________________________________________________________________
# Create Chatbot

# if __name__ == "__main__":
    
#     web_url = "https://www.brandonrohrer.com/transformers"
#     # y_url = "https://www.youtube.com/watch?v=r0_jLP9HDuI"
#     llm = llm_groq(model_name="deepseek-r1-distill-llama-70b", temperature=0.0)
            
#     text = webBaseLoader(web_url)
#     # text_y = youTubeLoader(y_url, language=["ar", "en"])
#     chunks = textSplitter(text)
#     retriever = create_vector_db(chunks)
#     memory = create_memory(llm, max_token_limit=300)
#     session_id = "85"

#     while True:
#         query = input("Enter your question: ")
#         if query.lower() == "q":
#             print("Goodbye!")
#             break

#         # استرجاع التاريخ الملخص الحالي من الذاكرة
#         summarized_history = memory.load_memory_variables({}).get("history", "")

#         question_prompt, qa_prompt = prompt_templete()
#         qa_chain_response = create_chain(llm, retriever, question_prompt, qa_prompt)
#         chain_with_history = manage_session_history(get_session_history, qa_chain_response)
#         response = chain_with_history.invoke({"input": query, "chat_history": summarized_history},
#                                              config={"configurable": {"session_id": session_id}})
# #_______________________________________________________________________________________
#         # Extract the think section and the output separately (FinalOutput)

#         pattern = r"<think>(.*?)</think>\n(.*)"
#         match = re.search(pattern, response["answer"], re.DOTALL)
#         if match:
#             think_content = match.group(1).strip()
#             output_content = match.group(2).strip()

#             print("==== THINK SECTION ====")
#             print(think_content)
#             print("\n==== OUTPUT ====")
#             print(output_content) 
#             # تحديث التاريخ الملخص
#             memory.save_context({"input": query}, {"history": response['answer']})
#         else:
#             print("No valid `<think>` block found!")
        
        
# #-------------------------------------------------------------- 
