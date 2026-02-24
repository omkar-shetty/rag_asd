import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

st.set_page_config(page_title="ASD Corpus Assistant")
st.title("🏥 ASD Knowledge Base")

# --- INITIALIZE DATABASE (Cached) ---
@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory="./asd_vector_db", embedding_function=embeddings)

db = load_vector_db()
retriever = db.as_retriever(search_kwargs={"k": 5})

# --- LLM SETUP ---
llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
template = "Context: {context}\n\nQuestion: {question}\n\nAnswer concisely based on the documents:"
prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt | llm
)

# --- CHAT UI ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if query := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").write(query)
    
    response = chain.invoke(query)
    
    st.session_state.messages.append({"role": "assistant", "content": response.content})
    st.chat_message("assistant").write(response.content)