import streamlit as st
import os
import json
from datetime import datetime
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
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return Chroma(persist_directory="./asd_vector_db", embedding_function=embeddings)
    except Exception as e:
        st.error(f"Failed to load vector database: {str(e)}")
        raise

def log_rag_interaction(query, answer, docs):
    try:
        log_entry = {
            "timestamp": str(datetime.now()),
            "query": query,
            "answer": answer,
            "retrieved_context": [doc.page_content for doc in docs]
        }

        with open("data/rag_logs.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        st.warning(f"Failed to log interaction: {str(e)}")

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
    
    try:
        # response = chain.invoke(query)
        docs = retriever.invoke(query)
        context = "\n".join([doc.page_content for doc in docs])

        response = llm.invoke(
            prompt.format(context=context, question=query)
        )

        answer = response.content

        log_rag_interaction(query, answer, docs)
        
        st.session_state.messages.append({"role": "assistant", "content": response.content})
        st.chat_message("assistant").write(response.content)
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        st.session_state.messages.pop()  # Remove the user message if processing failed