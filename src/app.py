import streamlit as st
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore
from langchain_classic.storage._lc_store import create_kv_docstore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(page_title="ASD Corpus Assistant")
st.title("ASD Knowledge Base")


def get_api_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        return os.getenv("GROQ_API_KEY")


@st.cache_resource() #caches the retriever
def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        collection_name="asd_corpus",
        embedding_function=embeddings,
        persist_directory="./vector_db_parent-child"
    )
    fs = LocalFileStore("./parent_store")
    store = create_kv_docstore(fs)
    return ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20),
        parent_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100),
    )


def log_interaction(query, answer, docs):
    try:
        os.makedirs("data", exist_ok=True)
        entry = {
            "timestamp": str(datetime.now()),
            "query": query,
            "answer": answer,
            "retrieved_context": [doc.page_content for doc in docs]
        }
        with open("data/rag_logs.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        st.warning(f"Failed to log interaction: {e}")


api_key = get_api_key()
if not api_key:
    st.error("GROQ_API_KEY not found. Add it to .env (local) or Streamlit secrets (cloud).")
    st.stop()

retriever = load_retriever()
llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0, api_key=api_key)
prompt = ChatPromptTemplate.from_template(
    "Context: {context}\n\nQuestion: {question}\n\nAnswer concisely based on the documents. "
    "If the documents do not contain relevant information, say so."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if query := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").write(query)

    try:
        docs = retriever.invoke(query)
        context = "\n".join(doc.page_content for doc in docs)
        response = llm.invoke(prompt.format(context=context, question=query))
        answer = response.content

        log_interaction(query, answer, docs)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)
    except Exception as e:
        st.error(f"Error: {e}")
        st.session_state.messages.pop()
