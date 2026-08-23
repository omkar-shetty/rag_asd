from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore
from langchain_classic.storage._lc_store import create_kv_docstore
from langchain_text_splitters import RecursiveCharacterTextSplitter

def build_retriever():
    """Builds a ParentDocumentRetriever against the persisted corpus."""

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