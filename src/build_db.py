from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import sys

try:
    # 1. Load everything in the folder
    print("Loading PDFs from corpus...")
    loader = PyPDFDirectoryLoader("corpus/")
    docs = loader.load()
    print(f"Loaded {len(docs)} documents.")

    # 2. Split (Standard RAG settings)
    print("Splitting documents...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    # 3. Embed & Persist (Save to disk so UI doesn't have to re-do this)
    print("Generating embeddings and building vector database...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./asd_vector_db" # This saves the DB to your folder
    )
    print(f"Successfully indexed {len(chunks)} chunks from the corpus.")
except FileNotFoundError as e:
    print(f"Error: Corpus directory or files not found: {str(e)}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Error building vector database: {str(e)}", file=sys.stderr)
    sys.exit(1)