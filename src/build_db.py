from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Load everything in the folder
loader = PyPDFDirectoryLoader("corpus/")
docs = loader.load()

# 2. Split (Standard RAG settings)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# 3. Embed & Persist (Save to disk so UI doesn't have to re-do this)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./asd_vector_db" # This saves the DB to your folder
)
print(f"Successfully indexed {len(chunks)} chunks from the corpus.")