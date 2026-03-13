from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore, LocalFileStore
from langchain_classic.storage._lc_store import create_kv_docstore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os
import shutil
import sys
import argparse


def build_vector_store_recursive(docs, persist_directory):
    """Build vector database using vanilla RecursiveCharacterTextSplitter"""
    if os.path.exists(persist_directory):
            print(f"Cleaning existing directory: {persist_directory}")
            shutil.rmtree(persist_directory)

    print("Building database with standard RecursiveCharacterTextSplitter...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks.")
    
    print("Generating embeddings and building vector database...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        collection_name="asd_corpus",
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    print(f"Successfully indexed {len(chunks)} chunks from the corpus.")
    return vectorstore


def build_database_parent_child(docs, persist_directory):
    """Build retriever using Parent-Child splitter strategy"""
    if os.path.exists(persist_directory):
            print(f"Cleaning existing directory: {persist_directory}")
            shutil.rmtree(persist_directory)
    print("Building retriever with Parent-Child splitter strategy...")
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        collection_name="asd_corpus", 
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    fs = LocalFileStore("./parent_store")
    store = create_kv_docstore(fs)
    
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    
    retriever.add_documents(docs, ids=None)
    print(f"Successfully added {len(docs)} documents to retriever.")
    return retriever


def build_vector_store_semantic(docs, persist_directory):
    """Build vector database using semantic chunking based on embeddings"""
    if os.path.exists(persist_directory):
            print(f"Cleaning existing directory: {persist_directory}")
            shutil.rmtree(persist_directory)

    print("Building database with SemanticChunker...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Initialize semantic chunker with embeddings
    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile"
    )
    
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks using semantic splitting.")
    
    print("Generating embeddings and building vector database...")
    vectorstore = Chroma.from_documents(
        collection_name="asd_corpus",
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    print(f"Successfully indexed {len(chunks)} chunks from the corpus.")
    return vectorstore


def main():
    parser = argparse.ArgumentParser(description="Build vector database for ASD RAG system")
    parser.add_argument(
        "--splitter",
        type=str,
        choices=["standard", "parent-child", "semantic"],
        default="standard",
        help="Splitter strategy to use (default: standard)"
    )
    args = parser.parse_args()
    
    try:
        # Load everything in the folder
        print("Loading PDFs from corpus...")
        loader = PyPDFDirectoryLoader("corpus/")
        docs = loader.load()
        print(f"Loaded {len(docs)} documents.")

        # Split based on selected strategy
        print(f"Splitting documents using {args.splitter} strategy...")
        path = f"./vector_db_{args.splitter}"
        if args.splitter == "standard":
            build_vector_store_recursive(docs, path)
        elif args.splitter == "parent-child":
            build_database_parent_child(docs, path)
        elif args.splitter == "semantic":
            build_vector_store_semantic(docs, path)
            
    except FileNotFoundError as e:
        print(f"Error: Corpus directory or files not found: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error building vector database: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()