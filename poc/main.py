import os
# from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# 1. SETUP
load_dotenv()
PDF_PATH = "data/ASD-Handbook.pdf"

print("--- Initializing OHSU RAG Test ---")

# 2. LOAD & SPLIT
print(f"Loading {PDF_PATH}...")
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(docs)
print(f"Created {len(chunks)} text chunks.")

# 3. LOCAL EMBEDDINGS (Free & Offline)
print("Loading HuggingFace model 'all-MiniLM-L6-v2'...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. VECTOR STORE
print("Indexing chunks into ChromaDB...")
vectorstore = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings,
    collection_name="ohsu_asd_test"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 5. DEFINE THE CHAIN
template = """
Use the following context from the OHSU ASD Handbook to answer the question.
Context: {context}
Question: {question}
Answer:"""

prompt = ChatPromptTemplate.from_template(template)
llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt 
    | llm 
    | StrOutputParser()
)

# 6. INTERACTIVE LOOP
print("\n--- System Ready! Type 'exit' to quit ---")
while True:
    user_input = input("\nAsk the Handbook: ")
    if user_input.lower() in ['exit', 'quit']:
        break
    
    print("Thinking...")
    response = rag_chain.invoke(user_input)
    print(f"\nOHSU Assistant: {response}")