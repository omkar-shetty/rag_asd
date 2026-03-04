# ASD RAG: Clinical Evidence-Based Knowledge System


---

## Executive Summary
The **ASD RAG** (Autism Spectrum Disorder - Retrieval-Augmented Generation) system is a specialized AI assistant designed to navigate the existing literature and information available around ASD. Developed to solve the "hallucination problem" in medical AI, this system uses a retrieval pipeline to ensure responses are mathematically grounded in verified clinical documents.



---

## Technical Architecture

### 1. Data Engineering Pipeline (`src/build_db.py`)
* **Ingestion:** Batch processing of the clinical corpus via `PyPDFDirectoryLoader`.
* **Chunking Strategy:** `RecursiveCharacterTextSplitter` with `chunk_size=1000` and `overlap=200`. 
    * *Note:* This prevents cutting critical medical qualifiers or diagnostic criteria mid-sentence, preserving semantic coherence.
* **Vectorization:** `HuggingFaceEmbeddings` using the `all-MiniLM-L6-v2` transformer (384-dimensional).
* **Storage:** Local persistence via **ChromaDB**, ensuring data privacy and zero-latency infrastructure costs.

### 2. Inference & UI Engine (`src/app.py`)
* **Orchestration:** Built on **LangChain Expression Language (LCEL)** for a modular, composable chain: `retriever | prompt | llm`.
* **Intelligence:** **Llama-3.1-8b-instant** (via Groq) for sub-second inference speeds.
* **Configuration:** `Temperature=0` for deterministic, fact-based responses and `k=5` for optimal context window balance.
* **Interface:** A high-concurrency **Streamlit** chat UI.

---

## Tech Stack

| Component      | Technology              | Why?                                                                 |
| :------------- | :---------------------- | :------------------------------------------------------------------- |
| **LLM** | Groq / Llama-3.1-8b     | Inference for real-time applications. |
| **Embeddings** | HuggingFace (MiniLM)    | Lightweight and highly efficient.     |
| **Vector DB** | ChromaDB                | Local-first storage.    |
| **Logic** | LangChain (LCEL)        | Modular architecture.         |
| **Front-end** | Streamlit               | Enables rapid deployment.      |


---

## Getting Started

### 1. Index the Data
Place your PDFs in the `corpus/` directory and run the indexing script:
```bash
python src/build_db.py
```

### 2. Launch the UI
Start the Streamlit interface to interact with the assistant:
```bash
streamlit run src/app.py
```