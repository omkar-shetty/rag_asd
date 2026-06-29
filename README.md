# ASD RAG: Clinical Evidence-Based Knowledge System


---

## Executive Summary
The **ASD RAG** (Autism Spectrum Disorder - Retrieval-Augmented Generation) system is a specialized AI assistant designed to navigate the existing literature and information available around ASD. Developed to solve the "hallucination problem" in medical AI, this system uses a retrieval pipeline to ensure responses are mathematically grounded in verified clinical documents.



---

## Technical Architecture

### 1. Data Engineering Pipeline (`src/build_db.py`)
* **Ingestion:** Batch processing of the clinical corpus via `PyPDFDirectoryLoader`.
* **Chunking Strategies:** Offers the option to choose from three approaches:
    1. **Standard** (`--splitter standard`): `RecursiveCharacterTextSplitter` with `chunk_size=1000`, `overlap=200`. Baseline for simple retrieval.
    2. **Parent-Child** (`--splitter parent-child`): Hierarchical splitting with small searchable chunks (`size=200`) retrieval but returns larger parent contexts (`size=1000`). Ensures a balance between search precision and context quality.
    3. **Semantic** (`--splitter semantic`): `SemanticChunker` that intelligently splits at meaningful boundaries using embedding similarity. Best coherence but slower processing.
* **Vectorization:** `HuggingFaceEmbeddings` using the `all-MiniLM-L6-v2` transformer (384-dimensional).
* **Storage:** Local persistence via **ChromaDB**, ensuring data privacy and zero-latency infrastructure costs.

### 2. Evaluation Pipeline (`src/compare_chunking.py`)
* **Performance Metrics:** Compares all three chunking strategies across:
    - Chunk statistics (count, size distribution, uniformity)
    - Chunking time
    - Retrieval latency
    - Faithfulness scores (cosine similarity between retrieved chunks and reference answers)
* **Hallucination Comparison** (`src/compare_hallucinations.py`): Runs the same questions through RAG and the bare LLM, scoring each against a reference to quantify grounding.

### 3. Inference & UI Engine (`src/app.py`)
* **Orchestration:** Built on **LangChain Expression Language (LCEL)** for a modular, composable chain: `retriever | prompt | llm`.
* **Intelligence:** **Llama-3.1-8b-instant** (via Groq) for sub-second inference speeds.
* **Configuration:** `Temperature=0` for deterministic, fact-based responses and `k=5` for optimal context window balance.
* **Interface:** A **Streamlit** chat UI.

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
Place your PDFs in the `corpus/` directory and choose a chunking strategy:

```bash
# Standard (fastest, recommended for baseline)
python src/build_db.py --splitter standard

# Parent-Child (balance of precision + context)
python src/build_db.py --splitter parent-child

# Semantic (most coherent, slower)
python src/build_db.py --splitter semantic
```

This creates a vector database in `vector_db_<strategy>/`.

### 2. Compare Chunking Strategies (Optional)
Benchmark all three strategies on your corpus:

```bash
python src/compare_chunking.py
```

To compare RAG vs LLM hallucination behaviour:

```bash
python src/compare_hallucinations.py
```

### 3. Launch the UI
Start the Streamlit interface to interact with the assistant:
```bash
streamlit run src/app.py
```

---

## Project Structure

```
rag_asd/
├── corpus/                          # Input PDFs for indexing (excluded from repo)
├── src/
│   ├── build_db.py                  # Index corpus with chosen chunking strategy
│   ├── compare_chunking.py          # Benchmark chunking strategies
│   ├── compare_hallucinations.py    # RAG vs LLM hallucination comparison
│   ├── logger_config.py             # Shared logging setup
│   └── app.py                       # Streamlit UI
├── vector_db_parent-child/          # Pre-built vector store (committed to repo)
├── parent_store/                    # Pre-built parent documents (committed to repo)
├── data/
│   └── rag_logs.jsonl              # Query logs from the UI
├── logs/                            # Script logs
├── .streamlit/
│   └── secrets.toml.example        # API key template for local setup
├── .env                             # API keys (local only, gitignored)
├── requirements.txt
└── README.md
```

---

## Environment Configuration

Create or update `.env` file with your API keys:

```bash
# Required for Groq LLM inference
GROQ_API_KEY=your_groq_key_here

# Optional: For LLM-based faithfulness evaluation
OPENAI_API_KEY=your_openai_key_here

# Optional: For HuggingFace Hub LLM models
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
```

Keys are automatically loaded via `python-dotenv` when scripts run.

---

## Chunking Strategy Comparison

Benchmarked on the ASD corpus (348 documents) across 3 domain-specific queries:

| Strategy     | Chunks | Avg Size  | Chunking Time | Retrieval Latency | Faithfulness (max / mean) |
| :----------- | -----: | --------: | ------------: | ----------------: | ------------------------: |
| Standard     | 735    | 749 chars | 0.04s         | 31.8ms            | 0.717 / 0.650             |
| Parent-Child | 3102   | 156 chars | 0.09s         | 23.4ms            | 0.745 / 0.690             |
| Semantic     | 713    | 680 chars | 106.4s        | 31.3ms            | 0.697 / 0.600             |

**Parent-Child is the recommended strategy.** Smaller 200-char search targets improve mean faithfulness by ~6% over standard and retrieval latency is the lowest of the three strategies. The cost is a larger index. Semantic chunking underperformed on this corpus.

Run `python src/compare_chunking.py` to reproduce on your own corpus.

---

## Hallucination Comparison: RAG vs LLM

Tested across 4 in-scope and 3 out-of-scope questions. The RAG scored higher on grounding in 3 of 4 in-scope questions. More critically, it correctly refused all 3 out-of-scope questions while the LLM answered two as fact. In a clinical context, refusing to answer outside the corpus is safer than confabulating from general knowledge.

Run `python src/compare_hallucinations.py` to reproduce on your own corpus.

---

## Logging

Each script writes INFO-level logs to console and to a file in `logs/`:

- `logs/build_db.log`
- `logs/compare_chunking.log`

---

## Troubleshooting

### ChromaDB Persistence Issues
Vector stores are persisted in `vector_db_*/` directories. Delete them to trigger re-indexing:
```bash
rm -r vector_db_*  # On Windows (PowerShell): Remove-Item -Recurse vector_db_*
```

Check `logs/build_db.log` to verify successful re-indexing.