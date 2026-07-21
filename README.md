# ASD RAG

A RAG (Retrieval Augmented Generation) system for ASD (Autism Spectrum Disorder) clinical and caregiver literature, built to answer only from source material. Three chunking strategies and three faithfulness evaluation methods are benchmarked. Built on LangChain, ChromaDB, and Llama 3.1 via Groq with a Streamlit UI.

```mermaid
flowchart TB
    subgraph ingestion [Ingestion]
        direction LR
        pdfs["corpus <br/>348 PDF documents"] --> build["build_db.py<br/>3 chunking strategies"] --> store["ChromaDB store<br/>parent-child, persisted"]
    end
 
    subgraph serving [Serving]
        direction LR
        app["app.py<br/>Streamlit chat UI"] --> llm["Llama 3.1 (Groq)<br/>temperature = 0"] --> logs["rag_logs.jsonl<br/>logs queries & answers"]
    end
 
    subgraph evaluation [Evaluation]
        direction LR
        chunking["compare_chunking.py<br/>chunk vs. reference<br/>cosine similarity"]
        halluc["compare_hallucinations.py<br/>answer vs. reference<br/>RAG vs. bare LLM"]
        judge["evaluate_rag.py<br/> LLM judge"]
    end
 
    store --> app
    logs --> chunking
    logs --> halluc
    logs --> judge
```

## 1. Purpose

Decisions around chunking strategy, embedding model, and prompt design shape how a RAG system behaves, but their effects are rarely verified. This project benchmarks and compares these decisions: 
* Three chunking strategies are compared on retrieval quality and latency
* Three separate evaluation methods (retrieval-level cosine similarity, generation-level grounding comparison, and LLM-judged faithfulness) are used to understand what "the answer is faithful to the source" actually means at different levels of the pipeline.


## 2. How it works

**Ingestion:** `build_db.py` loads PDFs from the `corpus/` folder and builds a persisted ChromaDB vector store using one of the three supported chunking strategies, selected via the `--splitter` argument:
* **Standard** (`--splitter standard`): `RecursiveCharacterTextSplitter` with `chunk_size=1000`, `overlap=200`. Baseline for simple retrieval.
* **Parent-Child** (`--splitter parent-child`): Hierarchical splitting with small searchable chunks (`size=200`) retrieval but returns larger parent contexts (`size=1000`). Ensures a balance between search precision and context quality.
* **Semantic** (`--splitter semantic`): `SemanticChunker` that intelligently splits at meaningful boundaries using embedding similarity. Best coherence but slower processing.
* **Vectorization:** `HuggingFaceEmbeddings` using the `all-MiniLM-L6-v2` transformer (384-dimensional).
* **Storage:** Local persistence via **ChromaDB**.

**Retrieval and generation:** `app.py` loads the parent-child vector store (the strategy the chunking benchmark below identifies as strongest) and serves a Streamlit chat UI. Retrieval uses `ParentDocumentRetriever`; generation uses `llama-3.1-8b-instant` via Groq at `temperature=0`, prompted to answer only from retrieved context and to say so explicitly when it cannot.

**Evaluation:** Three evaluation methods are implemented to test different features of the pipeline.

| Method | Script | Summary |
|---|---|---|
| Chunk-retrieval similarity | `compare_chunking.py` | Cosine similarity between a hand-written reference answer and the *retrieved chunks*. Used for evaluating the best chunking strategy for this application. |
| Grounding comparison | `compare_hallucinations.py` | Cosine similarity between the *generated answer* and a reference, comparing the full RAG pipeline against a bare LLM with no retrieval |
| LLM-judged faithfulness | `evaluate_rag.py` | DeepEval's `FaithfulnessMetric`, using a separate LLM (`llama-3.3-70b-versatile`) as judge to check whether each claim in a generated answer is supported by, contradicted by, or unverifiable from the retrieved context |

The LLM judged faithfulness is the most rigorous test here. It decomposes the response into individual claims, checking each one against the source material. This is also why a stronger model is used as a judge here compared to the faster model for production answering.


## 3. Current Capabilities
 
* Three chunking strategies (standard, parent-child, semantic), independently benchmarked.
* Persisted ChromaDB vector stores per strategy, built via a CLI (`build_db.py --splitter <strategy>`).
* Streamlit chat UI.
* Three independent evaluation methods, each targeting a different part of the pipeline.
* LLM-as-judge faithfulness evaluation with structured JSON logging and console summary stats (pass rate, verdict distribution).
* Consistent logging (`logger_config.py`) across all scripts.


## 4. Project Structure
 
```
rag_asd/
├── src/
│   ├── app.py                          # Streamlit chat UI
│   ├── logger_config.py                # common logging setup
│   ├── ingestion/
│   │   └── build_db.py                 # corpus loading + chunking + vector store build
│   └── evaluation/
│       ├── compare_chunking.py         # chunking strategy benchmark
│       ├── compare_hallucinations.py   # RAG vs. bare LLM grounding comparison
│       └── evaluate_rag.py             # LLM-as-judge faithfulness evaluation (DeepEval)
├── corpus/                             # input PDFs
├── vector_db_parent-child/             # persisted vector store
├── parent_store/                       # persisted parent documents for ParentDocumentRetriever
├── data/
│   └── rag_logs.jsonl                  # logged queries and answers, feeds evaluate_rag.py
├── logs/                               # script logs + evaluate_rag.py JSON output
├── .streamlit/
│   └── secrets.toml.example            # sample template for .streamlit/secrets.toml
├── requirements.txt
└── README.md
```

## 5. Setup and Usage
 
```bash
pip install -r requirements.txt
```
 
Add a `.env` file with `GROQ_API_KEY=your_key_here`.
 
**Build the vector store.** 

Place your PDFs in the `corpus/` directory and choose a chunking strategy. Currently, the app depends on the parent-child store specifically:
 
```bash
python -m src.ingestion.build_db --splitter parent-child
```
Alternate Options:

```bash
# Standard (fastest, recommended for baseline)
python -m src.ingestion.build_db --splitter standard

# Semantic (slower)
python -m src.ingestion.build_db --splitter semantic
```

**Launch the app:**
 
```bash
streamlit run src/app.py
```
 
**Reproduce the chunking benchmark** (builds and compares all three strategies from scratch):
 
```bash
python -m src.evaluation.compare_chunking
```
 
**Compare RAG vs. bare-LLM grounding** on in-scope and out-of-scope questions:
 
```bash
python -m src.evaluation.compare_hallucinations
```
 
**Run LLM-judged faithfulness evaluation** against logged queries (requires having used the app first, so `data/rag_logs.jsonl` has entries):
 
```bash
python -m src.evaluation.evaluate_rag
```

## 6. Results
 
### Chunking strategy comparison
 
Benchmarked on the ASD corpus (348 documents) across 3 domain-specific queries:
 
| Strategy | Chunks | Avg size | Chunking time | Retrieval latency | Faithfulness (max / mean) |
|---|---:|---:|---:|---:|---:|
| Standard | 735 | 749 chars | 0.05s | 31.9ms | 0.717 / 0.650 |
| Parent-child | 3102 | 156 chars | 0.11s | 23.4ms | **0.746 / 0.690** |
| Semantic | 713 | 680 chars | 94.4s | 37.6ms | 0.697 / 0.600 |
 
Parent-child chunking comes across better on both faithfulness measures and has the lowest retrieval latency, at the cost of a larger index (3,102 vs. 735 chunks). Semantic chunking's expected coherence advantage didn't translate into better retrieval on this corpus, and its chunking time is significantly slower than the alternatives. This is the basis for `app.py` using parent-child chunking in production.
 
### LLM-judged faithfulness
 
Evaluated 6 logged queries with `FaithfulnessMetric` (DeepEval), judged by `llama-3.3-70b-versatile`:
 
```
Pass rate: 100% (6/6 passed threshold 0.7)
Perfect scores (1.0): 6/6
Mean claims per case: 7.2
Verdicts across all cases: 42% yes, 58% idk, 0% no
```
 
**Key Findings**
* Zero claims were flagged as contradicting the retrieved context - across every logged query, the system never asserted something the source material disputed. 
* Only 42% of individual claims were assigned a clean "yes" (directly traceable to specific retrieved text); the remaining 58% were "idk" - not contradicted, but not strictly confirmable either, possibly because the model synthesized or lightly elaborated on what was retrieved rather than quoting it directly.

### Hallucination Comparison: RAG vs LLM

Tested across 4 in-scope and 3 out-of-scope questions. The RAG scored higher on grounding in 3 of 4 in-scope questions. More critically, it correctly refused all 3 out-of-scope questions while the LLM answered two as fact. In a clinical context, refusing to answer outside the corpus is safer than confabulating from general knowledge.

## 7. Environment Configuration

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

## 8. Known Limitations
 
* **Smaller Evaluation sets:** All benchmark tests use a small set of queries - Currently these are directionally indicative rather than statistically robust.
* **Cosine-similarity-based faithfulness** (chunking and hallucination comparisons) is a proxy, not a direct faithfulness check.
* **The Chroma import used throughout (`langchain_community.vectorstores.Chroma`)** is deprecated in favor of the `langchain_chroma` package. A migration is planned.


## 9. Roadmap
 
* Migrate to `langchain_chroma.Chroma`.
* Expand evaluation sets for more statistically meaningful benchmark numbers.

## License
MIT