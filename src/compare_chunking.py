from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from logger_config import setup_script_logger
import numpy as np
import statistics
import time
import sys

load_dotenv()
logger = setup_script_logger("compare_chunking")

#Set of questions to evaluate faithfulness of retrieved chunks.
EVAL_SET = [
    {
        "query": "What are the symptoms of ASD?",
        "reference": "ASD symptoms include social communication difficulties, repetitive behaviors, restricted interests, and sensory sensitivities."
    },
    {
        "query": "What interventions are effective for ASD?",
        "reference": "Evidence-based interventions for ASD include Applied Behavior Analysis, speech therapy, occupational therapy, and social skills training."
    },
    {
        "query": "How is ASD diagnosed?",
        "reference": "ASD is diagnosed through clinical observation, developmental history, and standardized tools such as the ADOS and ADI-R."
    },
]

def get_standard_chunks(docs, **kwargs):
    """Generate chunks using RecursiveCharacterTextSplitter"""
    start_time = time.time()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    elapsed_time = time.time() - start_time
    return chunks, elapsed_time


def get_parent_child_chunks(docs, **kwargs):
    """Generate chunks using parent-child strategy"""
    start_time = time.time()
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    chunks = child_splitter.split_documents(docs)
    elapsed_time = time.time() - start_time
    return chunks, elapsed_time


def get_semantic_chunks(docs, embeddings=None):
    """Generate chunks using SemanticChunker"""
    start_time = time.time()
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    splitter = SemanticChunker(embeddings=embeddings, breakpoint_threshold_type="percentile")
    chunks = splitter.split_documents(docs)
    elapsed_time = time.time() - start_time
    return chunks, elapsed_time


def calculate_chunk_stats(chunks):
    """Calculate statistics about chunks"""
    if not chunks:
        raise ValueError("No chunks provided")
    chunk_sizes = [len(chunk.page_content) for chunk in chunks]
    total_chars = sum(chunk_sizes)
    
    stats = {
        "count": len(chunks),
        "total_chars": total_chars,
        "avg_size": statistics.mean(chunk_sizes),
        "median_size": statistics.median(chunk_sizes),
        "min_size": min(chunk_sizes),
        "max_size": max(chunk_sizes),
        "stdev": statistics.stdev(chunk_sizes) if len(chunk_sizes) > 1 else 0
    }
    return stats

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D vectors."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

def benchmark_retrieval_latency(chunks,
                                query: str = EVAL_SET[0]["query"],
                                k: int = 5,
                                runs: int = 5,
                                embeddings=None,
                                vectorstore=None) -> dict:
    """Measure retrieval latency against an in-memory vector store."""
    if not embeddings:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    local = vectorstore is None
    if local:
        vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})

        latencies = []
        for _ in range(runs):
            start = time.time()
            retriever.invoke(query)
            latencies.append(time.time() - start)
    finally:
        if local:
            vectorstore.delete_collection()
    
    return {
        "avg_latency_ms":   statistics.mean(latencies) * 1000,
        "min_latency_ms":   min(latencies) * 1000,
        "max_latency_ms":   max(latencies) * 1000,
        "stdev_latency_ms": statistics.stdev(latencies) * 1000 if len(latencies) > 1 else 0,
    }

def score_faithfulness(chunks, eval_set: list = EVAL_SET, k: int = 5, embeddings=None, vectorstore=None) -> dict:
    """Measures cosine similarity between reference answers and retrieved chunks.
    """
    if not embeddings:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    local = vectorstore is None #check if local vectorstore is needed.
    if local:
        vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    max_sims = []
    mean_sims = []

    try:
        for item in eval_set:
            query     = item["query"]
            reference = item["reference"]
    
            retrieved_docs = retriever.invoke(query)
            if not retrieved_docs:
                logger.warning(f"No chunks retrieved for query: '{query}'")
                continue
    
            # Embed the reference answer and all retrieved chunks in one batch
            ref_vec    = np.array(embeddings.embed_query(reference))
            chunk_vecs = np.array(embeddings.embed_documents(
                [doc.page_content for doc in retrieved_docs]
            ))
    
            sims = [cosine_similarity(ref_vec, cv) for cv in chunk_vecs]
            max_sims.append(max(sims))
            mean_sims.append(statistics.mean(sims))
    finally:
        if local:
            vectorstore.delete_collection()

    if not max_sims:
        return {"max_faithfulness": 0.0, "mean_faithfulness": 0.0}
 
    return {
        # avg of the best-matching chunk score per query
        "max_faithfulness":  round(statistics.mean(max_sims), 4),
        # avg of all retrieved chunk scores per query
        "mean_faithfulness": round(statistics.mean(mean_sims), 4),
    }

def compare_chunking_strategies(docs):
    """Compare different chunking strategies and log detailed metrics."""

    strategies = {
        "Standard (RecursiveCharacter)": get_standard_chunks,
        "Parent-Child (Small chunks)": get_parent_child_chunks,
        "Semantic": get_semantic_chunks
    }
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    results = {}
    
    for strategy_name, strategy_func in strategies.items():
        logger.info(f"Processing {strategy_name}")
        try:
            chunks, elapsed_time = strategy_func(docs, embeddings=embeddings)
            vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
            stats = calculate_chunk_stats(chunks)

            logger.info(f"Benchmarking retrieval latency for {strategy_name}")
            latency = benchmark_retrieval_latency(chunks, embeddings=embeddings, vectorstore=vectorstore)

            logger.info(f"Scoring faithfulness for {strategy_name}")
            faithfulness = score_faithfulness(chunks, embeddings=embeddings, vectorstore=vectorstore)

            results[strategy_name] = {
                "stats": stats,
                "time": elapsed_time,
                "latency": latency,
                "faithfulness": faithfulness
            }
            logger.info(f"Completed in {elapsed_time:.2f}s")
        except Exception as e:
            logger.error(f"Error: {str(e)}", exc_info=True)
            results[strategy_name] = None
        finally:
            if vectorstore:
                vectorstore.delete_collection()
                vectorstore = None
    
    logger.info("--- Results ---")
    for strategy_name, result in results.items():
        if result is None:
            logger.info(f"{strategy_name}: failed")
            continue

        s = result["stats"]
        lat = result["latency"]
        faith = result["faithfulness"]

        logger.info(f"{strategy_name}")
        logger.info(f"  chunks: {s['count']}, avg size: {s['avg_size']:.0f} chars (stdev {s['stdev']:.0f}), range {s['min_size']}–{s['max_size']}")
        logger.info(f"  chunking time: {result['time']:.2f}s")
        logger.info(f"  retrieval latency: avg {lat['avg_latency_ms']:.1f}ms, min {lat['min_latency_ms']:.1f}ms, max {lat['max_latency_ms']:.1f}ms")
        logger.info(f"  faithfulness: max {faith['max_faithfulness']:.4f}, mean {faith['mean_faithfulness']:.4f}")


def main():
    try:
        logger.info("Loading PDFs from corpus...")
        loader = PyPDFDirectoryLoader("corpus/")
        docs = loader.load()
        logger.info(f"Loaded {len(docs)} documents.")
        
        if not docs:
            logger.error("No documents found in corpus/")
            sys.exit(1)
        
        compare_chunking_strategies(docs)
        logger.info("Comparison completed successfully.")
        
    except FileNotFoundError as e:
        logger.error(f"Corpus directory not found: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during comparison: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
