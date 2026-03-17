from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from logger_config import setup_script_logger
import statistics
import time
import sys

load_dotenv()
logger = setup_script_logger("compare_chunking")

def get_standard_chunks(docs):
    """Generate chunks using RecursiveCharacterTextSplitter"""
    start_time = time.time()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    elapsed_time = time.time() - start_time
    return chunks, elapsed_time


def get_parent_child_chunks(docs):
    """Generate chunks using parent-child strategy"""
    start_time = time.time()
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    chunks = child_splitter.split_documents(docs)
    elapsed_time = time.time() - start_time
    return chunks, elapsed_time


def get_semantic_chunks(docs):
    """Generate chunks using SemanticChunker"""
    start_time = time.time()
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    splitter = SemanticChunker(embeddings=embeddings, breakpoint_threshold_type="percentile")
    chunks = splitter.split_documents(docs)
    elapsed_time = time.time() - start_time
    return chunks, elapsed_time


def calculate_chunk_stats(chunks):
    """Calculate statistics about chunks"""
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

def benchmark_retrieval_latency(chunks, query: str = "What are the symptoms of ASD?", k: int = 5, runs: int = 5):
    """Build a temporary in-memory vector store and measure retrieval latency."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Build a temporary in-memory Chroma store (no persist_directory)
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    
    latencies = []
    for _ in range(runs):
        start = time.time()
        retriever.invoke(query)
        latencies.append(time.time() - start)
    
    vectorstore.delete_collection()  # cleanup
    
    return {
        "avg_latency_ms":   statistics.mean(latencies) * 1000,
        "min_latency_ms":   min(latencies) * 1000,
        "max_latency_ms":   max(latencies) * 1000,
        "stdev_latency_ms": statistics.stdev(latencies) * 1000 if len(latencies) > 1 else 0,
    }

def compare_chunking_strategies(docs):
    print("\n" + "="*80)
    print("CHUNKING STRATEGY COMPARISON")
    print("="*80 + "\n")
    
    strategies = {
        "Standard (RecursiveCharacter)": get_standard_chunks,
        "Parent-Child (Small chunks)": get_parent_child_chunks,
        "Semantic": get_semantic_chunks
    }
    
    results = {}
    
    for strategy_name, strategy_func in strategies.items():
        logger.info(f"Processing {strategy_name}...")
        try:
            chunks, elapsed_time = strategy_func(docs)
            stats = calculate_chunk_stats(chunks)

            logger.info(f"Benchmarking retrieval latency for {strategy_name}...")
            latency = benchmark_retrieval_latency(chunks)

            results[strategy_name] = {
                "stats": stats,
                "time": elapsed_time,
                "latency": latency
            }
            logger.info(f"Completed in {elapsed_time:.2f}s")
        except Exception as e:
            logger.error(f"Error: {str(e)}", exc_info=True)
            results[strategy_name] = None
    
    logger.info("\n" + "="*80)
    logger.info("DETAILED METRICS")
    logger.info("="*80 + "\n")
    
    logger.info(f"{'Metric':<25} {'Standard':<20} {'Parent-Child':<20} {'Semantic':<20}")
    logger.info("-" * 85)
    
    metrics = ["count", "total_chars", "avg_size", "median_size", "min_size", "max_size", "stdev"]
    
    for metric in metrics:
        metric_label = metric.replace("_", " ").title()
        row_data = [metric_label]
        
        for strategy_name in strategies.keys():
            if results[strategy_name]:
                value = results[strategy_name]["stats"][metric]
                if isinstance(value, float):
                    row_data.append(f"{value:.1f}")
                else:
                    row_data.append(str(value))
            else:
                row_data.append("ERROR")
        
        logger.info(f"{row_data[0]:<25} {row_data[1]:>20} {row_data[2]:>20} {row_data[3]:>20}")
    
    logger.info("\n" + "="*80)
    logger.info("PROCESSING TIME")
    logger.info("="*80 + "\n")
    
    for strategy_name in strategies.keys():
        if results[strategy_name]:
            time_val = results[strategy_name]["time"]
            logger.info(f"{strategy_name:<30} {time_val:.2f}s")

    
    logger.info("\n" + "="*80)
    logger.info("RETRIEVAL LATENCY (avg over 5 runs, ms)")
    logger.info("="*80 + "\n")

    logger.info(f"{'Strategy':<35} {'Avg':>10} {'Min':>10} {'Max':>10} {'StDev':>10}")
    logger.info("-" * 75)

    for strategy_name in strategies.keys():
        if results[strategy_name] and "latency" in results[strategy_name]:
            l = results[strategy_name]["latency"]
            logger.info(
                f"{strategy_name:<35} {l['avg_latency_ms']:>10.1f} "
                f"{l['min_latency_ms']:>10.1f} {l['max_latency_ms']:>10.1f} "
                f"{l['stdev_latency_ms']:>10.1f}"
            )


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

