import streamlit as st
import os
import json
from datetime import datetime
from dotenv import load_dotenv

from src.agent.graph import compiled_graph
from src.agent.state import AgentState

load_dotenv()

st.set_page_config(page_title="ASD Corpus Assistant")
st.title("ASD Knowledge Base")

missing_keys = [k for k in ("GROQ_API_KEY", "TAVILY_API_KEY") if not os.getenv(k)]
if missing_keys:
    st.error(f"Missing required environment variable(s): {', '.join(missing_keys)}. Add them to .env (local) or Streamlit secrets (cloud).")
    st.stop()

SOURCE_LABELS = {
    "corpus": "Answered from the ASD corpus",
    "web": "Answered from a curated web search",
    "direct": "Answered from general knowledge",
    "refused": "Declined - outside ASD scope",
}

EXAMPLE_QUESTIONS = {
    "Corpus question": "What is sensory invalidation?",
    "Recency question": "What are the most recent FDA-approved treatments for ASD announced this year?",
    "General ASD question": "What are common early signs of autism in toddlers?",
    "Out-of-scope question": "What's a good recipe for banana bread?",
}

def log_interaction(query, result):
    try:
        os.makedirs("data", exist_ok=True)
        entry = {
            "timestamp": str(datetime.now()),
            "query": query,
            "answer": result["answer"],
            "source": result["source"],
            "routing_decision": result["routing_decision"],
            "retrieved_context": [doc.page_content for doc in result["retrieved_docs"]],
            "web_results": result["web_results"],
        }
        with open("data/rag_logs.jsonl", "a") as file:
            file.write(json.dumps(entry) + "\n")
    except Exception as ex:
        st.warning(f"Failed to log interaction: {ex}")

def run_query(query: str):
    st.session_state.messages.append({"role": "user", "content": query})

    try:
        initial_state: AgentState = {
            "question": query,
            "retrieved_docs": [],
            "routing_decision": None,
            "web_results": None,
            "answer": None,
            "source": None,
        }
        result = compiled_graph.invoke(initial_state)

        log_interaction(query, result)
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "source": result["source"],
            "retrieved_docs": result["retrieved_docs"],
            "web_results": result["web_results"],
        })
    except Exception as ex:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Something went wrong: {ex}",
            "source": None,
        })

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.write("Try an example, or ask your own question below:")
    cols = st.columns(2)
    for i, (label, question) in enumerate(EXAMPLE_QUESTIONS.items()):
        if cols[i % 2].button(label, use_container_width=True):
            run_query(question)

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

    if msg["role"] == "assistant" and msg.get("source"):
        st.caption(SOURCE_LABELS.get(msg["source"], ""))

        if msg["source"] == "corpus" and msg.get("retrieved_docs"):
            with st.expander("Show sources"):
                for doc in msg["retrieved_docs"]:
                    st.write(doc.page_content[:300] + "...")
                    st.divider()

        elif msg["source"] == "web" and msg.get("web_results"):
            with st.expander("Show sources"):
                for url in msg["web_results"]:
                    st.write(url)

        elif msg["source"] == "direct":
            with st.expander("Show sources"):
                st.write("This answer used the model's general knowledge - no specific documents or web pages were retrieved.")

if query := st.chat_input("Ask a question about Autism Spectrum Disorder..."):
    run_query(query)
    st.rerun()
