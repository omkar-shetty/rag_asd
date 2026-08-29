from langchain_core.prompts import ChatPromptTemplate

from src.agent.state import AgentState
from src.agent.retriever import build_retriever
from src.agent.llm import llm
from src.constants import Constants
from src.agent.web_search import tavily_client

retriever = build_retriever()

def corpus_search_node(state: AgentState) -> dict:
    """Search the corpus for relevant documents and update the state."""

    docs = retriever.invoke(state["question"])

    return {"retrieved_docs": docs}

def quality_gate_node(state: AgentState) -> dict:
    """Write the routing decision based on the question and the retrieved documents."""

    context = " ".join([doc.page_content for doc in state["retrieved_docs"]])
    context = "No results found." if not context else context

    prompt = f"""
        You are a helpful assistant that evaluates the quality of retrieved documents based on a user's question.

        Question: {state["question"]}
        Context: {context}

        Please evaluate the question and the retrieved documents. If the question is unrelated to the Autism Spectrum Disorder (ASD), reply OUT_OF_SCOPE. 
        If the question is pertinent to ASD and the retrieved corpus context is sufficient to answer the question, reply SUFFICIENT.
        If the question is pertinent to ASD and specifically requires recent, current, or time-sensitive information (e.g., "latest," "recent," "this year," breaking research, current
        statistics) that a static corpus likely wouldn't have i.e. the question is about RECENCY, not about how much the retrieved context covers., reply NEEDS_WEB.
        If the question is pertinent to ASD but the corpus context doesn't fully answer the question, but the question itself isn't asking for anything recent or time-sensitive - general, timeless ASD knowledge is fine here, even if the retrieved chunks are incomplete., reply NEEDS_DIRECT.
        If a single course of action is not clear, reply NEEDS_DIRECT
        """

    response = llm.invoke(prompt)
    response_lower = response.content.lower()
    routing_decision = "needs_direct"

    keywords = ["out_of_scope", "sufficient", "needs_web", "needs_direct"]
    for keyword in keywords:
        if keyword.lower() in response_lower:
            routing_decision = keyword
            break

    return {"routing_decision": routing_decision}

def answer_from_corpus_node(state: AgentState) -> dict:
    """Generate a reply based on the available RAG context."""

    context = " ".join([doc.page_content for doc in state["retrieved_docs"]])

    prompt = ChatPromptTemplate.from_template(
    "Context: {context}\n\nQuestion: {question}\n\nAnswer concisely based on the documents. "
    "If the documents do not contain relevant information, say so."
    )

    rendered = prompt.format(context=context, question=state["question"])
    response = llm.invoke(rendered)

    return {"answer":response.content, "source":"corpus"}

def direct_answer_node(state: AgentState) -> dict:
    """Generate a reply directly without resorting to a corpus/web search."""

    prompt = ChatPromptTemplate.from_template(
        "Question: {question}\n\n Answer concisely and directly, without excessive hedging based on available information. " \
        "If the question is not relevant to Autism Spectrum Disorder (ASD), decline to answer."
    )

    rendered = prompt.format(question=state["question"])
    response = llm.invoke(rendered)

    return {"answer":response.content, "source":"direct"}

def refuse_node(state: AgentState) -> dict:
    """Generate a refusal to answer the question."""

    return {
        "answer": "I can only help with questions about Autism Spectrum Disorder (ASD). Could you rephrase your question to focus on that?",
        "source": "refused",
    }

def web_search_node(state: AgentState) -> dict:
    """Generate a reply based on a web search."""

    results = tavily_client.search(query = state["question"], 
                                   include_domains=Constants.CURATED_DOMAINS,
                                   max_results=5)
    context = " ".join(r["content"] for r in results.get("results", []))
    context = context if context else "No results found."

    prompt = ChatPromptTemplate.from_template(
        "Context from recent web search: {context}\n\nQuestion: {question}\n\n"
        "Answer concisely based on the search results. "
        "If the results do not contain relevant information, say so."
    )
    rendered = prompt.format(context=context, question=state["question"])
    response = llm.invoke(rendered)

    urls = [r["url"] for r in results.get("results", [])]

    return {
        "answer": response.content,
        "source": "web",
        "web_results": urls
    }