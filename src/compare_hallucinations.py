from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import numpy as np
import sys

load_dotenv()

# Questions grounded in the actual corpus content
IN_SCOPE = [
    {
        "question": "At what age can ASD typically be reliably diagnosed?",
        "reference": "ASD can usually be reliably diagnosed by the age of 2. It is important to seek an evaluation as soon as possible, as earlier diagnosis means treatments and services can begin sooner."
    },
    {
        "question": "What is echolalia?",
        "reference": "Echolalia is a behavior where a person repeats words or phrases, and is listed as one of the unusual or repetitive behaviors associated with ASD."
    },
    {
        "question": "What language preference does the autistic community have when referring to autistic people?",
        "reference": "Many autistic people prefer identity-first language, such as 'autistic person', rather than person-first language such as 'person with autism'."
    },
    {
        "question": "What does the DSM-5 say about ASD symptoms?",
        "reference": "According to the DSM-5, people with ASD often have difficulty with social communication and interaction, restricted interests and repetitive behaviors, and symptoms that affect their ability to function in school, work, and other areas of life."
    },
]

# Questions the corpus cannot answer — RAG should refuse, LLM will likely confabulate
OUT_OF_SCOPE = [
    "Who is the USA playing in the World Cup?",
    "What is the capital of Australia?",
    "Who wrote Harry Potter?",
]

RAG_PROMPT = "Context: {context}\n\nQuestion: {question}\n\nAnswer concisely based on the documents. If the documents do not contain relevant information, say so."
LLM_PROMPT = "Question: {question}\n\nAnswer concisely."


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def score_grounding(answer: str, reference: str, embeddings) -> float:
    a = np.array(embeddings.embed_query(answer))
    b = np.array(embeddings.embed_query(reference))
    return round(cosine_similarity(a, b), 4)


def rag_answer(question: str, retriever, llm) -> tuple[str, str]:
    docs = retriever.invoke(question)
    context = "\n".join(d.page_content for d in docs)
    prompt = RAG_PROMPT.format(context=context, question=question)
    return llm.invoke(prompt).content, context


def llm_answer(question: str, llm) -> str:
    prompt = LLM_PROMPT.format(question=question)
    return llm.invoke(prompt).content


def run():
    print("Loading models...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)

    print("Loading corpus...")
    docs = PyPDFDirectoryLoader("corpus/").load()
    if not docs:
        print("No documents found in corpus/")
        sys.exit(1)

    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    print("\n--- In-scope questions (grounding score vs reference) ---\n")
    for item in IN_SCOPE:
        question = item["question"]
        ref = item["reference"]

        rag = rag_answer(question, retriever, llm)[0]
        llm_only = llm_answer(question, llm)

        rag_score = score_grounding(rag, ref, embeddings)
        llm_score = score_grounding(llm_only, ref, embeddings)

        print(f"Q: {question}")
        print(f"  RAG   ({rag_score:.3f}): {rag}")
        print(f"  LLM   ({llm_score:.3f}): {llm_only}")
        print()

    print("--- Out-of-scope questions (RAG should refuse, LLM may hallucinate) ---\n")
    for question in OUT_OF_SCOPE:
        rag = rag_answer(question, retriever, llm)[0]
        llm_only = llm_answer(question, llm)

        print(f"Q: {question}")
        print(f"  RAG : {rag}")
        print(f"  LLM : {llm_only}")
        print()

    vectorstore.delete_collection()


if __name__ == "__main__":
    run()
