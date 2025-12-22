from huggingface_hub import InferenceClient
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import VECTOR_DB_DIR, HF_MODEL, HF_API_KEY, EMBEDDING_MODEL, TOP_K

from src.log_utils import log_rag_example

# -----------------------------
# Vector DB + Embeddings
# -----------------------------

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

vectordb = Chroma(
    persist_directory=VECTOR_DB_DIR,
    embedding_function=embeddings
)

# -----------------------------
# HuggingFace Client
# -----------------------------

client = InferenceClient(api_key=HF_API_KEY)

# -----------------------------
# System Prompt
# -----------------------------

SYSTEM_PROMPT = """
You must answer ONLY using the provided context when available.
If the answer is not in the documents, say you do not know.
Do NOT hallucinate.
"""

# -----------------------------
# LLM helpers
# -----------------------------

def generate_answer(question: str, context: str):
    response = client.chat.completions.create(
        model=HF_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"""
Question:
{question}

Context:
{context}

Answer the question using ONLY the context above.
"""
            }
        ],
        max_tokens=512,
        temperature=0.2,
    )

    return response.choices[0].message["content"]


def answer_without_rag(question: str):
    response = client.chat.completions.create(
        model=HF_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ],
        max_tokens=512,
        temperature=0.2,
    )

    return {
        "answer": response.choices[0].message["content"],
        "sources": []
    }

# -----------------------------
# RAG pipeline (with toggle)
# -----------------------------

RELEVANCE_THRESHOLD = 0.75  # calibrated for mixed PDF/DOCX/XLSX data

def answer_question(query: str, use_rag: bool = True):

    # 🔹 RAG OFF
    if not use_rag:
        return answer_without_rag(query)

    # 🔹 RAG ON
    results = vectordb.similarity_search_with_score(query, k=TOP_K)

    relevant_docs = [
        doc for doc, score in results
        if score <= RELEVANCE_THRESHOLD
    ]

    # ❌ Nothing relevant found
    if not relevant_docs:
        return {
            "answer": (
                "I could not find this information in the provided documents. "
                "Please ask something covered by the dataset."
            ),
            "sources": []
        }

    context = "\n\n".join(doc.page_content for doc in relevant_docs)

    # de-duplicate sources
    sources = list(dict.fromkeys(
        doc.metadata.get("source", "unknown") for doc in relevant_docs
    ))

    answer = generate_answer(query, context)
    
    # LOG REAL CHATBOT ANSWER + CONTEXTS
    log_rag_example(
        question=query,
        answer=answer,
        contexts=[doc.page_content for doc in relevant_docs],
    )
    

    return {
        "answer": answer,
        "sources": sources
    }
