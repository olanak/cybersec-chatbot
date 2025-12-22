from huggingface_hub import InferenceClient
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import VECTOR_DB_DIR, HF_MODEL, HF_API_KEY, EMBEDDING_MODEL, TOP_K
from src.evaluation import faithfulness_score

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
You are a defensive cybersecurity assistant.
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

def answer_question(query: str, use_rag: bool = True, use_rewrite: bool = True, use_multi_query: bool = True):

    small_talk = ["hi", "hello", "hey", "good morning", "good evening"]

    if query.lower().strip() in small_talk:
        return {
            "answer": "Hi! I'm your cybersecurity assistant. Ask me a question about the documents.",
            "sources": []
        }
    
    # Query rewriting
    if use_rewrite:
        query = rewrite_query(query)

    # RAG OFF
    if not use_rag:
        return answer_without_rag(query)

    # Multi-query retrieval
    if use_multi_query:
        queries = generate_multi_queries_llm(query)
        print("Multi-query variants:")
        for q in queries:
            print("  -", q)
    else:
        queries = [query]   

    # Retrieve documents for all queries
    all_results = []
    for q in queries:
        results = vectordb.similarity_search_with_score(q, k=TOP_K)
        all_results.extend(results)

    # Deduplicate documents (by content)
    unique_docs = {}
    for doc, score in all_results:
        if score <= RELEVANCE_THRESHOLD:
            unique_docs[doc.page_content] = doc

    
    relevant_docs = list(unique_docs.values())

    if not relevant_docs:
        return {
            "answer": "I could not find this information in the provided documents.",
            "sources": []
        }

    print(f"Retrieved {len(relevant_docs)} unique documents")

    context = "\n\n".join(doc.page_content for doc in relevant_docs)
    sources = list(dict.fromkeys(
        doc.metadata.get("source", "unknown") for doc in relevant_docs
    ))

    answer = generate_answer(query, context)
    # faith_score = faithfulness_score(answer, [doc.page_content for doc in relevant_docs])

    return {
        "answer": answer,
        "sources": sources,
       # "faithfulness": round(faith_score, 3)
    }

# -----------------------------
# Query rewriting/expansion
# -----------------------------

def rewrite_query(original_query: str) -> str:
    """
    Expand/clarify the original query using the LLM.
    """
    prompt = f"""
    Rewrite or expand the following question to make it clearer and include related keywords.
    Original question: {original_query}
    """
    response = client.chat.completions.create(
        model=HF_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        max_tokens=128,
        temperature=0.0,
    )

    expanded_query = response.choices[0].message["content"]
    return expanded_query.strip()

# -----------------------------
# Multi-query retrieval
# -----------------------------

def generate_multi_queries_llm(query: str, n: int = 4):
    """
    Generate multiple semantically different queries using LLM
    """

    prompt = f"""
Generate {n} different search queries that express the same intent as the original question.
Make them suitable for document retrieval.

Original question:
{query}

Return each query on a new line.
"""

    response = client.chat.completions.create(
        model=HF_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.7
    )

    queries = response.choices[0].message["content"].split("\n")

    return [q.strip("- ").strip() for q in queries if q.strip()]
