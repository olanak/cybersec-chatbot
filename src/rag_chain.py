from huggingface_hub import InferenceClient
from langchain_chroma import Chroma       # updated import (no deprecation warning)
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import VECTOR_DB_DIR, HF_MODEL, HF_API_KEY, EMBEDDING_MODEL, TOP_K


# -----------------------------
# Initialize Embeddings + Vector DB
# -----------------------------

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

vectordb = Chroma(
    persist_directory=VECTOR_DB_DIR,
    embedding_function=embeddings
)


# -----------------------------
# Initialize HuggingFace Client
# -----------------------------

# NOTE: Do NOT pass model here. Model is selected per request.
client = InferenceClient(api_key=HF_API_KEY)


# -----------------------------
# System Prompt
# -----------------------------

SYSTEM_PROMPT = """
You are a CYBERSECURITY DEFENSE assistant.
You ONLY provide safe, defensive, and best-practice security guidance.
You DO NOT provide offensive/hacking/exploitation instructions.

Always:
- Base your answer strictly on the provided retrieved context
- Provide clear, NIST-style defensive cybersecurity guidance
- If user asks something unsafe, politely refuse

Answer clearly and concisely.
"""


# -----------------------------
# Generate Answer from HF LLM
# -----------------------------

def generate_answer(question: str, context: str):
    prompt = f"""
User Question:
{question}

Relevant Retrieved Context:
{context}

Now provide the most accurate defensive cybersecurity answer:
"""

    # HuggingFace NEW chat completion API
    response = client.chat.completions.create(
        model=HF_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=512,
        temperature=0.2,
    )

    return response.choices[0].message["content"]


# -----------------------------
# RAG Pipeline
# -----------------------------

def answer_question(query: str):
    # Retrieve top-k relevant documents
    results = vectordb.similarity_search(query, k=TOP_K)

    # Build context from retrieved chunks
    context = "\n\n".join([doc.page_content for doc in results])
    sources = [doc.metadata.get("source", "unknown") for doc in results]

    # Generate answer using model + context
    answer = generate_answer(query, context)

    return {
        "answer": answer,
        "sources": sources
    }
