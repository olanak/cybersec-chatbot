from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.rag_chain import answer_question


app = FastAPI(
    title="Cybersecurity RAG Chatbot",
    description="Defensive cybersecurity assistant using RAG + Qwen/Qwen2.5-7B-Instruct",
    version="0.1.0",
)

# Allow frontend (e.g., localhost) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for development; you can restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Main chat endpoint.
    Takes a user message, runs RAG pipeline, returns answer + sources.
    """
    result = answer_question(request.message)

    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
    )

@app.get("/health")
def health():
    return {"status": "ok"}
