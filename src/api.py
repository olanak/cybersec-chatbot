from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.rag_chain import answer_question

app = FastAPI()

# CORS (local frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    use_rag: bool = True
    use_rewrite: bool = True # <- option for Query rewriting/expansion
    use_multi_query: bool = False # <- option for Multi-query retrieval

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(request: ChatRequest):
    return answer_question(
        query=request.message,
        use_rag=request.use_rag,
        use_rewrite=request.use_rewrite,
        use_multi_query=request.use_multi_query
    )
    return {
        "answer": result["answer"],
        "sources": result.get("sources", [])
    }


from fastapi.staticfiles import StaticFiles

app.mount(
    "/",
    StaticFiles(directory="frontend", html=True),
    name="frontend"
)