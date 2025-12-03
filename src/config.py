import os
from dotenv import load_dotenv

# Load .env file so HF_API_KEY is available
load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
VECTOR_DB_DIR = os.path.join(os.path.dirname(__file__), "..", "vectorstore")

# HuggingFace API
HF_API_KEY = os.getenv("HF_API_KEY")   # defined in .env
HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 4
