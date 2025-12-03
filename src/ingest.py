import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.config import DATA_DIR, VECTOR_DB_DIR, EMBEDDING_MODEL




def load_documents():
    docs = []
    for filename in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, filename)

        if filename.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif filename.endswith(".txt") or filename.endswith(".md"):
            loader = TextLoader(path)
        else:
            print(f"Skipping unsupported file: {filename}")
            continue

        docs.extend(loader.load())

    return docs

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    return splitter.split_documents(docs)

def build_vector_db(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR,
    )

    vectordb.persist()
    print("Vector store created successfully!")

if __name__ == "__main__":
    print("Loading documents...")
    docs = load_documents()

    print(f"Loaded {len(docs)} documents. Chunking...")
    chunks = chunk_documents(docs)

    print(f"Created {len(chunks)} chunks. Building vector DB...")
    build_vector_db(chunks)
