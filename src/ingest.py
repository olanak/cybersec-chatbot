import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader
)
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import DATA_DIR, VECTOR_DB_DIR, EMBEDDING_MODEL
from langchain_community.vectorstores.utils import filter_complex_metadata



def load_documents():
    """
    Load documents from DATA_DIR.
    Supported formats:
    - PDF (.pdf)
    - Word (.docx)
    - Excel (.xlsx)
    - Text / Markdown (.txt, .md)
    """
    documents = []

    for filename in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, filename)
        fname = filename.lower()

        try:
            if fname.endswith(".pdf"):
                loader = PyPDFLoader(path)

            elif fname.endswith(".docx"):
                loader = Docx2txtLoader(path)

            elif fname.endswith(".xlsx"):
                loader = UnstructuredExcelLoader(
                    path,
                    mode="elements"  # preserves rows/cells better
                )

            elif fname.endswith(".txt") or fname.endswith(".md"):
                loader = TextLoader(path)

            else:
                print(f"Skipping unsupported file: {filename}")
                continue

            docs = loader.load()

            # Attach filename as source metadata
            for doc in docs:
                doc.metadata["source"] = filename

            documents.extend(docs)
            print(f"Loaded: {filename}")

        except Exception as e:
            print(f"Error loading {filename}: {e}")

    return documents


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    #CRITICAL FIX: remove complex metadata (lists, dicts, etc.)
    chunks = filter_complex_metadata(chunks)

    return chunks



def build_vector_db(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR
    )

    print(" Vector store created and persisted successfully!")



if __name__ == "__main__":
    print("Loading documents...")
    docs = load_documents()

    print(f"Loaded {len(docs)} documents. Chunking...")
    chunks = chunk_documents(docs)

    print(f"Created {len(chunks)} chunks. Building vector DB...")
    build_vector_db(chunks)
