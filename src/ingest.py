import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
    CSVLoader, 
    UnstructuredImageLoader
)
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import DATA_DIR, VECTOR_DB_DIR, EMBEDDING_MODEL
from langchain_community.vectorstores.utils import filter_complex_metadata
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_core.documents import Document




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
                
            elif fname.endswith(".csv"):
                loader = CSVLoader(path)
                
            elif fname.endswith(".png") or fname.endswith(".jpg") or fname.endswith(".jpeg"):
                loader = UnstructuredImageLoader(path)
            

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

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_core.documents import Document


def load_youtube_videos():
    """Load transcripts from YouTube videos using youtube-transcript-api .fetch()."""
    youtube_urls = [
        "https://www.youtube.com/watch?v=tcqEUSNCn8I",
        # add more URLs here
    ]

    yt_docs = []

    for url in youtube_urls:
        try:
            video_id = url.split("v=")[-1].split("&")[0]

            # create API instance and fetch transcript
            api = YouTubeTranscriptApi()
            transcript = api.fetch(video_id, languages=["en"])   # NEW: .fetch()

            # transcript.snippets is a list of objects with .text
            text = " ".join(snippet.text for snippet in transcript.snippets)

            doc = Document(
                page_content=text,
                metadata={
                    "source": url,
                    "video_id": video_id,
                    "type": "youtube",
                    "language": transcript.language,
                },
            )
            yt_docs.append(doc)

            print(f"\n Loaded YouTube transcript: {url}")
            print("   Language:", transcript.language)
            print("   First 200 chars:\n", text[:200], "...\n")

        except TranscriptsDisabled:
            print(f"\n Transcript disabled for: {url}")
        except Exception as e:
            print(f"\n Error loading transcript for {url}: {e}")

    print(f"Total YouTube documents: {len(yt_docs)}")
    return yt_docs


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200
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
    print("Loading local documents...")
    file_docs = load_documents()

    print("Loading YouTube videos...")
    youtube_docs = load_youtube_videos()

    all_docs = file_docs + youtube_docs
    print(f"Total loaded documents: {len(all_docs)}. Chunking...")
    chunks = chunk_documents(all_docs)

    print(f"Created {len(chunks)} chunks. Building vector DB...")
    build_vector_db(chunks)
