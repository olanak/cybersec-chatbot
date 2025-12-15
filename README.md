
```markdown
# 🛡️ Cybersecurity RAG Chatbot (NIST-Based)

A **domain-specific cybersecurity chatbot** built using **Retrieval-Augmented Generation (RAG)** and a Large Language Model (LLM).  
The system answers questions based strictly on **local NIST cybersecurity documents** and supports **comparison between RAG and non-RAG modes**.

---

## 📌 Project Overview

This project implements a **Retrieval-Augmented Generation (RAG) pipeline** that:

- Ingests **PDF, Word (.docx), and Excel (.xlsx)** cybersecurity documents
- Stores embeddings in **ChromaDB**
- Uses **Qwen/Qwen2.5-7B-Instruct** via Hugging Face for answer generation
- Allows **runtime comparison** between:
  - **RAG-enabled answers**
  - **LLM-only (no RAG) answers**
- Reduces hallucination by grounding responses in local documents

The project is designed for **academic evaluation**, **cybersecurity research**, and **demonstrating RAG effectiveness**.

---

## 🧠 Key Features

- ✅ Retrieval-Augmented Generation (RAG)
- ✅ RAG ON / OFF toggle in the UI
- ✅ Local-only execution (no cloud dependency)
- ✅ Multi-format document ingestion:
  - PDF
  - DOCX
  - XLSX
- ✅ Hallucination reduction with relevance thresholds
- ✅ Source citation (only when grounded)
- ✅ Minimal, white, academic UI
- ✅ FastAPI backend

---

## 📂 Project Structure

```

cybersec-chatbot/
│
├── data/                  # Local cybersecurity documents (PDF, DOCX, XLSX)
├── vectorstore/           # ChromaDB persistent embeddings
│
├── src/
│   ├── ingest.py          # Document ingestion & embedding
│   ├── rag_chain.py       # RAG + non-RAG logic
│   ├── api.py             # FastAPI backend
│   ├── config.py          # Configuration variables
│   └── **init**.py
│
├── frontend/
│   └── index.html         # Minimal web UI with RAG toggle
│
├── requirements.txt
└── README.md

````

---

## ⚙️ Requirements

- Python 3.10+
- Hugging Face API key
- Virtual environment (recommended)

---

## 🔧 Installation

### 1️⃣ Create & activate virtual environment
```bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows
````

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Configuration

Create a `.env` file in the project root:

```env
HF_API_KEY=your_huggingface_api_key
```

---

## 📥 Document Ingestion

Place your documents in the `data/` directory:

```
data/
├── NISTCSFW.pdf
├── NIST.CSWP.29.ipd.docx
└── csf2.xlsx
```

Run ingestion:

```bash
python -m src.ingest
```

This will:

* Load documents
* Chunk text
* Generate embeddings
* Persist vectors in ChromaDB

---

## 🚀 Running the Application (Local)

### ▶ Backend (FastAPI)

From the project root:

```bash
uvicorn src.api:app --reload
```

Backend URL:

```
http://127.0.0.1:8000
```

Health check:

```
http://127.0.0.1:8000/health
```

---

### ▶ Frontend (UI)

From the `frontend` directory:

```bash
python -m http.server 5500 --bind 127.0.0.1
```

Open in browser:

```
http://127.0.0.1:5500
```

---

## 🧪 Using RAG vs Non-RAG Mode

The UI includes a checkbox:

* ✅ **Checked** → RAG enabled (answers grounded in documents + sources)
* ❌ **Unchecked** → LLM-only answers (no retrieval, no sources)

This allows direct **comparison of hallucination vs grounded answers**, as required by RAG evaluation methodologies.

---

## 📊 Example Questions

**In-domain (RAG ON):**

* What are the NIST CSF Tiers?
* How do Tiers support cybersecurity risk management?
* What is the purpose of CSF 2.0?

**Out-of-domain:**

* Explain quantum key distribution
* What is the OSI model?

Expected behavior:

* RAG answers grounded questions
* Non-RAG may hallucinate
* Out-of-scope questions return “not found” with no sources

---

## 🛡️ Hallucination Control

* Similarity-score thresholding
* Soft relevance gating
* No sources returned if no relevant document context exists
* Explicit refusal when information is not in documents

---

## 🎓 Academic Relevance

This project demonstrates:

* Practical RAG implementation
* LLM grounding techniques
* Multi-format document retrieval
* Controlled evaluation (RAG vs no-RAG)
* Responsible AI behavior in cybersecurity contexts

---

## 📜 License

This project is intended for educational and research purposes.

---