
# 🛡️ Cybersecurity RAG Chatbot

*A Defensive, NIST-Inspired Cybersecurity Assistant Powered by RAG + Qwen2.5 LLM*

---

## 📌 Overview

The **Cybersecurity RAG Chatbot** is a domain-specific assistant designed to provide **defensive cybersecurity guidance** using:

* **Retrieval-Augmented Generation (RAG)**
* **HuggingFace Qwen/Qwen2.5-7B-Instruct**
* **LangChain + ChromaDB**
* **FastAPI backend**
* **Simple browser-based chat UI**

This chatbot answers questions based strictly on **uploaded NIST-style cybersecurity documents**, making it reliable, explainable, and grounded in real policy/guidelines.

---

## 🧠 Features

### ✔ Retrieval-Augmented Generation (RAG)

Your cybersecurity PDFs/TXT files are converted into vector embeddings and stored in **ChromaDB**.
The chatbot retrieves the most relevant sections before generating an answer.

### ✔ Defensive Security Only

The assistant provides:

* NIST-aligned best practices
* Secure configuration guidance
* Incident response concepts
* Access control / audit / hardening recommendations

It **does NOT** provide offensive, hacking, or exploit instructions.

### ✔ HuggingFace Qwen2.5-7B-Instruct

Uses the **HF Inference API** via the modern:

```python
client.chat.completions.create(...)
```

### ✔ FastAPI Backend

Exposes a `/chat` endpoint used by the frontend.

### ✔ Simple, Clean Web UI

Built with vanilla HTML/JS — no framework required.

---

## 📁 Project Structure

```
cybersec-chatbot/
│
├── data/                 → Cybersecurity PDFs/TXT (ignored by Git)
├── vectorstore/          → ChromaDB persistent database
│
├── src/
│   ├── api.py            → FastAPI server
│   ├── ingest.py         → Loads & indexes documents into vector DB
│   ├── rag_chain.py      → RAG pipeline + HF LLM calls
│   ├── config.py         → Configuration (paths, HF API key)
│   └── __init__.py
│
├── frontend/
│   └── index.html        → Chat UI
│
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 2️⃣ Add your HuggingFace API Key

Create a `.env` file in the project root:

```
HF_API_KEY=your_huggingface_api_key
```

---

### 3️⃣ Add Cybersecurity Documents

Place your `.pdf` or `.txt` files into:

```
data/
```

These will be embedded and stored in the vector database.

---

### 4️⃣ Build the Vector Database

Run:

```bash
python -m src.ingest
```

You should see:

```
Loading documents...
Loaded X documents. Chunking...
Created Y chunks. Building vector DB...
Vector store created successfully!
```

---

### 5️⃣ Start the FastAPI Server

```bash
uvicorn src.api:app --reload
```

The backend runs at:

👉 [http://127.0.0.1:8000](http://127.0.0.1:8000)

Test:

👉 [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

### 6️⃣ Start the Frontend (Local Web Server)

```bash
cd frontend
python -m http.server 5500
```

Open:

👉 [http://127.0.0.1:5500/index.html](http://127.0.0.1:5500/index.html)

You now have a working chatbot 🎉

---

## 🧩 How It Works (Architecture)

1. **Document Ingestion (`ingest.py`)**

   * Load PDFs/Text
   * Split into chunks
   * Create embeddings
   * Store in ChromaDB

2. **RAG Query Flow (`rag_chain.py`)**

   * Convert question → embedding
   * Retrieve similar context chunks
   * Feed context → Qwen LLM
   * Generate grounded cybersecurity answer

3. **Qwen2.5-7B LLM**

   * Uses HuggingFace API
   * Chat completion mode
   * Strong reasoning for defensive security topics

4. **API Layer (`api.py`)**

   * `/chat` → Accepts questions
   * Calls RAG pipeline
   * Returns answer + cited sources

5. **Frontend**

   * Sends question via fetch()
   * Shows answers in a chat interface

---

## 🛡️ Safety & Guardrails

The model is instructed to:

* Provide **defensive security only**
* Refuse:

  * Hacking instructions
  * Exploitation techniques
  * Malware guidance
* Redirect harmful queries to safe practices

---

## 📌 Requirements

* Python 3.10+
* HuggingFace API key
* Internet access (for model inference)

---

## 🧪 Example Queries

Try asking:

* "According to NIST, what are the phases of incident response?"
* "What access control best practices should be implemented?"
* "How should logs be managed for audit and accountability?"
* "Explain configuration management from a cybersecurity perspective."

---

## 🤝 Contributing

Pull requests are welcome.
Please keep contributions aligned with **defensive cybersecurity principles**.

---

## 📜 License

MIT License (or choose your own).

---

## ⭐ If you like this project

Give the repo a star on GitHub — it helps others find it!

