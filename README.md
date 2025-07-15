# 🧠 RAG-Based Medical Chatbot

A domain-specific, document-grounded medical chatbot built using **LangChain**, **LangGraph**, and a **locally hosted open-source LLM**. This chatbot accurately answers user questions based solely on medical case study documents using a **Retrieval-Augmented Generation (RAG)** pipeline.

---

## 🔧 Features

- 📄 Q&A over structured case study PDFs
- 🧠 Uses a **local LLM** (via `Ollama` or Hugging Face)
- 🔁 Modular **LangGraph pipeline** with:
  - Question Rewriting
  - Topic Classification
  - Document Retrieval
  - Relevance Grading
  - Answer Generation
  - Off-topic/Fallback Handling
- 🗣️ Multi-turn conversation with **short- and long-term memory**
- ✅ Feedback collection for model evaluation and improvement

---

## 🧱 Project Structure

```bash
rag_medical_chatbot-2/
├── main.py
├── config.py
├── documents/
│   └── Case Study.pdf   # <-- not committed (gitignored)
├── feedback_log.csv
├── vectorstore_faiss/
├── rag/
│   ├── graph.py
│   ├── graph_nodes.py
│   └── rag_chain.py
├── data_loader/
│   └── pdf_parser.py
├── retriever/
│   └── vectorstore.py
├── llm/
│   └── local_llm.py
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🛠️ Installation

### ✅ Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/download) (for local LLMs)
- A supported local LLM model (e.g., `llama3`) downloaded via Ollama
- (Optional) Hugging Face token in `.env` if using HF models for embedding

---

### 💻 Setup (macOS / Linux)

```bash
# 1. Clone and navigate to the project
git clone https://github.com/namu05/rag_medical_chatbot-2.git
cd rag_medical_chatbot-2

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the model using Ollama (in a separate terminal)
ollama run llama3
```

---

### 💻 Setup (Windows)

```bash
# 1. Clone and navigate to the project
git clone https://github.com/namu05/rag_medical_chatbot-2.git
cd rag_medical_chatbot-2

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scriptsctivate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the model using Ollama (in another Command Prompt)
ollama run llama3
```

---

### ▶️ Run the Chatbot

```bash
python backend/main.py
```

You will be prompted for a user ID, and then the chatbot will begin:

```
Your question: What are the side effects of Repatha?
```

After the response, you'll be prompted for feedback:

```
🧠 [Feedback] Was this answer helpful? (Yes/No):
```
