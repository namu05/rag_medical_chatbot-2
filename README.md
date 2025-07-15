# 🧠 Medical RAG Chatbot (Local LLM with Feedback Logging)

This is a Retrieval-Augmented Generation (RAG) chatbot that answers medical queries based solely on a structured PDF case study. It uses a **locally hosted LLM via Ollama**, **LangChain**, and **FAISS** vector search with Hugging Face embeddings.

---

## 📚 Case Study Scope

- Drugs Information
- Dosage, Administration, Side Effects, and Clinical Data
- Comparative Analysis

---

## 🚀 Features

- **RAG pipeline using LangGraph**  
- **Dual memory (short-term + long-term)** tracking  
- **Local LLM with Ollama** (e.g., LLaMA 3)  
- **Human feedback** collection & CSV logging  
- **Table-aware PDF parsing** and chunking  
- CLI-based chat interface

---

## 🛠️ Installation

### ✅ Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/download) installed and running locally
- `llama3` model downloaded via Ollama
- (Optional) Hugging Face token for better embedding model access in .env file

---

### 💻 Setup on macOS/Linux

```bash
# Clone and enter the project
git clone https://github.com/yourusername/rag-medical-chatbot.git
cd rag-medical-chatbot

# Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download and run the model via Ollama (in another terminal tab)
ollama run llama3

---

### 💻 Setup on Windows

# Clone and enter the project
git clone https://github.com/yourusername/rag-medical-chatbot.git
cd rag-medical-chatbot

# Set up virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download and run the model via Ollama (in a separate command prompt window)
ollama run llama3



#Run the Chatbot

python main.py

You will be prompted for a user ID, then can begin chatting:

Your question: What are the side effects of Repatha?

After receiving an answer, you’ll be asked:
🧠 [Feedback] Was this answer helpful? (Yes/No):


