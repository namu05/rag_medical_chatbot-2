from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from typing import List
from dotenv import load_dotenv
import os

load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

FAISS_DIR = "vectorstore_faiss"  # directory where the FAISS index is saved

def create_vectorstore(docs: List[Document], model_name: str):
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True}
    )

    # Check if persistent index exists
    if os.path.exists(f"{FAISS_DIR}/index.faiss") and os.path.exists(f"{FAISS_DIR}/index.pkl"):
        print("📂 Loading cached FAISS index...")
        db = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)

    else:
        print("⚙️ Building new FAISS index...")
        db = FAISS.from_documents(docs, embeddings)
        db.save_local(FAISS_DIR)

    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 4})
    return retriever
