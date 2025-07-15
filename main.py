from config import PDF_PATH, EMBED_MODEL, OLLAMA_MODEL
from data_loader.pdf_parser import extract_structured_documents
from retriever.vectorstore import create_vectorstore
from llm.local_llm import load_local_llm
from rag.rag_chain import build_rag_chain
from rag.graph import build_graph
from rag.graph_nodes import UI_LOGS  # 👈 to print UI trace
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from rag.graph_nodes import feedback_collector
import os

# ---------------------------
# ✅ Load environment + models
# ---------------------------
load_dotenv()

print("🔄 Loading PDF documents...")
docs = extract_structured_documents(PDF_PATH)

print("🔄 Creating vectorstore...")
retriever = create_vectorstore(docs, EMBED_MODEL)

print("🧠 Loading local model:", OLLAMA_MODEL)
llm = load_local_llm(OLLAMA_MODEL)

print("🔗 Building RAG chain & graph...")
rag_chain = build_rag_chain(llm)
graph = build_graph(rag_chain, retriever, llm)

# ---------------------------
# ✅ Start chat loop
# ---------------------------
user_id = input("👤 Enter your user ID: ").strip()
print("\n✅ Chat ready. Type 'exit' to quit.\n")

while True:
    try:
        question = input("Your question: ").strip()
        if question.lower() in ("exit", "quit"):
            print("👋 Bye!")
            break

        # 🔁 Run LangGraph up to answer generation
        result_state = graph.invoke(
            input={"question": HumanMessage(content=question)},
            config={"configurable": {"thread_id": user_id}}
        )

        # ✅ Display only the final answer
        print("\n------ 🤖 Final Answer ------")
        for msg in reversed(result_state["messages"]):
            if isinstance(msg, AIMessage):
                print(f"\n🤖 {msg.content.strip()}\n")
                break

        # 🧠 Ask for feedback AFTER displaying answer
        result_state = feedback_collector(result_state)

    except Exception as e:
        print("❌ Error:", str(e))
