# rag/graph.py
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from rag.graph_nodes import (
    AgentState,
    question_rewriter,
    question_classifier,
    retrieve,
    retrieval_grader,
    refine_question,
    generate_answer,
    cannot_answer,
    off_topic_response,

)

def on_topic_router(state: AgentState):
    return "retrieve" if state["on_topic"] == "Yes" else "off_topic_response"

def proceed_router(state: AgentState):
    if state["proceed_to_generate"]:
        return "generate_answer"
    elif state.get("rephrase_count", 0) < 2:
        return "refine_question"
    else:
        return "cannot_answer"

def build_graph(rag_chain, retriever, llm):
    # Inject shared objects into graph_nodes
    import rag.graph_nodes as nodes
    nodes.rag_chain = rag_chain
    nodes.retriever = retriever
    nodes.llm = llm

    checkpointer = MemorySaver()
    builder = StateGraph(AgentState)

    builder.add_node("question_rewriter", question_rewriter)
    builder.add_node("question_classifier", question_classifier)
    builder.add_node("retrieve", retrieve)
    builder.add_node("retrieval_grader", retrieval_grader)
    builder.add_node("refine_question", refine_question)
    builder.add_node("generate_answer", generate_answer)
    builder.add_node("cannot_answer", cannot_answer)
    builder.add_node("off_topic_response", off_topic_response)

    builder.set_entry_point("question_rewriter")
    builder.add_edge("question_rewriter", "question_classifier")

    builder.add_conditional_edges("question_classifier", on_topic_router, {
        "retrieve": "retrieve",
        "off_topic_response": "off_topic_response"
    })

    builder.add_edge("retrieve", "retrieval_grader")
    builder.add_conditional_edges("retrieval_grader", proceed_router, {
        "generate_answer": "generate_answer",
        "refine_question": "refine_question",
        "cannot_answer": "cannot_answer"
    })

    builder.add_edge("refine_question", "retrieve")
    builder.add_edge("generate_answer", END)
    builder.add_edge("cannot_answer", END)
    builder.add_edge("off_topic_response", END)

    return builder.compile(checkpointer=checkpointer)
