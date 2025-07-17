from typing import List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document
import csv
import os

# ------------------------------
# AgentState with memory types
# ------------------------------
class AgentState(dict):
    """
    Maintains the conversational and memory state of the RAG chatbot.

    Attributes:
        messages: Full message history (user + AI).
        short_term: Recent context messages used for immediate response.
        long_term: Important historical context preserved across sessions.
        documents: Retrieved documents used to generate answer.
        on_topic: "Yes" or "No" to indicate if user query is relevant.
        rephrased_question: Question rewritten for optimal retrieval.
        proceed_to_generate: Boolean flag to proceed with answer generation.
        rephrase_count: Number of refinements attempted.
        question: Original HumanMessage input.
        feedback: User's feedback (Yes/No).
    """
    messages: List[BaseMessage] = []
    short_term: List[BaseMessage] = []
    long_term: List[BaseMessage] = []
    documents: List[Document] = []
    on_topic: str = ""
    rephrased_question: str = ""
    proceed_to_generate: bool = False
    rephrase_count: int = 0
    question: HumanMessage = None
    feedback: str = ""

# ------------------------------
# Shared components
# ------------------------------
llm = None
retriever = None
rag_chain = None

# ------------------------------
# Logger for UI
# ------------------------------
UI_LOGS = []

def log_ui(message: str):
    """
    Logs messages for CLI display and UI debugging.

    Args:
        message (str): The message to print and store.
    """    
    print(f"📜 {message}")
    UI_LOGS.append(message)

# ------------------------------
# Dual Memory Helpers
# ------------------------------
def is_important(message: str) -> bool:
    """
    Determines whether a message should be persisted to long-term memory.

    Args:
        message (str): The user question.

    Returns:
        bool: True if message is important based on keyword match.
    """    
    keywords = ["dosage", "side effect", "administration", "clinical", "aimovig", "repatha"]
    return any(k in message.lower() for k in keywords)

# ------------------------------
# Graph Nodes
# ------------------------------
def question_rewriter(state: AgentState):
    """
    Rewrites the user question into a standalone version optimized for document retrieval.

    If enough history is available, uses it to provide context-aware rephrasing.

    Args:
        state (AgentState): The current agent state.

    Returns:
        AgentState: Updated state with rephrased question.
    """

    state["documents"] = []
    state["on_topic"] = ""
    state["rephrased_question"] = ""
    state["proceed_to_generate"] = False
    state["rephrase_count"] = 0
    state.setdefault("messages", [])
    state.setdefault("short_term", [])
    state.setdefault("long_term", [])

    if state["question"] not in state["messages"]:
        state["messages"].append(state["question"])

    if len(state["messages"]) > 1:
        conv = state["short_term"][-2:]
        msgs = [
            SystemMessage(content="You are a helpful assistant that rephrases the user's question to be a standalone question optimized for retrieval.")
        ] + conv + [HumanMessage(content=state["question"].content)]
        rephrase_prompt = ChatPromptTemplate.from_messages(msgs)
        out = llm.invoke(rephrase_prompt.format())
        state["rephrased_question"] = out.strip()
    else:
        state["rephrased_question"] = state["question"].content
        log_ui(f"🗣️ User Question: {state['rephrased_question']}")

    return state

def question_classifier(state: AgentState):
    """
    Classifies whether the user question falls within the relevant domain scope.

    Returns 'Yes' or 'No' and updates the state.

    Args:
        state (AgentState): The current agent state.

    Returns:
        AgentState: Updated state with 'on_topic' label.
    """    
    # Drug-agnostic section labels
    neutral_sections = [
        "Overview of the medical document",
        "Indications and clinical use of biologic treatments",
        "Dosage and administration details of biologics",
        "Comparative analysis between biologic therapies",
        "Scientific references and clinical study sources"
    ]

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "You are a strict Yes/No classifier.\n"
            "Determine if the user's question relates to **any** of the following topics based on a medical summary document:\n\n"
            + "\n".join([f"{i+1}. {s}" for i, s in enumerate(neutral_sections)]) +
            "\n\nOnly respond with 'Yes' or 'No'. Do not explain or elaborate."
        )),
        HumanMessage(content=state["rephrased_question"])
    ])

    resp = llm.invoke(prompt.format())
    answer = resp.strip().lower()
    log_ui(f"📌 Classifier Response: {answer}")
    state["on_topic"] = "Yes" if answer.startswith("y") else "No"
    log_ui(f"✅ On Topic: {state['on_topic']}")
    return state

def retrieve(state: AgentState):
    """
    Retrieves documents using the rephrased question from the vectorstore retriever.

    Args:
        state (AgentState): The current agent state.

    Returns:
        AgentState: Updated state with retrieved documents.
    """    
    state["documents"] = retriever.invoke(state["rephrased_question"])
    log_ui(f"📚 Retrieved {len(state['documents'])} documents.")
    return state

def retrieval_grader(state: AgentState):
    """
    Grades each retrieved document for relevance to the question.

    Keeps only relevant documents and sets flag to proceed.

    Args:
        state (AgentState): The current agent state.

    Returns:
        AgentState: Updated state with filtered documents.
    """    
    relevant = []
    system = SystemMessage(content="Grade if the document is relevant to the user question. Answer Yes or No.")
    for doc in state["documents"]:
        human = HumanMessage(content=f"Q: {state['rephrased_question']}\nDoc: {doc.page_content}")
        prompt = ChatPromptTemplate.from_messages([system, human])
        response = llm.invoke(prompt.format())
        is_relevant = response.strip().lower().startswith("y")
        log_ui(f"🧪 Relevance Check: {'✅' if is_relevant else '❌'}")
        if is_relevant:
            relevant.append(doc)
    state["documents"] = relevant
    state["proceed_to_generate"] = len(relevant) > 0
    log_ui(f"📊 Proceed to generate: {state['proceed_to_generate']}")
    return state

def refine_question(state: AgentState):
    """
    Refines the rephrased question to improve retrieval if relevance grading fails.

    Limits the number of refinements to 2.

    Args:
        state (AgentState): The current agent state.

    Returns:
        AgentState: Updated state with refined question.
    """    
    rephrase_count = state.get("rephrase_count", 0)
    if rephrase_count >= 2:
        return state
    system_message = SystemMessage(
        content="You are a helpful assistant that slightly refines the user's question to improve retrieval results."
    )
    human_message = HumanMessage(
        content=f"Original question: {state['rephrased_question']}\n\nRefine it:"
    )
    refine_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
    out = llm.invoke(refine_prompt.format())
    state["rephrased_question"] = out.strip()
    state["rephrase_count"] = rephrase_count + 1
    return state

def generate_answer(state: AgentState):
    """
    Generates the final answer using short-term memory and retrieved documents.

    Persists relevant Q&A pairs into long-term memory.

    Args:
        state (AgentState): The current agent state.

    Returns:
        AgentState: Updated state with generated AI answer.
    """    

    short_context = state["short_term"][-5:] if len(state["short_term"]) >= 2 else state["short_term"]

    response = rag_chain.invoke({
        "history": short_context,
        "context": state["documents"],
        "question": state["rephrased_question"]
    })

    ai_msg = AIMessage(content=response.strip())

    # Update message logs
    state["messages"].append(ai_msg)
    state["short_term"].append(state["question"])
    state["short_term"].append(ai_msg)

    # Persist if important
    if is_important(state["question"].content):
        state["long_term"].append(state["question"])
        state["long_term"].append(ai_msg)

    log_ui("📝 Answer generated with dual memory.")
    return state

def cannot_answer(state: AgentState):
    """
    Fallback if no relevant documents are retrieved or rephrasing fails.

    Args:
        state (AgentState): The current agent state.

    Returns:
        AgentState: Updated state with fallback response.
    """

    state["messages"].append(AIMessage(content="Sorry, I couldn't find an answer."))
    log_ui("❌ No relevant documents found. Cannot answer.")
    return state

def off_topic_response(state: AgentState):
    """
    Fallback response for off-topic questions.

    Args:
        state (AgentState): The current agent state.

    Returns:
        AgentState: Updated state with off-topic response.
    """

    state["messages"].append(AIMessage(content="That question is off-topic for this case study."))
    log_ui("🚫 Question marked as off-topic.")
    return state

def feedback_collector(state: AgentState):
    """
    Collects human feedback (Yes/No) after answer generation and appends it to `feedback_log.csv`.

    Args:
        state (AgentState): The current agent state.

    Returns:
        AgentState: Updated state with feedback logged.
    """

    # Ask for feedback from user
    feedback = input("🧠 [Feedback] Was this answer helpful? (Yes/No): ").strip().lower()
    state["feedback"] = feedback

    # Extract the last AI answer only (no need to regenerate it!)
    answer_msg = next((msg for msg in reversed(state["messages"]) if isinstance(msg, AIMessage)), None)
    answer = answer_msg.content if answer_msg else ""

    # Log and store
    log_ui(f"🧾 Human Feedback: {feedback}")
    record = {
        "question": state["question"].content,
        "rephrased_question": state.get("rephrased_question", ""),
        "documents": " | ".join([doc.page_content.replace("\n", " ")[:300] for doc in state.get("documents", [])]),
        "answer": answer,
        "feedback": feedback
    }

    file_path = "feedback_log.csv"
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["question", "rephrased_question", "documents", "answer", "feedback"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)

    log_ui("📁 Feedback saved to feedback_log.csv")
    return state


