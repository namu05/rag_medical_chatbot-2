from langchain_core.prompts import ChatPromptTemplate

def build_rag_chain(llm):
    template = """Answer the question based on the following context and the Chathistory. Especially take the latest question into consideration:

Chathistory: {history}

Context: {context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)
    return prompt | llm