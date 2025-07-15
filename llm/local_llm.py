from langchain_ollama.llms import OllamaLLM

def load_local_llm(model_name="llama3.1"):
    return OllamaLLM(
        model=model_name,
        temperature=0.3,
        top_p=0.9,
        max_tokens=1080,   # increase output size
    )