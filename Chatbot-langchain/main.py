from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3.1")
response = llm.invoke("Hey, are you here?")

print(response)