from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings

CHROMA_PATH = "chroma"
DATA_PATH = "data"

class Models():
    def __init__(self):
        # select ollama model for embedding generation
        self.embeddings_ollama = OllamaEmbeddings(
            model="mxbai-embed-large"
        )

        # select ollama model for QA task and response generation
        self.model_ollama = OllamaLLM(
            model="hf.co/bartowski/gemma-2-9b-it-abliterated-GGUF:Q4_K_S",
            # model="llama3.2",
            temperature=0.1,
        )


