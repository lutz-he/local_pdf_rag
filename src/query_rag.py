import re
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

from src.utils import Models, CHROMA_PATH, DATA_PATH

# Init models
models = Models()
embeddings = models.embeddings_ollama  
llm = models.model_ollama

def remove_thinking(response: str) -> str:
    """Remove any 'thinking' steps from the model response."""
    start_pattern = "<think>"
    end_pattern = "</think>"
    excl = re.compile(f"{start_pattern}.*?{end_pattern}", re.DOTALL)
    return excl.sub("", response)

# Prompt Template
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}

Don't include any thinking steps or additional information not contained in the context.
Just respond with the answer. If you don't know the answer, just say "I don't know".
"""

def query_rag(query_text: str, top_k: int = 5, similarity_threshold: float = 0.25) -> str:
    db = Chroma(
        collection_name="documents",
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        )

    # Search the DB.
    retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": top_k, "score_threshold": similarity_threshold})
    results = retriever.invoke(query_text)

    # If no results, lower the threshold until we get some results.
    new_threshold = similarity_threshold
    if len(results) == 0:
        while len(results) == 0 and new_threshold > 0:
            new_threshold -= 0.1
            retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": top_k, "score_threshold": new_threshold})
            results = retriever.invoke(query_text)
    if len(results) == 0:
        print("No results found. Please try a different query.")
        return "No results found."

    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    response_text = llm.invoke(prompt)
    response_text = remove_thinking(response_text)

    sources = [doc.metadata.get("id", None) for doc in results]

    return response_text, sources, results

def print_output(response_text: str, sources: list[str], DATA_PATH="") -> None:
    print(f"Response: {response_text}")

    sources = [src.replace(DATA_PATH, "") for src in sources if src is not None]
    sources = [src.replace("\\", "") for src in sources if src is not None]

    print("Sources:")
    for i, src in enumerate(sources):
        print(f"- [{i+1}] {src}")