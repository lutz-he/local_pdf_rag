# PDF RAG Tool

A **very simple** local Retrieval-Augmented Generation (RAG) system for querying PDF documents using LLMs. This tool uses Ollama for running local language models and ChromaDB for vector storage.

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install Ollama:**
   - Download and install Ollama from [https://ollama.com/download](https://ollama.com/download)
   - Start Ollama:
     ```bash
     ollama serve
     ```
   - Pull required models (see [`src/utils.py`](./src/utils.py)):
     ```bash
     ollama pull llama3
     ```

## Usage

### 1. Ingest PDF(s) into the database

Place your PDF files in the `data/` directory. Then run:
```bash
python src/db_ingest.py
```
This will process all PDFs in `data/` and store their embeddings in ChromaDB. Use `--clean` to clean the database before ingestion.

### 2. Query the RAG system

Run the main script and follow the prompts:
```bash
python main.py
```
Enter your question when prompted. The system will retrieve relevant document chunks and generate an answer using the local LLM via Ollama.

## Notes
- Ensure Ollama is running before querying.
- The database is stored in the `chroma/` directory.
- Models are configured in `src.utils.Models`
- For advanced usage, see comments in the source files in `src/`.
    - e.g. `python main.py --top_k 5 --similarity_threshold 0.25`


## Ideas

* Interface: add file browser and db_ingest
* Testing & Eval
* Include Metadata


---

**Special thanks to [pixegami](https://github.com/pixegami) for the informative tutorial that inspired this project.**
