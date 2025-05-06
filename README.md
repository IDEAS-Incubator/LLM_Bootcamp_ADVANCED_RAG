# RAG Data Preprocessing & Embedding Pipeline

This project provides a robust pipeline for preparing and embedding text data for Retrieval-Augmented Generation (RAG) and other NLP tasks. It includes data cleanup, chunking, and embedding using the `nomic-embed-text` model from Ollama.

---

## Features
- **Data Cleanup:** Remove HTML, normalize text, strip unwanted characters, and more.
- **Chunking:** Split large documents into overlapping chunks for better context retention.
- **Embedding:** Generate high-quality vector embeddings using Ollama's `nomic-embed-text` model.
- **Similarity Search:** Find the most relevant text chunks for a given query.

---

## Setup

1. **Clone the repository**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **(Optional) Set up your virtual environment**

---

## Usage

### 1. Data Cleanup
See `RAG_optimization/Data preprocessing/1.1_data_cleanup_techbologies.py` for advanced text cleaning functions.

### 2. Chunking
Use the provided chunking functions to split your cleaned text into overlapping chunks.

### 3. Embedding
Generate embeddings for each chunk using Ollama:
```python
import ollama
response = ollama.embeddings(model='nomic-embed-text', prompt="your text chunk here")
embedding = response['embedding']
```

### 4. Similarity Search
Use cosine similarity (e.g., with scikit-learn) to compare query embeddings to your chunk embeddings.

---

## Example Pipeline
See the code in `RAG_optimization/Data preprocessing/1.3_embedding_using_ollama.py` for a full example, including:
- Data cleanup
- Chunking
- Embedding
- Similarity search

---

## Requirements
See `requirements.txt` for all dependencies.

---

## License
MIT 