# RAG (Retrieval-Augmented Generation) Implementation

This project implements a Retrieval-Augmented Generation (RAG) system using LangChain and Ollama. It demonstrates how to build a question-answering system that combines document retrieval with large language models.

Below is a detailed overview of each Python script in this repository:

- **01_Introduction_To_RAG_using_ollama.py**
  - Basic process of building RAG application
- **02_Query_Transformations_using_ollama.py**
  - Techniques for modifying questions for retrieval
- **03_Advanced_RAG_using_ollama.py**
  - Advanced RAG techniques and implementations
- **04_Indexing_To_VectorDBs_using_ollama.py**
  - Various indexing methods in the Vector DB
- **05_Retrieval_Mechanisms_using_ollama.py**
  - Reranking, RAG Fusion, and other techniques
- **06_Self_Reflection_Rag_using_ollama.py**
  - RAG with self-reflection/self-grading on retrieved documents and generations
- **07_Agentic_Rag_using_ollama.py**
  - RAG with agentic flow on retrieved documents and generations
- **08_Adaptive_Agentic_Rag_using_ollama.py**
  - RAG with adaptive agentic flow
- **09_Corrective_Agentic_Rag_using_ollama.py**
  - RAG with corrective agentic flow on retrieved documents and generations
- **10_LLAMA_3_Rag_Agent_Local_using_ollama.py**
  - LLAMA 3 8B Agent RAG that works locally

## Prerequisites

- Python 3.11+
- Ollama (for running LLMs locally)
- Required Python packages (see Installation)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd RAG
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Install and set up Ollama:
```bash
# Install Ollama (follow instructions at https://ollama.ai)
# Pull required models
ollama pull llama3.2
ollama pull nomic-embed-text
```

## Usage

1. Start Ollama:
```bash
ollama serve
```

2. Run any of the example scripts:
```bash
python 01_Introduction_To_RAG_using_ollama.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Acknowledgments

- [LangChain](https://www.langchain.com/)
- [Ollama](https://ollama.ai/)
- [ChromaDB](https://www.trychroma.com/) 