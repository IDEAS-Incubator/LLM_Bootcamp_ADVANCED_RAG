# RAG (Retrieval-Augmented Generation) Implementation

This project implements a Retrieval-Augmented Generation (RAG) system using LangChain and Ollama. It demonstrates how to build a question-answering system that combines document retrieval with large language models.

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