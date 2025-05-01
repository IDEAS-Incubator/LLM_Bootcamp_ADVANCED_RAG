# pip install -U langchain langchain-community langchain-ollama chromadb beautifulsoup4

import os
# Set USER_AGENT environment variable
os.environ["USER_AGENT"] = "MyCustomUserAgent/1.0"

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import bs4

"""
| Parameter | Impact |
|:---|:---|
| `chunk_size` too small | → Less context for LLM, may miss meaning |
| `chunk_size` too large | → Slower search, noisy embeddings, risk of truncation |
| `chunk_overlap` too small | → Risk of broken thoughts at chunk boundaries |
| `chunk_overlap` too large | → Redundant context, more memory usage, slower indexing |


"""
# ---- INDEXING ----

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

embedding_model = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)
retriever = vectorstore.as_retriever()

# ---- RETRIEVAL + GENERATION ----

prompt_template = """
Use the following context to answer the question.
Context:
{context}

Question: {question}
"""

llm = OllamaLLM(model="llama3.2")  # or any model you have installed

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


rag_chain = (
    {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
    | RunnableLambda(lambda x: prompt_template.format(**x))
    | llm
    | StrOutputParser()
)

print(rag_chain.invoke("What is Task Decomposition?"))
      
