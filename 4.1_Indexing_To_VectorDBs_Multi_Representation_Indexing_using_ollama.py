import os
# Set USER_AGENT environment variable
os.environ["USER_AGENT"] = "MyCustomUserAgent/1.0"

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryByteStore
import bs4
from typing import List
import uuid

# Step 1: Load documents
print("Step 1: Loading documents...")
loader = WebBaseLoader(
    web_paths=[
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2024-02-05-human-data-quality/"
    ],
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()
print(f"Loaded {len(docs)} documents")

# Step 2: Create summaries
print("\nStep 2: Creating summaries...")
llm = OllamaLLM(model="llama3.2")

summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that creates concise summaries of documents."),
    ("user", "Please provide a concise summary of the following text:\n\n{text}")
])

summaries = []
for i, doc in enumerate(docs):
    print(f"⏳ Summarizing document {i+1}/{len(docs)}...")
    summary = llm.invoke(summary_prompt.format_messages(text=doc.page_content))
    summaries.append(summary)
print("✅ Summarization complete")

# Step 3: Create vector store and retriever
print("\nStep 3: Creating vector store and retriever...")
embedding_model = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(
    collection_name="multi_vector_store",
    embedding_function=embedding_model
)

# Create document store
docstore = InMemoryByteStore()

# Create retriever
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    id_key="doc_id"
)

# Add documents to retriever
for i, (doc, summary) in enumerate(zip(docs, summaries)):
    # Create unique ID for the document
    doc_id = str(uuid.uuid4())
    
    # Add summary to vector store
    retriever.vectorstore.add_texts(
        texts=[summary],
        metadatas=[{"doc_id": doc_id}]
    )
    
    # Add original document to docstore
    retriever.docstore.mset([(doc_id, doc.page_content)])

print(f"Indexed {len(docs)} documents")

# Step 4: Test the implementation
print("\nStep 4: Testing the implementation...")
query = "How do autonomous agents use memory for decision making?"

# Use invoke instead of get_relevant_documents
results = retriever.invoke(query)

print("\n=== Query Results ===")
for i, res in enumerate(results):
    # Handle both string and Document objects
    content = res.page_content if isinstance(res, Document) else res
    print(f"\n[{i}] {content[:500]}...\n---") 