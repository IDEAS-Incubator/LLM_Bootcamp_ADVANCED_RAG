import os
# Set USER_AGENT environment variable
os.environ["USER_AGENT"] = "MyCustomUserAgent/1.0"

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryByteStore
import bs4, uuid, os

# ---- STEP 1: Load Web Docs ----

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2024-02-05-human-data-quality/"
]

docs = []
for url in urls:
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header")))
    )
    loaded = loader.load()
    print(f"Loaded {len(loaded)} docs from: {url}")
    docs.extend(loaded)

# ---- STEP 2: Summarize Each Doc ----

llm = OllamaLLM(model="llama3.2")
prompt = ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
summarizer_chain = (
    {"doc": lambda x: x.page_content}
    | prompt
    | llm
    | StrOutputParser()
)

print("\n⏳ Summarizing...")
summaries = summarizer_chain.batch(docs, {"max_concurrency": 5})
print("✅ Summarization complete.")

# ---- STEP 3: Embed Summaries + Store in Chroma ----

embedding_model = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(collection_name="summaries", embedding_function=embedding_model)

# Link summaries to original docs
store = InMemoryByteStore()
id_key = "doc_id"
doc_ids = [str(uuid.uuid4()) for _ in docs]

summary_docs = [
    Document(page_content=summaries[i], metadata={id_key: doc_ids[i]})
    for i in range(len(summaries))
]

retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key
)

retriever.vectorstore.add_documents(summary_docs)
retriever.docstore.mset(list(zip(doc_ids, docs)))

print(f"\n✅ Indexed {len(summary_docs)} summarized chunks to ChromaDB.")

# ---- STEP 4: Try Query ----

query = "How do autonomous agents use memory for decision making?"
print(f"\n🔍 Running query: {query}\n")

results = retriever.invoke(query)

print("=== Top Results ===")
for i, res in enumerate(results, 1):
    print(f"\n[{i}] {res.page_content[:500]}...\n---")
