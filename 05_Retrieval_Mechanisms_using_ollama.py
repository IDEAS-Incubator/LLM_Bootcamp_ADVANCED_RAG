from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from sklearn.metrics import pairwise_distances
import os, bs4

# ---- STEP 1: Load Blog ----

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))
    ),
)
docs = loader.load()
print(f"✅ Loaded {len(docs)} documents.")

# ---- STEP 2: Split ----

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
splits = text_splitter.split_documents(docs)
print(f"✅ Split into {len(splits)} chunks.")

# ---- STEP 3: Index ----

embedding_model = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(splits, embedding=embedding_model)
retriever = vectorstore.as_retriever()
print("✅ Documents embedded and indexed in Chroma.")

# ---- STEP 4: RAG-Fusion (multi-query generation) ----

llm = OllamaLLM(model="llama3.2", temperature=0)
prompt_rag_fusion = ChatPromptTemplate.from_template("""
You are a helpful assistant that generates multiple search queries based on a single input query. 
Generate 4 search queries related to: {question}

Output (one per line):
""")

generate_queries_chain = (
    {"question": RunnablePassthrough()}
    | prompt_rag_fusion
    | llm
    | StrOutputParser()
    | (lambda x: [line.strip() for line in x.strip().split("\n") if line.strip()])
)

# ---- STEP 5: Reciprocal Rank Fusion (RRF) ----

def reciprocal_rank_fusion(results_lists, k=60):
    fused_scores = {}
    for results in results_lists:
        for rank, doc in enumerate(results):
            doc_id = doc.page_content[:50]  # simple ID using content snippet
            score = 1 / (k + rank)
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + score
    sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc for doc_id, _ in sorted_docs for results in results_lists for doc in results if doc.page_content.startswith(doc_id)]

# ---- STEP 6: Run a Query with RAG-Fusion ----

question = "How do autonomous agents use memory?"
print(f"\n🔍 User Question: {question}")

queries = generate_queries_chain.invoke(question)

print("\n📌 Generated Sub-Queries:")
for i, q in enumerate(queries, 1):
    print(f"{i}. {q}")

# Retrieve for each sub-query
results_per_query = [retriever.invoke(q) for q in queries]

# Fuse
final_results = reciprocal_rank_fusion(results_per_query)

# ---- STEP 7: Show Final Results ----

print("\n📚 Top Fused Results:")
for i, doc in enumerate(final_results[:3], 1):
    print(f"\n[{i}] {doc.page_content[:500]}...\n---")
