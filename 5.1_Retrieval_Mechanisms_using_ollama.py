import os
# Set USER_AGENT environment variable
os.environ["USER_AGENT"] = "MyCustomUserAgent/1.0"

from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from sklearn.metrics import pairwise_distances
import bs4

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

# ---- STEP 6: Prompt for Answer Generation with Fallback ----

rag_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.

Use the following context to answer the user's question. 
If the context is unrelated, incomplete, or unhelpful, use your own knowledge to give the best possible answer.

Context:
{context}

Question: {question}
""")

# ---- STEP 7: Run a Query with RAG-Fusion ----

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

# Combine context from top RRF results
combined_context = "\n\n".join([doc.page_content for doc in final_results[:3]])

# Final answer generation
answer_chain = (
    {"context": lambda _: combined_context, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

final_answer = answer_chain.invoke(question)

# ---- STEP 8: Display Results ----

print("\n📚 Top Retrieved Context Chunks:")
for i, doc in enumerate(final_results[:3], 1):
    print(f"\n[{i}] {doc.page_content[:500]}...\n---")

print("\n💡 Final Answer:")
print(final_answer)
