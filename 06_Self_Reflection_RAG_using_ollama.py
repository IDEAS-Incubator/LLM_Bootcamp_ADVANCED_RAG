from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from pprint import pprint
import bs4, os


# ---- Load and index docs ----

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"
]

docs = []
for url in urls:
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header")))
    )
    docs.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=0)
doc_splits = text_splitter.split_documents(docs)

embedding_model = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(doc_splits, embedding=embedding_model)
retriever = vectorstore.as_retriever()

llm = OllamaLLM(model="llama3.2")

# ---- Prompt templates ----

grade_doc_prompt = PromptTemplate.from_template("""
You are a grader assessing whether retrieved content is relevant to a user question.
Context:
{context}
Question: {question}
Answer with "yes" or "no" and explain briefly.
""")

rag_prompt = PromptTemplate.from_template("""
Answer the question based on the following context.

Context:
{context}

Question: {question}
""")

grade_generation_prompt = PromptTemplate.from_template("""
You are a helpful assistant evaluating whether the answer makes sense given the question and the retrieved context.

Context:
{context}

Answer:
{generation}

Question:
{question}

Reply with one of:
- "useful"
- "not useful"
- "not supported"

Explain why.
""")

# ---- LangGraph State ----

class GraphState(dict):
    question: str
    context: str
    generation: str

# ---- Nodes ----

def retrieve(state):
    docs = retriever.invoke(state["question"])
    context = "\n\n".join([doc.page_content for doc in docs])
    print("\n📥 Retrieved context.")
    return {"question": state["question"], "context": context}

def grade_documents(state):
    prompt = grade_doc_prompt.format(context=state["context"], question=state["question"])
    result = llm.invoke(prompt).lower()
    print("\n🧪 Document Grading Result:", result)
    if "yes" in result:
        return {"grade_documents": "generate"}
    return {"grade_documents": "transform_query"}

def generate(state):
    prompt = rag_prompt.format(context=state["context"], question=state["question"])
    answer = llm.invoke(prompt)
    print("\n💬 Generated Answer:\n", answer)
    return {"generation": answer, "context": state["context"], "question": state["question"]}

def transform_query(state):
    print("\n🔁 Rephrasing query (looping back to retrieve)...")
    return {"question": state["question"]}  # In a real setup, you'd transform it

def grade_generation_v_documents_and_question(state):
    prompt = grade_generation_prompt.format(
        context=state["context"],
        generation=state["generation"],
        question=state["question"]
    )
    result = llm.invoke(prompt).lower()
    print("\n🧠 Self-Reflection Result:", result)

    if "not supported" in result:
        return "not supported"
    elif "not useful" in result:
        return "not useful"
    return "useful"

# ---- Build LangGraph ----

workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")

workflow.add_conditional_edges("grade_documents", lambda x: x["grade_documents"], {
    "generate": "generate",
    "transform_query": "transform_query",
})

workflow.add_edge("transform_query", "retrieve")

workflow.add_conditional_edges("generate", grade_generation_v_documents_and_question, {
    "useful": END,
    "not useful": "transform_query",
    "not supported": "generate"
})

app = workflow.compile()

# ---- Run It ----

question = "Explain how the different types of agent memory work?"
inputs = {"question": question}

print("\n=== Running Self-Reflective RAG for Question ===")
print(question)
print("===============================================")

for step in app.stream(inputs):
    for node, value in step.items():
        pprint(f"🧩 Node '{node}':")
    print("\n---\n")

# Final output
print("✅ Final Generation:")
print(value.get("generation"))
