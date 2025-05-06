import os
# Set USER_AGENT environment variable
os.environ["USER_AGENT"] = "MyCustomUserAgent/1.0"

# Set GRAPH_RECURSION_LIMIT to avoid recursion limit errors
os.environ["GRAPH_RECURSION_LIMIT"] = "60"


from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from pprint import pprint
import bs4, os

"""
Self-Reflective RAG
Step-by-Step Flow:
1.	User Query → Retrieve top-k docs → Generate initial answer
2.	Self-Reflection Prompt: 
    Ask LLM to analyze its own answer:
        o	“Was the answer complete?”
        o	“Did I use evidence from the documents?”
        o	“Was anything missing or vague ?”
3.	If reflection finds gaps ( anything worng or missing):
    o	Reformulate the query
    o	Trigger secondary retrieval
    o	Generate improved answer
4.	Return final verified answer

"""
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

text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
doc_splits = text_splitter.split_documents(docs)

embedding_model = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(doc_splits, embedding=embedding_model)
retriever = vectorstore.as_retriever()

llm = OllamaLLM(model="llama3.2")

# ---- Prompt templates ----

"""
Self-Reflection Prompt: 
    Ask LLM to analyze its own answer:
        o	“content is relevant to a user question ?”
        o	“whether the answer makes sense given the question and the retrieved context ?”
"""

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
    print("\nRetrieved context.")
    return {"question": state["question"], "context": context}

def grade_documents(state):
    prompt = grade_doc_prompt.format(context=state["context"], question=state["question"])
    result = llm.invoke(prompt).lower()
    print("\nDocument Grading Result:", result)
    if "yes" in result:
        return {"grade_documents": "generate"}
    return {"grade_documents": "transform_query"}

def generate(state):
    prompt = rag_prompt.format(context=state["context"], question=state["question"])
    answer = llm.invoke(prompt)
    print("\nGenerated Answer:\n", answer)
    return {"generation": answer, "context": state["context"], "question": state["question"]}

def transform_query(state):
    print("\nRephrasing query (looping back to retrieve)...")
    return {"question": state["question"]}  # In a real setup, you'd transform it

def grade_generation_v_documents_and_question(state):
    prompt = grade_generation_prompt.format(
        context=state["context"],
        generation=state["generation"],
        question=state["question"]
    )
    result = llm.invoke(prompt).lower()
    print("\nSelf-Reflection Result:", result)

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

# ---- Run LangGraph Workflow ----

question = "Explain how the different types of agent memory work ? "

# question = "Explain how the different types of deep learning network works ?"

inputs = {"question": question}

print("\n=== Running Self-Reflective RAG for Question ===")
print(question)
print("===============================================")

"""
# orginal code to run the workflow
for step in app.stream(inputs):
    for node, value in step.items():
        pprint(f"Node '{node}':")
    print("\n---\n")

"""

"""
# Fix - 1 exception handling for recursion limit
try:
    for step in app.stream(inputs):
        for node, value in step.items():
            pprint(f"Node '{node}':")
        print("\n---\n")
except RecursionError as e:
    print("\n RecursionError: The graph recursion limit was exceeded.")
    print("Consider increasing the GRAPH_RECURSION_LIMIT environment variable.")
    print(f"Error details: {e}")
"""

# Fix -2 check relevance of the user question before running the workflow
relevance_prompt = PromptTemplate.from_template("""
You are a relevance checker. Determine if the following question is related to the knowledge base.

Knowledge Base Topics:
- Agent memory
- Prompt engineering
- Adversarial attacks on LLMs

Question:
{question}

Reply with "relevant" or "irrelevant".
""")

def check_relevance(question):
    print("\n Checking question relevance...")
    relevance = llm.invoke(relevance_prompt.format(question=question)).strip().lower()
    print(f"Relevance Check Result: {relevance}")
    return relevance == "relevant"


# Check if the question is relevant to the knowledge base
if not check_relevance(question):
    print("\n The question is not relevant to the knowledge base. Bypassing RAG.")
    print("Please ask a question related to agent memory, prompt engineering, or adversarial attacks on LLMs.")
    # todo: query at LLM instead of using RAG
else:
    try:
        for step in app.stream(inputs):
            for node, value in step.items():
                pprint(f"Node '{node}':")
            print("\n---\n")
    except RecursionError as e:
        print("\n RecursionError: The graph recursion limit was exceeded.")
        print("Consider increasing the GRAPH_RECURSION_LIMIT environment variable.")
        print(f"Error details: {e}")

    # Final output
    print("Final Generation:")
    print(value.get("generation"))


