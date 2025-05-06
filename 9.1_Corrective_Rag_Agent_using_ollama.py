
import os
# Set USER_AGENT environment variable
os.environ["USER_AGENT"] = "MyCustomUserAgent/1.0"

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from typing import TypedDict, List
from pprint import pprint
import bs4
import os

"""

Corrective RAG

The pipeline follows these steps:

Retrieve Documents: Retrieve relevant documents from a vectorstore.
Grade Documents: Evaluate whether the retrieved documents are relevant to the user's question.
Decide to Generate or Rewrite:
If the documents are sufficient, proceed to generate an answer.
If the documents are insufficient, rewrite the query and simulate a web search.
Simulate Web Search: Retrieve external information if the query is rewritten.
Generate Answer: Use the retrieved context to generate an answer.
Output Answer: Return the final answer to the user.

"""
# ---- STEP 1: Load and Index Documents ----
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

loader = WebBaseLoader(
    web_paths=urls,
    bs_kwargs={"parse_only": bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))}
)

docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)
split_docs = splitter.split_documents(docs)

embedding = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(split_docs, embedding=embedding)
retriever = vectorstore.as_retriever()

# ---- STEP 2: LLM Setup ----
llm = OllamaLLM(model="llama3.2", temperature=0)

# ---- STEP 3: Prompt Templates ----

grade_prompt = ChatPromptTemplate.from_template("""
You are a grader checking if the retrieved document is relevant to the user question.

Context:
{document}

Question:
{question}

Answer with only: "yes" or "no", and briefly explain why.
""")

rewrite_prompt = ChatPromptTemplate.from_template("""
You are a question rewriter optimizing queries for web-like search.

Original question:
{question}

Rewrite it to make it clearer or more specific.
""")

generate_prompt = ChatPromptTemplate.from_template("""
Use the following context to answer the user's question.

Context:
{context}

Question:
{question}
""")

rag_chain = generate_prompt | llm | StrOutputParser()

# ---- STEP 4: LangGraph State ----

class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    web_search: str

# ---- STEP 5: Graph Nodes ----

def retrieve(state: GraphState):
    print("\n Retrieving documents...")
    docs = retriever.invoke(state["question"])
    return {"documents": docs, "question": state["question"]}

def grade_documents(state: GraphState):
    print("\n Grading documents...")
    filtered_docs = []
    web_search_flag = "No"
    for doc in state["documents"]:
        response = llm.invoke(
            grade_prompt.format(document=doc.page_content, question=state["question"])
        ).lower()
        if "yes" in response:
            print(" Relevant document found.")
            filtered_docs.append(doc)
        else:
            print(" Irrelevant document.")
            web_search_flag = "Yes"
    return {"documents": filtered_docs, "question": state["question"], "web_search": web_search_flag}

def decide_to_generate(state: GraphState):
    print("\n Decision node:")
    if state["web_search"] == "Yes":
        print(" Not enough relevant context. Rewriting query...")
        return "transform_query"
    else:
        print(" Context sufficient. Proceeding to generation.")
        return "generate"

def transform_query(state: GraphState):
    print("\n Rewriting query...")
    better_question = llm.invoke(rewrite_prompt.format(question=state["question"]))
    return {"question": better_question, "documents": []}

def web_search(state: GraphState):
    print("\n Simulating web search...")
    fake_doc = Document(page_content=f"[Simulated web content for]: {state['question']}")
    return {"documents": [fake_doc], "question": state["question"]}

def generate(state: GraphState):
    print("\n Generating answer...")
    context = "\n\n".join([doc.page_content for doc in state["documents"]])
    generation = rag_chain.invoke({"context": context, "question": state["question"]})
    return {"question": state["question"], "documents": state["documents"], "generation": generation}

# ---- STEP 6: Build LangGraph ----

graph = StateGraph(GraphState)

graph.add_node("retrieve", retrieve)
graph.add_node("grade_documents", grade_documents)
graph.add_node("generate", generate)
graph.add_node("transform_query", transform_query)
graph.add_node("web_search_node", web_search)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "grade_documents")
graph.add_conditional_edges("grade_documents", decide_to_generate, {
    "generate": "generate",
    "transform_query": "transform_query",
})
graph.add_edge("transform_query", "web_search_node")
graph.add_edge("web_search_node", "generate")
graph.add_edge("generate", END)

app = graph.compile()

# ---- STEP 7: Run Example ----

question = "What are the different types of agent memory?"
inputs = {"question": question}

print("\n=== Running Corrective RAG Agent ===")

for step in app.stream(inputs):
    for node, val in step.items():
        print(f"\n Node: {node}")
        pprint(val)
        print("\n---")

print("\n Final Answer:")
pprint(val.get("generation"))
