import os


os.environ["USER_AGENT"] = "MyCustomUserAgent/1.0"

from typing import TypedDict, List
from pprint import pprint
import bs4

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

# ---- STEP 1: Load, Split, Embed ----


"""
RAG Agent

The pipeline follows these steps:

Route Question: Decide whether to retrieve from the vectorstore or simulate a web search based on the user's query.
Retrieve Documents: Retrieve relevant documents from a vectorstore.
Grade Documents: Evaluate whether the retrieved documents are relevant to the user's question.
Decide to Generate or Web Search:
If the documents are sufficient, proceed to generate an answer.
If the documents are insufficient, simulate a web search.
Generate Answer: Use the retrieved context to generate an answer.
Validate Answer:
Check if the answer is supported by the context (hallucination check).
Evaluate if the answer is useful to the user's question.
Iterative Improvement:
If the answer is unsupported, regenerate it.
If the answer is not useful, simulate a web search for additional context.

"""
# ---- STEP 1: Load, Split & Embed ----

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

splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=0)
splits = splitter.split_documents(docs)

embedding = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(splits, embedding=embedding)
retriever = vectorstore.as_retriever()

# ---- STEP 2: LLM Setup ----

llm = OllamaLLM(model="llama3.2", temperature=0)
parser = StrOutputParser()

# ---- STEP 3: Prompts ----

generate_prompt = ChatPromptTemplate.from_template("""
Use the following context to answer the user's question.

Context:
{context}

Question:
{question}
""")
generate_chain = generate_prompt | llm | parser

grade_doc_prompt = PromptTemplate.from_template("""
Does the following document help answer the question?

Document:
{document}

Question:
{question}

Respond with 'yes' or 'no' only.
""")

# ---- STEP 4: LangGraph State ----

class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    web_search: str
    fallback_used: bool

# ---- STEP 5: Graph Nodes ----

def retrieve(state: GraphState):
    docs = retriever.invoke(state["question"])
    return {"documents": docs, "question": state["question"], "fallback_used": False}

def grade_documents(state: GraphState):
    relevant_docs = []
    for doc in state["documents"]:
        result = llm.invoke(grade_doc_prompt.format(document=doc.page_content, question=state["question"]))
        if "yes" in result.lower():
            relevant_docs.append(doc)
    web_search_flag = "Yes" if not relevant_docs else "No"
    return {
        "documents": relevant_docs,
        "question": state["question"],
        "web_search": web_search_flag,
        "fallback_used": False
    }

def decide_to_generate(state: GraphState):
    return "websearch" if state["web_search"] == "Yes" else "generate"

def web_search(state: GraphState):
    print(" Simulated web search used.")
    fake_result = Document(page_content=f"Simulated web content for: {state['question']}")
    # return {"documents": [fake_result], "question": state["question"]}
    return "fallback" if state["web_search"] == "Yes" else "generate"


def generate(state: GraphState):
    context = "\n\n".join(doc.page_content for doc in state["documents"])
    answer = generate_chain.invoke({"question": state["question"], "context": context})
    return {
        "question": state["question"],
        "documents": state["documents"],
        "generation": answer,
        "fallback_used": False
    }

def fallback(state: GraphState):
    print("🤖 Using LLM to answer directly (fallback mode).")
    answer = llm.invoke(f"Answer the following using your own knowledge:\n{state['question']}")
    return {
        "question": state["question"],
        "documents": [],
        "generation": answer,
        "fallback_used": True
    }

# ---- STEP 6: Build LangGraph ----

graph = StateGraph(GraphState)

graph.add_node("retrieve", retrieve)
graph.add_node("grade_documents", grade_documents)
graph.add_node("generate", generate)
graph.add_node("fallback", fallback)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "grade_documents")

graph.add_conditional_edges("grade_documents", decide_to_generate, {
    "generate": "generate",
    "fallback": "fallback"
})

graph.add_edge("generate", END)
graph.add_edge("fallback", END)

app = graph.compile()

# ---- STEP 7: Run the Agent ----

question = "What are the different types of agent memory?" 

inputs = {
    "question": question,
    "fallback_used": False
}

print("\n=== 🧠 Running LLAMA 3 RAG Agent with Fallback ===")
for step in app.stream(inputs):
    for node, value in step.items():
        print(f"\n Node: {node}")
        pprint(value)
        print("\n---")

print("\n Final Answer:")
pprint(value.get("generation"))
