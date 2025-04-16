
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from typing import TypedDict, List
from pprint import pprint
import bs4, os

# ---- STEP 1: Load, Split & Embed ----

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

loader = WebBaseLoader(web_paths=urls, bs_kwargs={"parse_only": bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))})
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=0)
splits = splitter.split_documents(docs)

embedding = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(splits, embedding=embedding)
retriever = vectorstore.as_retriever()

# ---- STEP 2: LLM & Prompts ----

llm = OllamaLLM(model="llama3.2", temperature=0)
parser = StrOutputParser()

# For generation
generate_prompt = ChatPromptTemplate.from_template("""
Use the following context to answer the user's question.

Context:
{context}

Question: {question}
""")
generate_chain = generate_prompt | llm | parser

# Graders
grade_doc_prompt = PromptTemplate.from_template("""
Does the following document help answer the question?

Document:
{document}

Question:
{question}

Respond with 'yes' or 'no' only.
""")

hallucination_prompt = PromptTemplate.from_template("""
Do the following context support the answer?

Context:
{context}

Answer:
{generation}

Respond with 'yes' or 'no' only.
""")

usefulness_prompt = PromptTemplate.from_template("""
Is the answer helpful to resolve the user's question?

Answer:
{generation}

Question:
{question}

Respond with 'yes' or 'no' only.
""")

# ---- STEP 3: LangGraph State ----

class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    web_search: str

# ---- STEP 4: Graph Nodes ----

def route_question(state: GraphState):
    if "today" in state["question"].lower() or "latest" in state["question"].lower():
        return "websearch"
    return "retrieve"

def retrieve(state: GraphState):
    docs = retriever.invoke(state["question"])
    return {"documents": docs, "question": state["question"]}

def grade_documents(state: GraphState):
    relevant_docs = []
    web_search_flag = "No"
    for doc in state["documents"]:
        result = llm.invoke(grade_doc_prompt.format(document=doc.page_content, question=state["question"]))
        if "yes" in result.lower():
            relevant_docs.append(doc)
        else:
            web_search_flag = "Yes"
    return {"documents": relevant_docs, "question": state["question"], "web_search": web_search_flag}

def decide_to_generate(state: GraphState):
    return "websearch" if state["web_search"] == "Yes" else "generate"

def web_search(state: GraphState):
    print("🌐 Simulated web search used.")
    fake_result = Document(page_content=f"Simulated web content for: {state['question']}")
    return {"documents": [fake_result], "question": state["question"]}

def generate(state: GraphState):
    context = "\n\n".join(doc.page_content for doc in state["documents"])
    answer = generate_chain.invoke({"question": state["question"], "context": context})
    return {"question": state["question"], "documents": state["documents"], "generation": answer}

def grade_generation_v_documents_and_question(state: GraphState):
    context = "\n\n".join(doc.page_content for doc in state["documents"])
    answer = state["generation"]
    
    hallucinated = llm.invoke(hallucination_prompt.format(context=context, generation=answer)).strip().lower()
    if "no" in hallucinated:
        return "not supported"
    
    useful = llm.invoke(usefulness_prompt.format(generation=answer, question=state["question"])).strip().lower()
    return "useful" if "yes" in useful else "not useful"

# ---- STEP 5: LangGraph Construction ----

graph = StateGraph(GraphState)

graph.add_node("retrieve", retrieve)
graph.add_node("grade_documents", grade_documents)
graph.add_node("websearch", web_search)
graph.add_node("generate", generate)

graph.set_conditional_entry_point(route_question, {
    "websearch": "websearch",
    "retrieve": "retrieve"
})

graph.add_edge("retrieve", "grade_documents")

graph.add_conditional_edges("grade_documents", decide_to_generate, {
    "websearch": "websearch",
    "generate": "generate"
})

graph.add_edge("websearch", "generate")

graph.add_conditional_edges("generate", grade_generation_v_documents_and_question, {
    "useful": END,
    "not supported": "generate",
    "not useful": "websearch"
})

app = graph.compile()

# ---- STEP 6: Run the Agent ----

inputs = {"question": "What are the different types of agent memory?"}

print("\n=== 🧠 Running LLAMA 3 RAG Agent Locally ===")
for step in app.stream(inputs):
    for node, value in step.items():
        print(f"\n🧩 Node: {node}")
        pprint(value)
        print("\n---")

print("\n✅ Final Answer:")
pprint(value.get("generation"))
