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
from typing import TypedDict
from langchain_core.messages import HumanMessage
import bs4, os
from pprint import pprint

"""
Adaptive RAG

The pipeline follows these steps:

Router Decision: Based on the user's question, decide whether to retrieve information from the vectorstore or perform a web search.
Retrieve Context:
If the router selects "vectorstore," retrieve relevant documents from the pre-indexed vector database.
If the router selects "web_search," simulate a web search to retrieve external information.
Generate Answer: Use the retrieved context to generate an answer to the user's question.
Output Answer: Return the final answer to the user.

"""

# ---- STEP 1: Load & Split Documents for Vectorstore ----

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
]

loader = WebBaseLoader(
    web_paths=urls,
    bs_kwargs={"parse_only": bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))}
)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = splitter.split_documents(docs)

embedding = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(splits, embedding=embedding)
retriever = vectorstore.as_retriever()

# Simulated “web search” content
def fake_web_search(query: str):
    return f"Simulated web result for query: '{query}' — [Current events or external info would go here.]"

# ---- STEP 2: Router (plain text version) ----

llm = OllamaLLM(model="llama3.2", temperature=0)

router_prompt = ChatPromptTemplate.from_template("""
You are a smart router. Choose one of the following based on the user question:
- "vectorstore"
- "web_search"

Question: {question}

Answer with just one word.
""")

router_chain = (
    {"question": lambda x: x["question"]}
    | router_prompt
    | llm
    | StrOutputParser()
)

# ---- STEP 3: Answer Generator ----

answer_prompt = ChatPromptTemplate.from_template("""
Use the following context to answer the user's question.

Context:
{context}

Question:
{question}
""")

generate_chain = (
    {"question": lambda x: x["question"], "context": lambda x: x["context"]}
    | answer_prompt
    | llm
    | StrOutputParser()
)

# ---- STEP 4: LangGraph Structure ----

class AgentState(TypedDict):
    question: str
    context: str
    generation: str

def route_question(state: AgentState):
    route = router_chain.invoke({"question": state["question"]}).strip().lower()
    print("🧭 Routing decision:", route)
    return "web_search" if "web" in route else "vectorstore"

def retrieve_vectorstore(state: AgentState):
    docs = retriever.invoke(state["question"])
    context = "\n\n".join(doc.page_content for doc in docs)
    return {"question": state["question"], "context": context}

def retrieve_web(state: AgentState):
    context = fake_web_search(state["question"])
    return {"question": state["question"], "context": context}

def generate_answer(state: AgentState):
    answer = generate_chain.invoke({"question": state["question"], "context": state["context"]})
    return {"question": state["question"], "context": state["context"], "generation": answer}

# ---- STEP 5: Build Graph ----

graph = StateGraph(AgentState)

graph.add_node("retrieve_vectorstore", retrieve_vectorstore)
graph.add_node("retrieve_web", retrieve_web)
graph.add_node("generate", generate_answer)

graph.set_conditional_entry_point(route_question, {
    "vectorstore": "retrieve_vectorstore",
    "web_search": "retrieve_web"
})

graph.add_edge("retrieve_vectorstore", "generate")
graph.add_edge("retrieve_web", "generate")
graph.add_edge("generate", END)

app = graph.compile()

# ---- STEP 6: Run it ----

question = "What's the latest on AI alignment conferences this year?"

print("\n=== Running Adaptive RAG Router ===")
for step in app.stream({"question": question}):
    for node, output in step.items():
        print(f"\n Node: {node}")
        pprint(output)
        print("\n---")

print("\nFinal Answer:")
pprint(output.get("generation"))
