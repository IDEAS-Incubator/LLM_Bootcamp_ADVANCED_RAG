import os
# Set USER_AGENT environment variable
os.environ["USER_AGENT"] = "MyCustomUserAgent/1.0"

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import bs4

# ---- STEP 1: LOAD & INDEX ----

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

embedding_model = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)
retriever = vectorstore.as_retriever()

# ---- STEP 2: MULTI-QUERY GENERATION ----

multi_query_template = """
You are an AI assistant. Given a question, generate 3 different perspectives or aspects to explore.
Each perspective should be a complete question that helps understand a different aspect of the original question.
Separate each question with a newline.

Original question: {question}

Perspective questions:"""

prompt = PromptTemplate.from_template(multi_query_template)
llm = OllamaLLM(model="llama3.2")

generate_perspectives_chain = (
    {"question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ---- STEP 3: RAG ANSWERING ----

rag_prompt = PromptTemplate.from_template("""
Answer the question based on the following context.

Context:
{context}

Question: {question}
""")

format_docs = lambda docs: "\n\n".join([doc.page_content for doc in docs])

def retrieve_with_perspectives(original_question):
    # Step 1: generate perspective questions
    perspectives_text = generate_perspectives_chain.invoke(original_question)
    perspective_questions = [q.strip() for q in perspectives_text.strip().split("\n") if q.strip()]
    
    # Add original question
    all_questions = [original_question] + perspective_questions
    
    print("\n=== Original Question ===")
    print(original_question)
    print("\n=== Generated Perspective Questions ===")
    for idx, q in enumerate(perspective_questions, 1):
        print(f"{idx}. {q}")

    # Step 2: retrieve docs for each question
    all_docs = []
    for q in all_questions:
        results = retriever.invoke(q)
        all_docs.extend(results)

    # Step 3: remove duplicates
    seen = set()
    unique_docs = []
    for doc in all_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)
    
    return unique_docs

# ---- STEP 4: RUN ----

question = "How do autonomous agents balance exploration and exploitation in their decision-making?"
relevant_docs = retrieve_with_perspectives(question)

# Format + LLM call
final_prompt = rag_prompt.format(context=format_docs(relevant_docs), question=question)
answer = llm.invoke(final_prompt)

print("\n=== Final Answer ===")
print(answer) 