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

# ---- STEP 2: HYPOTHETICAL DOCUMENT EMBEDDINGS (HyDE) ----

hyde_template = """
You are an AI assistant. Given a question, write a hypothetical document that would contain the answer.
The document should be informative and directly relevant to the question.

Question: {question}

Hypothetical document:"""

prompt = PromptTemplate.from_template(hyde_template)
llm = OllamaLLM(model="llama3.2")

generate_hypothetical_doc_chain = (
    {"question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ---- STEP 3: RAG ANSWERING ----

rag_prompt = PromptTemplate.from_template("""
You are a knowledgeable assistant.

Use the following context to answer the question. If the context is missing, unclear, or doesn't directly answer the question, use your own understanding to provide a complete and helpful answer.

Be clear, concise, and avoid over-explaining.

Context:
{context}

Question: {question}
""")

format_docs = lambda docs: "\n\n".join([doc.page_content for doc in docs])

def retrieve_with_hyde(original_question):
    # Step 1: generate hypothetical document
    hypothetical_doc = generate_hypothetical_doc_chain.invoke(original_question)
    
    print("\n=== Original Question ===")
    print(original_question)
    print("\n=== Generated Hypothetical Document ===")
    print(hypothetical_doc)

    # Step 2: get embedding for hypothetical document
    hypothetical_embedding = embedding_model.embed_query(hypothetical_doc)
    
    # Step 3: retrieve docs using similarity search
    results = vectorstore.similarity_search_by_vector(hypothetical_embedding, k=3)
    
    return results

# ---- STEP 4: RUN ----

question = "What are the key components of an autonomous agent's memory system?"
relevant_docs = retrieve_with_hyde(question)

# Format + LLM call
final_prompt = rag_prompt.format(context=format_docs(relevant_docs), question=question)
answer = llm.invoke(final_prompt)

print("\n=== Final Answer ===")
print(answer) 