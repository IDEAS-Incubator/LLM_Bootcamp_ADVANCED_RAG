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
from langchain.load import dumps, loads
import bs4
import numpy as np

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

# ---- STEP 2: RAG FUSION SETUP ----

# Query transformation templates
hyde_template = """
You are an AI assistant. Given a question, write a hypothetical document that would contain the answer.
The document should be informative and directly relevant to the question.

Question: {question}

Hypothetical document:"""

stepback_template = """
You are an AI assistant. Given a specific question, generate a more general, high-level question that would help understand the broader context.
The stepback question should focus on the underlying concepts and principles.

Specific question: {question}

Stepback question:"""

multi_query_template = """
You are an AI assistant. Given a question, generate 3 different perspectives or aspects to explore.
Each perspective should be a complete question that helps understand a different aspect of the original question.
Separate each question with a newline.

Original question: {question}

Perspective questions:"""

# Initialize LLM and chains
llm = OllamaLLM(model="llama3.2")

hyde_prompt = PromptTemplate.from_template(hyde_template)
stepback_prompt = PromptTemplate.from_template(stepback_template)
multi_query_prompt = PromptTemplate.from_template(multi_query_template)

generate_hypothetical_doc_chain = (
    {"question": RunnablePassthrough()}
    | hyde_prompt
    | llm
    | StrOutputParser()
)

generate_stepback_chain = (
    {"question": RunnablePassthrough()}
    | stepback_prompt
    | llm
    | StrOutputParser()
)

generate_perspectives_chain = (
    {"question": RunnablePassthrough()}
    | multi_query_prompt
    | llm
    | StrOutputParser()
)

# ---- STEP 3: RAG FUSION IMPLEMENTATION ----

def reciprocal_rank_fusion(results: list[list], k=60):
    """Reciprocal Rank Fusion that takes multiple lists of ranked documents 
    and an optional parameter k used in the RRF formula"""
    
    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    return reranked_results

rag_prompt = PromptTemplate.from_template("""
Answer the question based on the following context.

Context:
{context}

Question: {question}
""")

format_docs = lambda docs: "\n\n".join([doc.page_content for doc in docs])

def retrieve_with_rag_fusion(original_question):
    # Step 1: Generate different query variants
    hypothetical_doc = generate_hypothetical_doc_chain.invoke(original_question)
    stepback_question = generate_stepback_chain.invoke(original_question)
    perspectives_text = generate_perspectives_chain.invoke(original_question)
    perspective_questions = [q.strip() for q in perspectives_text.strip().split("\n") if q.strip()]
    
    # Combine all questions
    all_questions = [original_question, stepback_question] + perspective_questions
    
    print("\n=== Original Question ===")
    print(original_question)
    print("\n=== Generated Stepback Question ===")
    print(stepback_question)
    print("\n=== Generated Perspective Questions ===")
    for idx, q in enumerate(perspective_questions, 1):
        print(f"{idx}. {q}")
    print("\n=== Generated Hypothetical Document ===")
    print(hypothetical_doc)

    # Step 2: Retrieve documents using different methods
    # Method 1: Direct retrieval
    direct_docs = retriever.invoke(original_question)
    
    # Method 2: HyDE retrieval
    hypothetical_embedding = embedding_model.embed_query(hypothetical_doc)
    hyde_docs = vectorstore.similarity_search_by_vector(hypothetical_embedding, k=3)
    
    # Method 3: Multi-query retrieval
    multi_query_docs = []
    for q in all_questions:
        results = retriever.invoke(q)
        multi_query_docs.extend(results)
    
    # Step 3: Combine and rank documents using RRF
    all_results = [direct_docs, hyde_docs, multi_query_docs]
    reranked_docs = reciprocal_rank_fusion(all_results)
    
    # Extract just the documents from the reranked results
    final_docs = [doc for doc, score in reranked_docs]
    
    return final_docs

# ---- STEP 4: RUN ----

question = "What are the key components and challenges in building autonomous agents?"
relevant_docs = retrieve_with_rag_fusion(question)

# Format + LLM call
final_prompt = rag_prompt.format(context=format_docs(relevant_docs), question=question)
answer = llm.invoke(final_prompt)

print("\n=== Final Answer ===")
print(answer) 