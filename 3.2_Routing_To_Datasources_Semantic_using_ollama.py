import os
# Set USER_AGENT environment variable
os.environ["USER_AGENT"] = "MyCustomUserAgent/1.0"

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Step 1: Define your prompts
physics_template = """You are a very smart physics professor. \
You are great at answering questions about physics in a concise and easy-to-understand manner. \
When you don't know the answer to a question, you admit that you don't know.

Here is a question:
{query}"""

math_template = """You are a very good mathematician. You are great at answering math questions. \
You are so good because you are able to break down hard problems into their component parts, \
answer the component parts, and then put them together to answer the broader question.

Here is a question:
{query}"""

generic_template = """You are a helpful AI assistant. You are great at answering questions in a concise and easy-to-understand manner. \
When you don't know the answer to a question, you admit that you don't know.

Here is a question:
{query}"""

prompt_templates = [physics_template, math_template, generic_template]

# Step 2: Embed the prompts
embedding_model = OllamaEmbeddings(model="nomic-embed-text")
prompt_embeddings = embedding_model.embed_documents(prompt_templates)

# Step 3: Create router logic with confidence scores
def prompt_router(input):
    query_embedding = embedding_model.embed_query(input["query"])
    similarity_scores = cosine_similarity([query_embedding], prompt_embeddings)[0]
    best_index = np.argmax(similarity_scores)
    confidence = similarity_scores[best_index]
    
    print("\n=== Prompt Routing Decision ===")
    if confidence < 0.5:
        print(f"Confidence too low ({confidence:.2f}), using GENERIC prompt")
        best_index = 2  # Use generic prompt
    elif best_index == 0:
        print(f"Using PHYSICS prompt (Confidence: {confidence:.2f})")
    elif best_index == 1:
        print(f"Using MATH prompt (Confidence: {confidence:.2f})")
    
    return PromptTemplate.from_template(prompt_templates[best_index]).format(
        query=input["query"]
    )

# Step 4: Create the full chain
llm = OllamaLLM(model="llama3.2")

chain = (
    {"query": RunnablePassthrough()}
    | RunnableLambda(prompt_router)
    | llm
    | StrOutputParser()
)

# Step 5: Ask something!
questions = [
    "What is the speed of light?",
    "What is the derivative of x^2?",
    "How does quantum entanglement work?",
    "What is the Pythagorean theorem?",
    "What is the meaning of life?",  # This should use generic prompt
    "Tell me about the history of art"  # This should use generic prompt
]

print("\n=== Testing Semantic Router with Different Questions ===")
for question in questions:
    print("\n=== User Question ===")
    print(question)
    
    response = chain.invoke(question)
    
    print("\n=== Final Answer ===")
    print(response)
    print("-" * 80) 