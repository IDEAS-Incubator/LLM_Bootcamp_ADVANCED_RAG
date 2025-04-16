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

prompt_templates = [physics_template, math_template]

# Step 2: Embed the prompts
embedding_model = OllamaEmbeddings(model="nomic-embed-text")
prompt_embeddings = embedding_model.embed_documents(prompt_templates)

# Step 3: Create router logic
def prompt_router(input):
    query_embedding = embedding_model.embed_query(input["query"])
    similarity_scores = cosine_similarity([query_embedding], prompt_embeddings)[0]
    best_index = np.argmax(similarity_scores)
    chosen_prompt = prompt_templates[best_index]

    print("\n=== Prompt Routing Decision ===")
    print("Using MATH prompt" if best_index == 1 else "Using PHYSICS prompt")
    return PromptTemplate.from_template(chosen_prompt)

# Step 4: Create the full chain
llm = OllamaLLM(model="llama3.2")

chain = (
    {"query": RunnablePassthrough()}
    | RunnableLambda(prompt_router)
    | llm
    | StrOutputParser()
)

# Step 5: Ask something!
question = "What's a black hole?"
print("\n=== User Question ===")
print(question)

response = chain.invoke(question)

print("\n=== Final Answer ===")
print(response)
