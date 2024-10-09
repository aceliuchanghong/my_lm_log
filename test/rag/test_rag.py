# https://python.langchain.com/docs/integrations/text_embedding/ollama/
from openai import OpenAI
import os
from dotenv import load_dotenv
# pip install -qU langchain-ollama
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

load_dotenv()

llm = OpenAI(api_key=os.getenv('API_KEY'), base_url=os.getenv('BASE_URL'))
embeddings = OllamaEmbeddings(model=os.getenv('EMB_MODEL'), base_url=os.getenv('EMB_BASE_URL'))

doc = [
    "Alpha is the first letter of Greek alphabet",
    "Beta is the second letter of Greek alphabet",
    "Gamma is the third letter of Greek alphabet",
    "QAQ is the 4th letter of Greek alphabet",
    "QCQ is the 5th letter of Greek alphabet",
    "QDQ is the 6th letter of Greek alphabet",
    "QFG is the 7th letter of Greek alphabet",
]
query = "What is the second letter of Greek alphabet"

two_vectors = embeddings.embed_documents(
    doc
)
embedded_query = embeddings.embed_query(query)

vectorstore = InMemoryVectorStore.from_texts(
    doc,
    embedding=embeddings,
)
vectorstore.asearch(query, search_type='mmr', k=2)
vectorstore.similarity_search(query, k=2)

# Use the vectorstore as a retriever
retriever = vectorstore.as_retriever()

# Retrieve the most similar text
retrieved_documents = retriever.invoke(query)

# show the retrieved document's content
print(retrieved_documents[0].page_content)
