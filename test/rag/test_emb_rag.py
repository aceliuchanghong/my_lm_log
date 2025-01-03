# https://python.langchain.com/docs/integrations/text_embedding/ollama/
import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from sentence_transformers import util


def emb_docs(embeddings, docs):

    vectors = embeddings.embed_documents(docs)
    # for vector in vectors:
    #     print(len(vector))
    return vectors


def emb_query(embeddings, query):
    embedded_query = embeddings.embed_query(query)
    # print(len(embedded_query), embedded_query[:10])
    return embedded_query


def get_retrieve(embeddings, docs, query):
    # 初始化内存向量存储
    vectorstore = InMemoryVectorStore.from_texts(
        docs,
        embedding=embeddings,
    )
    # 检索器
    retriever = vectorstore.as_retriever()
    retrieved_documents = retriever.invoke(query)

    # 打印第一个检索结果
    # print(len(retrieved_documents), retrieved_documents[0].page_content)

    return retrieved_documents


if __name__ == "__main__":
    # export no_proxy="localhost,112.48.199.202,127.0.0.1"
    # python test/rag/test_emb_rag.py
    load_dotenv()
    docs = [
        "Alpha is the first letter of Greek alphabet",
        "Beta is the second letter of Greek alphabet",
        "Gamma is the third letter of Greek alphabet",
        "QAQ is the 4th letter of Greek alphabet",
        "QCQ is the 5th letter of Greek alphabet",
        "QDQ is the 6th letter of Greek alphabet",
        "QFG is the 7th letter of Greek alphabet",
    ]
    query = "What is the 5th letter of Greek alphabet"
    embeddings = OllamaEmbeddings(
        model=os.getenv("EMB_MODEL"), base_url=os.getenv("EMB_BASE_URL")
    )

    docs_vector = emb_docs(embeddings, docs)
    query_vector = emb_query(embeddings, query)
    print(f"vec:{query_vector[:5]}, dim:{len(query_vector)}")
    similarity = util.cos_sim(query_vector, docs_vector)
    similarity_list = similarity.squeeze().tolist()
    print("Similarity:", similarity_list)

    # text_new = get_retrieve(embeddings, docs, query)
    # print(text_new)
