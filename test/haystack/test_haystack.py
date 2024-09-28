# 之后再看
# https://haystack.deepset.ai/tutorials
from haystack.document_stores.in_memory import InMemoryDocumentStore
from datasets import load_dataset
from haystack import Document

from haystack.components.embedders import SentenceTransformersDocumentEmbedder

document_store = InMemoryDocumentStore()

dataset = load_dataset("./data", split="train")
docs = [Document(content=doc["content"], meta=doc["meta"]) for doc in dataset]
print(docs[0])
doc_embedder = SentenceTransformersDocumentEmbedder(model=r"C:\Users\lawrence\Documents\llm\jina_emb_v3")
doc_embedder.warm_up()
