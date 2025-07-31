# rag_tools/setup_vectorstore.py


import pickle
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from langchain.schema import BaseRetriever
from langchain.docstore.document import Document
from typing import List
import os

# Load pre-chunked documents
with open("rag_tools/chunks.pkl", "rb") as f:
    chunks: List[Document] = pickle.load(f)

# Load vector DB from disk
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.load_local(
    folder_path="rag_tools/rag_jobs_db",
    embeddings=embedder,
    allow_dangerous_deserialization=True
)

# Create BM25 index
tokenized = [doc.page_content.split() for doc in chunks]
bm25 = BM25Okapi(tokenized)

# Define Hybrid Retriever
class HybridRetriever(BaseRetriever):
    chunks: List[Document]
    db: FAISS
    bm25: BM25Okapi
    k: int = 10

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        dense_docs = self.db.as_retriever(search_kwargs={"k": self.k}).get_relevant_documents(query)
        kw_docs = self.bm25_retrieve(query, top_k=self.k)
        unique = {doc.page_content: doc for doc in dense_docs + kw_docs}
        return list(unique.values())[:self.k]

    def bm25_retrieve(self, query: str, top_k: int = 10) -> List[Document]:
        scores = self.bm25.get_scores(query.split())
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.chunks[i] for i in top_idx]

# Instantiate Hybrid Retriever
hybrid_retriever = HybridRetriever(chunks=chunks, db=db, bm25=bm25, k=20)

