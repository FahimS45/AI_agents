# rag_tools/setup_vectorstore.py

import os
import pandas as pd
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import BaseRetriever
from rank_bm25 import BM25Okapi
from typing import List
from dotenv import load_dotenv

# Load env in case needed
load_dotenv()

# ================================
# Step 1: Load and preprocess data
# ================================

df = pd.read_csv("IT_jobs.csv")  # Change path if needed

df["combined"] = (
    "Job Title: " + df["designation"].fillna("") + "\n"
    "Job Type: " + df["work_type"].fillna("N/A") + "\n"
    "Involvement: " + df["involvement"].fillna("N/A") + "\n"
    "Industry: " + df["industry"].fillna("N/A") + "\n"
    "Level: " + df["level"].fillna("N/A") + "\n"
    "Location: " + df["City"].fillna("") + ", " + df["State"].fillna("") + "\n"
    "Job Description: " + df["job_details"].fillna("")
)

# ================================
# Step 2: Create Documents
# ================================

docs = []

for _, row in df.iterrows():
    text = f"""Job Title: {row["designation"]}
Job Type: {row["work_type"]}
Involvement: {row["involvement"]}
Location: {row["City"]}, {row["State"]}
Job Description: {row["job_details"]}
"""
    meta = {
        "designation": row["designation"],
        "location": f"{row['City']}, {row['State']}",
        "work_type": row["work_type"],
        "involvement": row["involvement"],
    }
    docs.append(Document(page_content=text, metadata=meta))

# ================================
# Step 3: Chunking
# ================================

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# ================================
# Step 4: Embeddings and Vector DB
# ================================

embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedder)
vectorstore.save_local("rag_jobs_db")

# ================================
# Step 5: Hybrid Retriever Setup
# ================================

# Load the DB
db = FAISS.load_local("rag_jobs_db", embedder, allow_dangerous_deserialization=True)

# Tokenize for BM25
tokenized = [chunk.page_content.split() for chunk in chunks]
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

print("✅ Hybrid retriever and vector DB are ready.")
