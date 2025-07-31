# rag_tools/chunk_documents.py

import os
import pandas as pd
import pickle
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


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

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Save chunks as pickle
with open("rag_tools/chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("✅ Chunks saved to rag_tools/chunks.pkl")


embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedder)
save_path = os.path.abspath("rag_tools/rag_it_jobs_db")
print("Saving vectorstore to:", save_path)
vectorstore.save_local(save_path)


