# rag_tools/rag_skills.py

from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.schema import BaseRetriever
from openai import AsyncOpenAI
from typing import List
from agents import function_tool
from pydantic import BaseModel
import os

# --- Load RAG components ---
from rag_tools.setup_vectorstore import hybrid_retriever
from rag_tools.llm_loader import llm


# --- Define prompt for skill extraction ---
qa_prompt_for_skills = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant for extracting technical job skills from job descriptions.\n"
     "Given the following context, extract and list the key required skills for the job title below.\n"
     "- Use only the content in the context.\n"
     "- Return a bullet-point list of skills only.\n"
     "- Do NOT make up any skill not explicitly mentioned.\n\n"
     "Context:\n{context}"
    ),
    MessagesPlaceholder("chat_history"),
    ("user", "{input}"),
])

# --- Output schema ---
class SkillGapResult(BaseModel):
    missing_skills: List[str]

# --- RAG Skill Extraction Tool ---
@function_tool
async def get_required_skills_with_rag(job_title: str) -> SkillGapResult:
    """
    Use RAG to extract required skills for a given job title from real job postings.
    """
    query = f"What are the required skills for a {job_title}?"
    retriever = hybrid_retriever  # already loaded globally
    retriever_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": qa_prompt_for_skills}
    )

    response = await retriever_chain.ainvoke({"query": query})

    skills = response["result"].split("\n")
    cleaned_skills = [s.strip("•- ").strip() for s in skills if s.strip()]
    return SkillGapResult(missing_skills=cleaned_skills)

print("✅ Loaded successfully!")
