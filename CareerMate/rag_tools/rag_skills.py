# rag_tools/rag_skills.py

from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.schema import BaseRetriever
from openai import AsyncOpenAI
from typing import List
from pydantic import BaseModel
import os

# --- NEW IMPORTS ---
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from agents import function_tool

# --- Load RAG components ---
from rag_tools.setup_vectorstore import hybrid_retriever
from rag_tools.llm_loader import llm


# --- Define prompt for skill extraction (simplified) ---
qa_prompt_for_skills = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant for extracting technical job skills from job descriptions.\n"
     "Given the following context, extract and list the key required skills for the job title below.\n"
     "- Use only the content in the context.\n"
     "- Return a bullet-point list of skills only.\n"
     "- Do NOT make up any skill not explicitly mentioned.\n\n"
     "Context:\n{context}"
    ),
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
    
    # 1. Create a chain to combine the documents into a single prompt
    #    This part no longer expects 'chat_history'
    document_chain = create_stuff_documents_chain(llm, qa_prompt_for_skills)
    
    # 2. Create the full RAG chain that first retrieves documents, then passes them
    #    to the document_chain.
    rag_chain = create_retrieval_chain(retriever, document_chain)

    # Invoke the chain using the correct asynchronous method 'ainvoke'
    # The input key is "input"
    response = await rag_chain.ainvoke({"input": query})
    
    # The result is now in response["answer"]
    skills_text = response["answer"]
    
    # Process the extracted skills
    skills = skills_text.split("\n")
    cleaned_skills = [s.strip("•- ").strip() for s in skills if s.strip()]
    return SkillGapResult(missing_skills=cleaned_skills)

