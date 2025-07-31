# rag_tools/rag_jobs.py

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import RetrievalQA
from langchain_core.runnables import Runnable
from langchain.schema import BaseRetriever
from langchain_core.documents import Document
from openai import AsyncOpenAI
from typing import List, Optional
from pydantic import BaseModel
from agents import function_tool

from rag_tools.setup_vectorstore import hybrid_retriever
from rag_tools.llm_loader import llm


# ---------------- Prompt ----------------

qa_prompt_for_jobs = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant who matches job seekers to relevant job listings.\n"
     "Given the following context from job postings, extract only the top 3 most relevant job listings that match the following:\n"
     "- User skills\n"
     "- Preferred location (if given)\n"
     "- Job involvement (e.g. full-time or part-time)\n"
     "- Work type (e.g. remote, on-site, hybrid)\n\n"
     "Only use the job information from the context. Do NOT invent job titles or details.\n\n"
     "Format the output like:\n"
     "1. **Job Title** at Company (Location)\n"
     "   - Type: [Full-time/Part-time], [Remote/On-site/Hybrid]\n"
     "   - Requirements: [...]\n"
     "   - Description: ...\n\n"
     "   - Contact information: (include only if available...)\n\n"
     "Context:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("user", "{input}"),
])

# ---------------- Output Schema ----------------

class JobListing(BaseModel):
    title: str
    company: str
    location: str
    requirements: List[str]
    description: str
    contact: Optional[str] = None
    

# ---------------- RAG Job Search Tool ----------------

@function_tool
async def find_jobs_with_rag(
    skills: List[str],
    location: Optional[str] = None,
    involvement: Optional[str] = None,
    work_type: Optional[str] = None
) -> List[JobListing]:
    """
    Use RAG to search job listings that match the user's skills, location, involvement (e.g. full-time), and work type (e.g. remote).
    """
    # Construct natural language query
    query_parts = [f"Find jobs requiring: {', '.join(skills)}."]
    if location:
        query_parts.append(f"Location: {location}.")
    if involvement:
        query_parts.append(f"Involvement: {involvement}.")
    if work_type:
        query_parts.append(f"Work type: {work_type}.")
    query = " ".join(query_parts)

    retriever_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=hybrid_retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": qa_prompt_for_jobs}
    )

    response = await retriever_chain.ainvoke({"query": query})
    raw_result = response["result"]

    # Parse result into structured list
    job_blocks = raw_result.split("\n\n")
    job_list = []
    for block in job_blocks:
        lines = block.strip().split("\n")
        if not lines or len(lines) < 3:
            continue
        try:
            header = lines[0]
            title_part = header.split("**")[1]
            company_location = header.split(" at ")[1].split(" (")
            company = company_location[0]
            location = company_location[1].rstrip(")")
            type_line = lines[1].replace("Type:", "").strip()
            involvement_str, work_type_str = [s.strip() for s in type_line.split(",")]
            requirements_line = lines[2].replace("Requirements:", "").strip()
            description = " ".join(line.strip() for line in lines[3:]).replace("Description:", "").strip()
            job_list.append(JobListing(
                title=title_part,
                company=company,
                location=location,
                requirements=[r.strip() for r in requirements_line.split(",")],
                description=description
            ))
        except Exception:
            continue

    return job_list[:3]

print("✅ Loaded successfully!")
