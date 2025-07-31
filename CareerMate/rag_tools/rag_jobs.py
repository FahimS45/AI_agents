# rag_tools/rag_jobs.py

from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.schema import BaseRetriever
from openai import AsyncOpenAI
from typing import List, Optional
from pydantic import BaseModel
from rag_tools.setup_vectorstore import hybrid_retriever
from rag_tools.llm_loader import llm

# NEW IMPORTS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from agents import function_tool

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
    ("user", "{input}")
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
    Use RAG to search job listings that match the user's skills, location, involvement, and work type.
    """
    # Build a natural language query
    query_parts = [f"Find jobs requiring: {', '.join(skills)}."]
    if location:
        query_parts.append(f"Location: {location}.")
    if involvement:
        query_parts.append(f"Involvement: {involvement}.")
    if work_type:
        query_parts.append(f"Work type: {work_type}.")
    query = " ".join(query_parts)

    # --- REPLACED RetrievalQA with modern create_retrieval_chain pattern ---
    
    # 1. Create a chain to combine the documents into a single prompt
    #    This part knows about the {context} and {input} placeholders
    document_chain = create_stuff_documents_chain(llm, qa_prompt_for_jobs)
    
    # 2. Create the full RAG chain that first retrieves documents, then passes them
    #    to the document_chain. This chain expects the input key "input".
    rag_chain = create_retrieval_chain(hybrid_retriever, document_chain)

    # Invoke the chain using the input key "input"
    response = await rag_chain.ainvoke({"input": query})
    
    # --- End of replacement ---

    # The response object is slightly different.
    # The result is now in response["answer"]
    raw_result = response["answer"]
    
    # ... The rest of your code remains the same ...
    
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
            description = ""
            contact = None

            # Collect description and contact lines from the rest of lines (3 onward)
            for line in lines[3:]:
                line = line.strip()
                if line.lower().startswith("description:"):
                    # Remove "Description:" and add text to description
                    description += line[len("Description:"):].strip() + " "
                elif line.lower().startswith("contact information:"):
                    # Remove label and assign contact info
                    contact = line[len("Contact information:"):].strip()
                else:
                    # If other lines exist, consider them part of description as well
                    description += line + " "

            description = description.strip()

            job_list.append(JobListing(
                title=title_part,
                company=company,
                location=location,
                requirements=[r.strip() for r in requirements_line.split(",")],
                description=description,
                contact=contact
            ))
        except Exception:
            continue

    return job_list[:3]

