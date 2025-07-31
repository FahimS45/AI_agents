
import os
import json
import asyncio
from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import (
    Agent, OpenAIChatCompletionsModel, Runner, function_tool, 
    RunContextWrapper, InputGuardrailTripwireTriggered
)
from rag_tools.rag_skills import get_required_skills_with_rag  # RAG Skill Extraction
from rag_tools.rag_jobs import find_jobs_with_rag              # RAG Job Finder

# Load environment variables
load_dotenv()

BASE_URL = os.getenv("BASE_URL") 
API_KEY = os.getenv("API_KEY") 
MODEL_NAME = os.getenv("MODEL_NAME") 

if not BASE_URL or not API_KEY or not MODEL_NAME:
    raise ValueError("Please set BASE_URL, API_KEY, and MODEL_NAME.")

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

# ----------------------------- MODELS -----------------------------

class SkillGapResult(BaseModel):
    missing_skills: List[str]

class JobListing(BaseModel):
    title: str
    company: str
    location: str
    requirements: List[str]
    description: str
    Contact: str

class CourseRecommendation(BaseModel):
    skill: str
    courses: List[str]

@dataclass
class UserContext:
    user_id: str
    current_skills: List[str] = None
    target_job: Optional[str] = None
    preferred_location: Optional[str] = None
    involvement: Optional[str] = None           # "full-time" / "part-time"
    work_type: Optional[str] = None             # "remote" / "on-site" / "hybrid"
    missing_skills: List[str] = None
    session_start: datetime = None

    def __post_init__(self):
        if self.current_skills is None:
            self.current_skills = []
        if self.missing_skills is None:
            self.missing_skills = []
        if self.session_start is None:
            self.session_start = datetime.now()

# ----------------------------- TOOLS -----------------------------

@function_tool
def get_required_skills_for_job(wrapper: RunContextWrapper[UserContext], job_title: str) -> SkillGapResult:
    """Extract missing skills using the RAG system given a job title and current skills."""
    current_skills = wrapper.context.current_skills
    result = get_required_skills_with_rag(job_title=job_title)
    # Compare with current_skills
    result.missing_skills = [skill for skill in result.missing_skills if skill not in current_skills]
    wrapper.context.missing_skills = result.missing_skills
    return result

@function_tool
def find_matching_jobs(wrapper: RunContextWrapper[UserContext]) -> List[JobListing]:
    """Search for jobs using the RAG system based on skills, location, work type, and involvement."""
    return find_jobs_with_rag(
        skills=wrapper.context.current_skills,
        location=wrapper.context.preferred_location,
        involvement=wrapper.context.involvement,
        work_type=wrapper.context.work_type
    )

# Dummy course recommendation (only this part is not RAG-based)
course_data = {
    "SQL": [
        "SQL Basics - Coursera",
        "Advanced SQL - Udemy",
        "SQL for Data Science - edX"
    ],
    "Pandas": [
        "Pandas for Data Analysis - Datacamp",
        "Data Manipulation with Pandas - Coursera"
    ],
    "Statistics": [
        "Intro to Stats - Khan Academy",
        "Statistics with R - Coursera"
    ],
    "Machine Learning": [
        "ML Crash Course - Google",
        "Machine Learning A-Z - Udemy",
        "Deep Learning Specialization - Coursera"
    ],
    ".NET": [
        "C# Basics for Beginners - Udemy",
        "ASP.NET Core Fundamentals - Pluralsight",
        "Building Web Applications with ASP.NET - Coursera"
    ],
    "Java": [
        "Java Programming Masterclass - Udemy",
        "Java Fundamentals - Pluralsight",
        "Object Oriented Programming in Java - Coursera"
    ],
    "C": [
        "C Programming For Beginners - Udemy",
        "Introduction to Programming in C - Coursera",
        "C Fundamentals - Pluralsight"
    ],
    "C#": [
        "C# Intermediate Programming - Udemy",
        "Learn C# Fundamentals - Microsoft Learn",
        "Advanced C# Programming - Pluralsight"
    ],
    "Unity": [
        "Unity Game Development Fundamentals - Coursera",
        "Create with Code - Unity Learn",
        "Introduction to Unity - Udemy"
    ]
}

@function_tool
def recommend_courses(missing_skills: List[str]) -> List[CourseRecommendation]:
    """Recommend online courses for the user's missing skills."""
    recommendations = []
    for skill in missing_skills:
        courses = course_data.get(skill, ["No courses found"])
        recommendations.append(CourseRecommendation(skill=skill, courses=courses))
    return recommendations

# ----------------------------- AGENTS -----------------------------

job_finder_agent = Agent[UserContext](
    name="Job Finder Specialist",
    handoff_description="Specialist agent for finding jobs based on the user's context",
    instructions="""
    You find job openings using a RAG-based retrieval system.
    Match user's current skills, location, involvement, and work type.
    user's preferences are available in the context, including current skills, and work type.
    Use `find_matching_jobs` and return top 3 jobs in structured format.
    
    Always explain the reasoning behind your recommendations.
    
    Format your response in a clear, organized way with details.
    """,
    tools=[find_matching_jobs],
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
    output_type=List[JobListing]
)

course_recommender_agent = Agent[UserContext](
    name="Course Recommender Specialist",
    handoff_description="Specialist agent for finding online courses for user's missing skills.",
    instructions="""
    You suggest online courses for the user's missing skills.
    Use the `recommend_courses` tool.
    """,
    tools=[recommend_courses],
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
    output_type=List[CourseRecommendation]
)

skill_gap_agent = Agent[UserContext](
    name="Skill Gap Specialist",
    handoff_description="Specialist agent for finding missing skills for a target job",
    instructions="""
    You help users identify what skills they need to qualify for a target job.
    Use the tool `get_required_skills_for_job`.
    
    After showing missing skills, ask:
    - "Would you like me to recommend courses for these skills?"
    - "Or shall I find matching jobs based on your current skills?"
    Then hand off accordingly.
    """,
    tools=[get_required_skills_for_job],
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
    output_type=SkillGapResult,
    handoffs=[course_recommender_agent, job_finder_agent] 
)

conversation_agent = Agent[UserContext](
    name="Conversation Agent",
    instructions="""
    You are the main controller. You name is CareerMate. Always address yourself with this name.
    Route user queries to:
    - Skill Gap Agent → for job-related goals or if asked what skills are required for a specific job or want to become something related to IT, software, data, analysis, AI, game, programming, computer science and technology industry, etc. 
    - Job Finder Agent → If specifically asked for assistance in find a particular job in a particular job related to IT, software, data, analysis, AI, game, programming, computer science and technology industry, etc.
    - Course Recommender Agent → for learning resources

    If query is related to technology or IT Engineering, handle it directly. You can have Computer Science and Technology related conversation with the user.
    """,
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
    handoffs=[skill_gap_agent, job_finder_agent, course_recommender_agent]
)

# ----------------------------- MAIN -----------------------------

async def main():
    user_context = UserContext(
        user_id="user456",
        current_skills=["Python", "Excel"],
        target_job="data scientist",
        preferred_location="Delhi",
        involvement="full-time",
        work_type="remote"
    )

    queries = [
        "I want to be a data scientist",
        "Can you help me find jobs?",
        "How do I learn SQL and Pandas?",
        "What is deep learning?"
    ]

    for query in queries:
        print("\n" + "="*60)
        print(f"QUERY: {query}")
        print("="*60)
        try:
            result = await Runner.run(conversation_agent, query, context=user_context)
            print("\nRESPONSE:")
            print(result.final_output)
        except InputGuardrailTripwireTriggered:
            print("\n⚠️ GUARDRAIL TRIGGERED ⚠️")

if __name__ == "__main__":
    asyncio.run(main())
