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
# Assuming these are the corrected files from previous conversations
from rag_tools.rag_skills import get_required_skills_with_rag  
from rag_tools.rag_jobs import find_jobs_with_rag
from rag_tools.rag_skills import SkillGapResult 
from rag_tools.rag_jobs import JobListing


# Load environment variables
load_dotenv()

BASE_URL = os.getenv("BASE_URL") 
API_KEY = os.getenv("API_KEY") 
MODEL_NAME = os.getenv("MODEL_NAME") 

if not BASE_URL or not API_KEY or not MODEL_NAME:
    raise ValueError("Please set BASE_URL, API_KEY, and MODEL_NAME.")

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

# ----------------------------- MODELS -----------------------------

class CourseRecommendation(BaseModel):
    skill: str
    courses: List[str]

@dataclass
class UserContext:
    user_id: str
    current_skills: List[str] = None
    target_job: Optional[str] = None
    preferred_location: Optional[str] = None
    involvement: Optional[str] = None      # "full-time" / "part-time"
    work_type: Optional[str] = None        # "remote" / "on-site" / "hybrid"
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
async def get_required_skills_for_job(wrapper: RunContextWrapper[UserContext]) -> SkillGapResult:
    """Extract missing skills using the RAG system given a job title and current skills."""
    current_skills = wrapper.context.current_skills
    job_title = wrapper.context.target_job

    # Check if a target job is available in the context
    if not job_title:
        # Handle the case where no target job is set
        return SkillGapResult(missing_skills=[])

    # CRITICAL FIX: 'await' is required here because get_required_skills_with_rag is an async function
    required_skills_result = await get_required_skills_with_rag(job_title=job_title)
    
    # Compare with current_skills to find missing ones
    required_skills = required_skills_result.missing_skills
    missing_skills = [skill for skill in required_skills if skill not in current_skills]
    
    wrapper.context.missing_skills = missing_skills
    return missing_skills


@function_tool
async def find_matching_jobs(wrapper: RunContextWrapper[UserContext]) -> List[JobListing]:
    """Search for jobs using the RAG system based on skills, location, work type, and involvement."""
    # CRITICAL FIX: 'await' is required here because find_jobs_with_rag is an async function
    return await find_jobs_with_rag(
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
    User's preferences are available in the context. Use the `find_matching_jobs` tool.

    Once you receive the list of jobs from the tool:
    - If the list is empty, respond politely by saying no jobs were found for the current criteria. Suggest broadening the search by changing the location, involvement, or work type.
    - If jobs are found, present them in a clear, organized, and conversational manner.
    - Use a numbered list for the jobs. For each job, include the Title, Company, Location, and a brief description.
    - Always provide a concluding sentence to maintain a helpful tone.

    Do not return the raw list of JobListing objects. Format your final response as a string.
    """,
    tools=[find_matching_jobs],
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
    output_type=str  # Changed to string for conversational output
)

course_recommender_agent = Agent[UserContext](
    name="Course Recommender Specialist",
    handoff_description="Specialist agent for finding online courses for user's missing skills.",
    instructions="""
    You suggest online courses for the user's missing skills.
    Use the `recommend_courses` tool to get the data.

    Once you receive the course recommendations from the tool:
    - Format the recommendations in a clear and conversational manner.
    - Use a bullet-point list for the courses under each skill.
    - If a skill has no recommended courses, state that politely.

    Do not return the raw list of CourseRecommendation objects. Format your final response as a string.
    """,
    tools=[recommend_courses],
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
    output_type=str # Changed to string for conversational output
)

skill_gap_agent = Agent[UserContext](
    name="Skill Gap Specialist",
    handoff_description="Specialist agent for finding missing skills for a target job",
    instructions="""
    You help users identify what skills they need to qualify for a target job.
    Use the tool `get_required_skills_for_job`.
    
    After you receive the list of missing skills from the tool, present them to the user in a friendly, conversational manner.
    Then, ask the user the following two questions to guide the conversation:
    - "Would you like me to recommend courses for these skills?"
    - "Or shall I find matching jobs based on your current skills?"
    
    Do not return the raw list of skills. Format your response as a complete message.
    """,
    tools=[get_required_skills_for_job],
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
    # IMPORTANT: The output_type of the agent should be a string, not the pydantic model
    # The agent's job is to convert the pydantic model into a string.
    output_type=str,
    handoffs=[course_recommender_agent, job_finder_agent] 
)

conversation_agent = Agent[UserContext](
    name="Conversation Agent",
    instructions="""
    You are the main controller. Your name is CareerMate. Always address yourself with this name.
    Your main goal is to route user queries to the appropriate specialist agent.
    - Skill Gap Agent: For job-related goals, or if asked what skills are required for a specific job.
    - Job Finder Agent: For requests to find jobs based on user's skills and preferences.
    - Course Recommender Agent: For requests for learning resources or courses.
    
    If a query doesn't fit a specific tool, handle it directly. For example, you can have a general conversation about computer science and technology.
    """,
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
    handoffs=[skill_gap_agent, job_finder_agent, course_recommender_agent]
)

# ----------------------------- MAIN -----------------------------

async def main():
    user_context = UserContext(
        user_id="user456",
        current_skills=["Python", "SQL", "Machine Learning"],
        target_job="Data Analyst",
        preferred_location="Delhi",
        involvement="full-time",
        work_type="remote"
    )

    queries = [
        #"I want to be a Game Developer",
        "Find me a job."
        #"What extra skills do I need to become a Data Analyst?"
        #"How do I learn SQL and Pandas?",
        #"What is deep learning?"
    ]

    for query in queries:
        print(f"User: {query}")
        try:
            result = await Runner.run(conversation_agent, query, context=user_context)
            print(f"CareerMate: {result.final_output}")
        except InputGuardrailTripwireTriggered:
            print("CareerMate: ⚠️ GUARDRAIL TRIGGERED ⚠️")
        print("-" * 60) # Use a simple separator

if __name__ == "__main__":
    asyncio.run(main())