# multiagent.py

import os
import asyncio
from ddgs import DDGS
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import AsyncOpenAI
from llm_loader import llm
from rag_retriever import retrieve_web_chunks
from agents import (
    Agent, OpenAIChatCompletionsModel, Runner,
    function_tool, set_tracing_disabled, ModelSettings,
    InputGuardrail, GuardrailFunctionOutput, InputGuardrailTripwireTriggered,
    RunContextWrapper
)
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions

#import logfire
#logfire.configure()
#logfire.instrument_openai_agents()

load_dotenv()

BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

if not BASE_URL or not API_KEY or not MODEL_NAME:
    raise ValueError("Please set BASE_URL, API_KEY, and MODEL_NAME.")

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)


class ClaimCheckResult(BaseModel):
    verdict: str
    sources: List[str]
    summary: str

class TrendingTopics(BaseModel):
    category: str
    headlines: List[str]

@dataclass
class UserContext:
    user_id: str
    session_start: datetime = None

    def __post_init__(self):
        if self.session_start is None:
            self.session_start = datetime.now()

# --- Tools ---

@function_tool
def get_trending_news(topic: Optional[str] = None) -> List[str]:
    """Fetch trending news snippets. If a topic is provided, fetch news on that topic; otherwise, fetch general trending news."""
    query = topic if topic else "latest news"
    results = []

    with DDGS() as ddgs:
        for result in ddgs.text(query, max_results=10):
            if "body" in result:
                snippet = result["body"].strip()
                if snippet and snippet not in results:  # Avoid duplicates
                    results.append(snippet)

    return results[:5]

@function_tool
def fact_check_claim(claim: str) -> Dict[str, Any]:
    """Retrieve relevant web chunks for the claim."""
    retrieved_docs = retrieve_web_chunks(claim, top_k=10)
    sources = list({doc['source'] for doc in retrieved_docs})[:3]
    combined_text = "\n\n".join([doc['text'] for doc in retrieved_docs])

    return {
        "sources": sources,
        "evidence": combined_text
    }

# --- Agents ---

trending_agent = Agent(
    name="Trending News Agent",
    handoff_description="Specialized agent for pulling trending news across categories like tech, politics, finance, etc., or generally if no topic is mentioned.",
    instructions=prompt_with_handoff_instructions(
        """
        You are a news intelligence assistant that identifies and generates headlines based on real-time trending news.
        
        You are provided with a tool called `get_trending_news`, which retrieves recent news snippets from the web.
        
        Your responsibilities:
        1. If the user specifies a topic or category (e.g., 'tech', 'politics', 'finance', 'health', etc.), pass that as the input to the tool.
        2. If no specific topic is mentioned, call the tool without arguments to retrieve general trending news.
        3. Rephrase the snippets and make them short headlines for better understanding.
        4. Respond in a structured format with:
           - category: the topic if mentioned (or "general" if none was provided)
           - headlines: a list of rephrased shorter top unique news headlines (no duplicates)
        
        Important guidelines:
        - Do not make up any headlines
        - Only use what is returned from the tool and rephrase it for better understanding
        - Avoid opinion, summary, or analysis—just clean, and rephrased shorter headlines.
        
        """
    ),
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
    tools=[get_trending_news],
    output_type=TrendingTopics
)

fact_checker_agent = Agent(
    name="Fact Checker Agent",
    handoff_description="Specialized agent for Verifying factual claims using retrieved web documents.",
    instructions=prompt_with_handoff_instructions(
        """
        You are a helpful fact-checking assistant.

        Your task is to assess the **truthfulness of a user's claim** based on factual information retrieved from reliable web sources.

        Use  tool to get the evidence.

        You will be provided with:
        - `claim`: the user's input statement to evaluate.

        Then, you must use the 'fact_check_claim' tool to get the evidence and the sources of evidence: 
        - `evidence`: a combined snippet of factual text retrieved from multiple web documents.
        - `sources`: a list of URLs where the evidence was retrieved from.

        Your job is to:
        1. Carefully analyze the **claim**.
        2. Compare it against the **evidence** text provided.
        3. Decide whether the claim is:
        - "Likely True"
        - "Likely False"
        - "Unclear" (if evidence is insufficient or inconclusive)

        Then, generate:
        - A clear **verdict**
        - A **concise summary** (3-4 sentences) of the most relevant supporting or refuting points
        - A list of the **top 3 sources** that support your verdict

        Guidelines:
        - Do **not** fabricate facts or cite sources that are not explicitly in the `evidence`.
        - If information is missing or unclear, choose "Unclear" as the verdict.
        - Be objective, transparent, and evidence-based in your reasoning.
        
        """
    ),
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
    tools=[fact_check_claim],  
    output_type=ClaimCheckResult  
)


conversation_agent = Agent[UserContext](
    name="Conversation Controller",
    instructions=
        """
        You are a smart assistant designed to manage user conversations intelligently.

        Your role is to:
        1. **Understand the user’s intent** — whether they want:
            - Trending news
            - Fact-checking a claim

        2. Based on the user's request, hand off the task to the right specialized agent

        You can
        - Do cansual news realted coversation with the user.
        - Hand off to the specialized agent for pulling trending news, and verifying factual claims.
        - Never invent content. If the intent is unclear, politely ask the user to clarify.

        Remember:
        You are just a **controller** and should not perform any analysis yourself.
        
        """,
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client), 
    handoffs=[trending_agent, fact_checker_agent]
)
