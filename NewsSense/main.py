# main.py

import warnings
import logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

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

import logfire
logfire.configure()
logfire.instrument_openai_agents()

load_dotenv()

BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

if not BASE_URL or not API_KEY or not MODEL_NAME:
    raise ValueError("Please set BASE_URL, API_KEY, and MODEL_NAME.")

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

class NewsBrief(BaseModel):
    title: str
    points: List[str]

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
def summarize_news(article_text: str) -> str:
    """Summarize an article into 3–5 bullet points using LLM. Returns plain text."""
    prompt = f"""
You are a helpful assistant that summarizes long news or articles into shorter bullet points.

You have to summarize the long articles with rephrased, much shorter, easy to understand bullet points.

Summarize the following article or topic into 3 to 5 much shorter, easy to understand bullet points.

TEXT:
\"\"\"{article_text}\"\"\"

Format:
- Bullet 1
- Bullet 2
- Bullet 3
...
"""
    return llm.invoke(prompt)


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
    handoff_description="Pulls trending news across categories like tech, politics, finance, etc., or generally if no topic is mentioned.",
    instructions="""
    You are a news intelligence assistant that identifies and generate **headings** based on real-time trending news.

    You are provided with a tool called `get_trending_news`, which retrieves recent **news snippets** from the web.

    Your responsibilities:
    1. If the user specifies a topic or category (e.g., "tech", "politics", "finance", "health", etc.), pass that as the input to the tool.
    2. If no specific topic is mentioned, call the tool without arguments to retrieve **general trending news**.
    3. Rephrase the snippets and make them short headlines for better understanding.
    4. Respond in a structured format with:
       - `category`: the topic if mentioned (or "general" if none was provided)
       - `headlines`: a list of rephrased shorter top unique news headlines (no duplicates)

    - Do not make up any headlines.
    - Only use what is returned from the tool and rephrase it for better understanding.
    - Avoid opinion, summary, or analysis—just clean, and rephrased shorter headlines.
    """,
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
    tools=[get_trending_news],
    output_type=TrendingTopics
)



summarizer_agent = Agent(
    name="News Summarizer Agent",
    handoff_description="Summarizes long news articles",
    instructions="""
    You are a news summarization expert.

    You will receive a long news article or topic. Use the `summarize_news` tool to extract the key points.

    Steps:
    1. Pass the article/topic to the tool to get 3–5 bullet points.
    2. Format the final output as:
    - `title`: Always "Summary of Article"
    - `points`: A list of 3–5 bullet points from the tool output.
    3. Do not invent or hallucinate points. Only use what the tool returns.
    4. Ensure the bullet points are clear, factual, and non-redundant.
    """,
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
    tools=[summarize_news],
    output_type=NewsBrief
)


fact_checker_agent = Agent(
    name="Fact Checker Agent",
    handoff_description="Verifies factual claims using retrieved web documents.",
    instructions="""
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
""",
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
    tools=[fact_check_claim],  
    output_type=ClaimCheckResult  
)


conversation_agent = Agent[UserContext](
    name="Conversation Controller",
    handoff_description="Handles user questions and routes to the appropriate agent.",
    instructions="""
    You are a smart assistant designed to manage user conversations intelligently.

    Your role is to:
    1. **Understand the user’s intent** — whether they want:
        - Trending news
        - Fact-checking a claim
        - Summarizing a news article or content

    2. Based on the user's request, **handoff the task to the right specialized agent**

    Rules:
    - Do not answer the user's query directly.
    - Always route the task to the correct agent.
    - Never invent content. If the intent is unclear, politely ask the user to clarify.

    Remember:
    You are just a **controller** and should not perform any analysis yourself.
    """,
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
    tools=[],  
    handoffs=[trending_agent, summarizer_agent, fact_checker_agent],
)


# --- Main Entry ---

async def main():
    user_context = UserContext(user_id="user001")
    queries = [
        "Hi!",
        "What are trending news in Data Science today?",
        "Humans are going to live in MARS one day!",
        "Summarize this:\n\nArtificial Intelligence is no longer a support function—it’s evolving into a core component of how software is created. Two of the world’s biggest tech companies, Meta and Microsoft, are leading the charge in reshaping software development through AI. This shift is not about automation alone; it’s about redefining the role of the engineer and accelerating the development lifecycle. At the heart of this transition is a bold new vision: AI tools are no longer assistants—they’re co-developers. Already, AI systems are helping write substantial portions of production code, perform complex debugging, and streamline testing. What was once considered a novelty is fast becoming the norm. Microsoft reports that a significant percentage of code across its products is now written by AI. In some cases, AI is responsible for nearly all the initial drafting of code, with human engineers playing the role of reviewers and refiners. Rather than replacing developers, AI is shifting their focus—from being code writers to becoming code supervisors and architectural thinkers. Similarly, Meta is investing in AI systems designed not just for autocompletion but for comprehensive coding—capable of writing, testing, and optimizing code independently. The company believes AI will soon be responsible for half of its engineering output. This evolution is helping their teams build faster, experiment more boldly, and respond quickly to changing market demands."
    ]

    for query in queries:
        print("\n" + "=" * 50)
        print(f"QUERY: {query}")
        print("=" * 50)

        try:
            result = await Runner.run(conversation_agent, query, context=user_context)

            print("\n✅ FINAL RESPONSE:\n")

            # Trending News
            if hasattr(result.final_output, "category") and hasattr(result.final_output, "headlines"):
                trending = result.final_output
                print(f"📰 CATEGORY: {trending.category.capitalize()}")
                print("📢 Headlines:")
                for i, headline in enumerate(trending.headlines, 1):
                    print(f"  {i}. {headline}")

            # Claim Fact-Check
            elif hasattr(result.final_output, "verdict"):
                verdict = result.final_output
                print(f"🧾 VERDICT: {verdict.verdict}")
                print("\n🧠 SUMMARY:")
                print(verdict.summary)
                print("\n🔗 SOURCES:")
                for i, src in enumerate(verdict.sources, 1):
                    print(f"  {i}. {src}")

            # Article Summary
            elif hasattr(result.final_output, "title") and hasattr(result.final_output, "points"):
                summary = result.final_output
                print(f"🗞️ {summary.title}")
                print("\n📌 Bullet Points:")
                for i, point in enumerate(summary.points, 1):
                    print(f"  {i}. {point}")

            else:
                print(result.final_output)

        except InputGuardrailTripwireTriggered as e:
            print("\n⚠️ Guardrail Triggered ⚠️")

        
if __name__ == "__main__":
    asyncio.run(main())

