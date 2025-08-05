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
from agents import (
    Agent, OpenAIChatCompletionsModel, Runner,
    function_tool, set_tracing_disabled, ModelSettings,
    InputGuardrail, GuardrailFunctionOutput, InputGuardrailTripwireTriggered,
    RunContextWrapper
)
#import logfire
from rag_retriever import retrieve_web_chunks

#logfire.configure()
#logfire.instrument_openai_agents()
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
def get_trending_news(topic: str) -> List[str]:
    """Fetch trending headlines for a given topic using DuckDuckGo Search."""
    results = []
    with DDGS() as ddgs:
        for result in ddgs.text(topic, max_results=10):
            if "title" in result:
                results.append(result["title"])

    return results[:5]


@function_tool
def summarize_news(article_text: str) -> str:
    """Summarize an article into 3–5 bullet points using LLM. Returns plain text."""
    prompt = f"""
You are a helpful assistant that summarizes news content.

Summarize the following article or topic into 3 to 5 bullet points.

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
    handoff_description="Pulls trending headlines across categories like tech, politics, finance, etc.",
    instructions="""
    You are a news intelligence assistant that identifies and reports **trending news**.

    You are provided with a `get_trending_news` tool, which retrieves **real-time headlines** for a given category like:
    - tech
    - politics
    - finance
    - sports
    - health

    Your job is to:
    1. Use the `get_trending_news` tool with a clear category or topic.
    2. Receive the list of top headlines for that category.
    3. Group or rank topics if multiple are mentioned (only if needed).
    4. Return a structured response using:
        - `category` (e.g., "politics")
        - `headlines` (top 3–5 items, no duplicates)

    You must not make up headlines. Only report what's fetched via the tool.
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

    2. Based on the user's request, **handoff the task to the right specialized agent**:
        - If the user wants **recent headlines**, news trends, or what's popular — handoff to `trending_agent`.
        - If the user provides a **claim** or statement to verify — handoff to `fact_checker_agent`.
        - If the user provides a **long article, topic, or pasted news** — handoff to `summarizer_agent`.

    Rules:
    - Do not answer the user's query directly.
    - Always route the task to the correct agent.
    - Be concise and clear in your reasoning for the handoff.
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
        "What's trending in AI today?",
        "Is Apple partnering with OpenAI?"
    ]
    for query in queries:
        print("\n" + "="*60)
        print(f"USER: {query}")
        try:
            result = await Runner.run(conversation_agent, query, context=user_context)
            print("\nRESPONSE:")
            print(result.final_output)
        except InputGuardrailTripwireTriggered as e:
            print("\n⚠️ Guardrail Triggered ⚠️")

if __name__ == "__main__":
    asyncio.run(main())
