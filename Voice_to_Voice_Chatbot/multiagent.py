import os
import asyncio
from ddgs import DDGS
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from dataclasses import dataclass
from datetime import datetime
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
    conversation_history: List[Dict[str, str]] = None

    def __post_init__(self):
        if self.session_start is None:
            self.session_start = datetime.now()
        if self.conversation_history is None:
            self.conversation_history = []
    
    def add_to_history(self, user_input: str, agent_response: str):
        """Add interaction to conversation history"""
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user": user_input,
            "assistant": agent_response
        })
        # Keep only last 3 interactions to prevent context overflow and reduce costs
        if len(self.conversation_history) > 3:
            self.conversation_history = self.conversation_history[-3:]


# --- Voice Formatting Function ---

def format_for_voice(agent_output: Any) -> str:
    """Convert structured output to natural speakable sentences for TTS model"""
    if isinstance(agent_output, TrendingTopics):
        category_text = agent_output.category if agent_output.category != "general" else ""
        intro = f"Here are the top {category_text} news headlines: " if category_text else "Here are the top trending news headlines: "
        
        # Format all headlines naturally for speech
        headlines_text = ". ".join(agent_output.headlines)
        
        return intro + headlines_text
    
    elif isinstance(agent_output, ClaimCheckResult):
        verdict_intro = {
            "Likely True": "Based on my research, this claim appears to be true.",
            "Likely False": "Based on my research, this claim appears to be false.",
            "Unclear": "I couldn't find enough reliable evidence to verify this claim."
        }.get(agent_output.verdict, f"My verdict is: {agent_output.verdict}")
        
        return f"{verdict_intro} {agent_output.summary}"
    
    else:
        # Handle regular string responses
        return str(agent_output)


# --- Tools with caching for uniqueness ---

# Global cache to store previously fetched news
_news_cache = {}

@function_tool
def get_trending_news(topic: Optional[str] = None, get_more: bool = False) -> List[str]:
    """
    Fetch trending news snippets. 
    If a topic is provided, fetch news on that topic; otherwise, fetch general trending news.
    If get_more is True, fetch additional unique results beyond previously shown ones.
    """
    query = topic if topic else "latest news"
    cache_key = query.lower()
    
    # Initialize cache for this topic if not exists
    if cache_key not in _news_cache:
        _news_cache[cache_key] = {
            'shown_snippets': set(),
            'all_snippets': [],
            'last_fetch': None
        }
    
    cache_entry = _news_cache[cache_key]
    
    # Fetch more results if requesting more details or cache is empty
    max_results = 20 if get_more or not cache_entry['all_snippets'] else 10
    
    with DDGS() as ddgs:
        fresh_results = []
        for result in ddgs.text(query, max_results=max_results):
            if "body" in result:
                snippet = result["body"].strip()
                if snippet and snippet not in cache_entry['shown_snippets']:
                    fresh_results.append(snippet)
        
        # Add new unique snippets to cache
        for snippet in fresh_results:
            if snippet not in cache_entry['all_snippets']:
                cache_entry['all_snippets'].append(snippet)
    
    # Get results that haven't been shown yet
    if get_more:
        # Return next batch of unseen results
        unseen_results = [
            snippet for snippet in cache_entry['all_snippets'] 
            if snippet not in cache_entry['shown_snippets']
        ]
        results_to_show = unseen_results[:5]
    else:
        # Return first batch of results
        results_to_show = cache_entry['all_snippets'][:5]
    
    # Mark these results as shown
    for snippet in results_to_show:
        cache_entry['shown_snippets'].add(snippet)
    
    return results_to_show

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
    handoff_description="Specialized agent for pulling trending news across categories like tech, politics, finance, data science, AI, etc., or generally if no topic is mentioned. Can also provide more details about previously shown news.",
    instructions=prompt_with_handoff_instructions(
        """
        You are a news intelligence assistant that identifies and generates headlines based on real-time trending news.
        
        You are provided with a tool called `get_trending_news`, which retrieves recent news snippets from the web.
        
        Your responsibilities:
        1. If the user specifies a topic or category (e.g., 'tech', 'politics', 'finance', 'health', 'data science', 'AI', etc.), pass that as the input to the tool.
        2. If no specific topic is mentioned, call the tool without arguments to retrieve general trending news.
        3. If the user asks for "more details", "tell me more", or follow-up questions about previously shown news, call the tool with get_more=True to fetch additional unique results.
        4. Rephrase the snippets and make them short, clear headlines for better understanding.
        5. Respond in a structured format with:
           - category: the topic if mentioned (or "general" if none was provided)
           - headlines: a list of rephrased shorter top unique news headlines (no duplicates)
        
        Important guidelines:
        - Do not make up any headlines
        - Only use what is returned from the tool and rephrase it for better understanding
        - Keep headlines concise and informative
        - Avoid opinion, summary, or analysis—just clean, and rephrased shorter headlines.
        - When user asks for more details about a topic, use get_more=True to get fresh unique content
        
        """
    ),
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
    tools=[get_trending_news],
    output_type=TrendingTopics
)

fact_checker_agent = Agent(
    name="Fact Checker Agent",
    handoff_description="Specialized agent for verifying factual claims, statements, or when user asks to check if something is true/false.",
    instructions=prompt_with_handoff_instructions(
        """
        You are a helpful fact-checking assistant.

        Your task is to assess the **truthfulness of a user's claim** based on factual information retrieved from reliable web sources.

        Use the fact_check_claim tool to get the evidence.

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
        - Keep your summary conversational and easy to understand when spoken aloud.
        
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
        You are a smart and friendly AI assistant designed to have natural conversations with users while intelligently routing specialized tasks to expert agents.

        Your role is to:
        1. **Engage in natural conversation** - Be conversational, helpful, and engaging
        2. **Understand user intent** and determine when to hand off tasks:
            - For **trending news requests**: Hand off to Trending News Agent
            - For **fact-checking, claim verification, or "is it true that..."** questions: Hand off to Fact Checker Agent
        3. **Handle follow-up questions** naturally by referencing previous conversation context
        4. **Maintain conversation flow** - Keep track of what was discussed before

        **When to hand off:**
        - News requests: "latest news", "trending topics", "what's happening in [topic]", "news about [subject]"
        - Fact-checking: "is it true that...", "can you verify...", "fact-check this", "is this correct", statements that sound like claims to be verified

        **Conversation guidelines:**
        - Be warm, natural, and conversational
        - Reference previous parts of the conversation when relevant
        - Ask clarifying questions if user intent is unclear
        - For general questions, provide helpful responses without handoffs
        - Keep responses concise but informative
        - Remember the conversation context and build upon it

        **Important:**
        - You are the main conversation controller - users will primarily interact with you
        - Only hand off specific tasks (news/fact-checking) to specialist agents
        - For everything else, have a natural conversation
        - Never mention technical details about agents or handoffs to the user
        
        Recent conversation context is available in your context. Use it to maintain continuity and handle follow-ups naturally.
        """,
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client), 
    handoffs=[trending_agent, fact_checker_agent]
)