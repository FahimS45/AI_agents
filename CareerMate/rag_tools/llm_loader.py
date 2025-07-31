# rag_tools/llm_loader.py

import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from langchain_openai import ChatOpenAI

# ✅ Load environment variables from .env file
load_dotenv()


print("API Key:", os.getenv("API_KEY"))

BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

if not BASE_URL or not API_KEY or not MODEL_NAME:
    raise ValueError("Please set BASE_URL, API_KEY, and MODEL_NAME in your .env file")

# Initialize OpenAI client

llm = ChatOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    model=MODEL_NAME,
    temperature=0.3,
   streaming=False
)
