Sure! Here's a professional and informative `README.md` for your GitHub repository:

---

# 🧠 NewsSense: AI-Powered News Intelligence Agents

**NewsSense** is a multi-agent system that uses Large Language Models (LLMs) to intelligently **fetch, verify, and summarize news** using real-time web data and prompt-driven agent reasoning.

This project simulates a **conversational news assistant** with specialized agents capable of handling trending news, fact-checking claims, and summarizing lengthy articles — all backed by real-time search using [DDGS](https://pypi.org/project/ddgs/) (DuckDuckGo Search).

---

## 🔧 Features

### 🗂️ Multi-Agent Architecture

| Agent                       | Role                                                       |
| --------------------------- | ---------------------------------------------------------- |
| **Conversation Controller** | Routes user queries to appropriate agents                  |
| **Trending News Agent**     | Fetches trending headlines by topic (e.g., Tech, Politics) |
| **Fact Checker Agent**      | Verifies factual claims using retrieved web snippets       |
| **News Summarizer Agent**   | Summarizes long news articles into concise bullet points   |

---

## 🤖 How It Works

### 1. 🌐 Real-Time Retrieval (DDGS)

* Uses **DuckDuckGo Search** (via `ddgs`) to fetch fresh, unbiased news snippets and web documents.

### 2. 🔍 Claim Verification

* RAG-style retrieval gathers top-k relevant web chunks.
* The agent **compares** those against a user-provided claim.
* Returns a structured verdict:

  * `Likely True`
  * `Likely False`
  * `Unclear`
* Alongside a short **summary** and **top 3 supporting sources**.

### 3. 📈 Trending News

* Agent retrieves **top headlines** for any topic using DDGS.
* Headlines are grouped, deduplicated, and returned by category.

### 4. 📝 Summarization

* Accepts long articles or pasted content.
* Uses LLM to return **3–5 clear bullet points** with key highlights.

---

## 🏗️ Tech Stack

* **LangChain** for agent management and tool calling
* **OpenAI GPT (via LangChain)** as the core LLM
* **DDGS** for DuckDuckGo Search integration
* **FAISS** for in-memory vector search (in RAG)
* **HuggingFace Embeddings** (`all-MiniLM-L6-v2`)
* `Pydantic` for structured output models

---

## 🧪 Example Use Cases

### ✅ Fact Checking

```python
"Is Donald Trump the current president of the USA?"
```

→ `Fact Checker Agent` returns:

* **Verdict**: Likely True
* **Summary**: "Trump was sworn in as the 47th President in January 2025..."
* **Sources**: 3 reliable links

---

### 📰 Trending News

```python
"What's trending in AI today?"
```

→ `Trending News Agent` returns:

* **Category**: AI
* **Top Headlines**: \["Meta's LLM leads benchmarks", "Grok 2.0 rolls out to beta users", ...]

---

### 📝 Summarization

```python
"Summarize this article: [Paste full text]"
```

→ `News Summarizer Agent` returns:

* Bullet points summarizing the key content

---

## 🚀 Getting Started

1. **Clone the repo**

   ```bash
   git clone https://github.com/your-username/NewsSense.git
   cd NewsSense
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv project05
   source project05/bin/activate  # or use `project05\Scripts\activate` on Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Add your OpenAI key**

   * In `.env` or as a variable:

     ```bash
     export OPENAI_API_KEY="your-key"
     ```

5. **Run Tests**

   * Example: `python test_fact_checker.py`

---

## 📁 Project Structure

```
project/
├── agents/
│   ├── conversation_agent.py
│   ├── fact_checker_agent.py
│   ├── trending_agent.py
│   └── summarizer_agent.py
├── rag_tools/
│   ├── rag_retriever.py
│   └── llm_loader.py
├── models/
│   └── schemas.py  # Pydantic models
├── test/
│   └── test_fact_checker.py
├── main.py
└── README.md
```




