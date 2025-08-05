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

## 🧪 Example Outputs

Below are some example conversations and how the AI News Intelligence Agent (`NewsSense`) responds to different types of queries:

---

### 🗨️ **Query:** Hi!

```
✅ FINAL RESPONSE:

Hello! How can I assist you today? Are you looking for trending news, fact-checking a claim, or a summary of a news article?
```

---

### 🗨️ **Query:** What are trending news in Data Science today?

```
✅ FINAL RESPONSE:

📰 CATEGORY: Data science  
📢 Headlines:
  1. Data Science: Interdisciplinary field combining statistics, computing, and domain expertise  
  2. Data Science integrates methods to analyze noisy and unstructured data  
  3. Recognized as a 'fourth paradigm' of science driven by data and IT impact  
  4. Data scientists blend programming and statistics to summarize and extract insights  
  5. Data science: Multidisciplinary approach critical for business decision-making  
  6. Growing demand for skilled data scientists as career opportunities expand
```

---

### 🗨️ **Query:** Humans are going to live in MARS one day!

```
✅ FINAL RESPONSE:

🧾 VERDICT: Likely True

🧠 SUMMARY:
Multiple sources indicate that humans are planning and developing technologies with the goal of living on Mars one day. NASA and other organizations like SpaceX have active programs and missions aimed at sending humans to Mars in the coming decades, with aspirations to establish a human presence on the planet. While the exact timeline remains uncertain, the scientific consensus and space agencies involved believe it is feasible that humans will live on Mars in the future.

🔗 SOURCES:
  1. https://scienceandspacenews.com/2025/01/21/when-will-humans-go-to-mars/
  2. https://abcnews.go.com/US/nasa-hopes-send-astronauts-mars-2030s/story?id=111859633
  3. https://www.sciencenewstoday.org/will-humans-ever-live-on-mars
```

---

### 🗨️ **Query:** Summarize this article:

> *Artificial Intelligence is no longer a support function—it’s evolving into a core component of how software is created. Two of the world’s biggest tech companies, Meta and Microsoft, are leading the charge in reshaping software development through AI...*

```
✅ FINAL RESPONSE:

🗞️ Summary of Article

📌 Bullet Points:
  1. AI is becoming a central part of software development, moving beyond just a support tool to a co-developer role.  
  2. Meta and Microsoft are leading this change, with AI writing significant portions of code and handling tasks like debugging and testing.  
  3. AI helps speed up development by allowing engineers to focus more on reviewing and designing rather than writing every line of code.  
  4. Microsoft reports that AI now writes a large share of their code, while Meta expects AI to produce half of its engineering work soon.  
  5. This shift enables faster innovation, more experimentation, and quicker responses to market needs.
```

---








