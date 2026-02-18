# 🚀 Guide Genie — AI-Powered Getting Started Guide Generator

<div align="center">

**Paste your resources. Get a polished, beginner-friendly guide — within a few minutes.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![CrewAI](https://img.shields.io/badge/CrewAI-Multi--Agent-ff6b35?style=flat-square)](https://crewai.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

</div>

---

## 📺 Demo Walkthrough

> Click the thumbnail below to watch the full project walkthrough.

<div align="center">
  <a href="https://drive.google.com/file/d/1w5hFdGVTlYYAtiptlvIYyrUM-jV4u_6K/view?usp=sharing" target="_blank">
    <img src="assets/Screenshot 2026-02-17 182636.png" alt="Guide Genie Demo Walkthrough" width="780" style="border-radius: 10px; border: 2px solid #2c7be5;" />
  </a>
  <br/>
  <sub>▶ Click to watch the walkthrough video</sub>
</div>

---

## 📌 What is Guide Genie?

**Guide Genie** is an AI-powered tool that automatically generates comprehensive, beginner-friendly **Getting Started Guides** from multiple input sources. You provide the resources — Guide Genie does the reading, research, and writing for you.

### How it works

Provide any combination of:
- 📺 **YouTube** video or channel URLs
- 🌐 **Web pages** — docs, blog posts, tutorials
- 📄 **Research papers** — arXiv links
- 📁 **Documents** — PDF, TXT, MD, MDX files

Guide Genie spins up a **multi-agent CrewAI pipeline** that:
1. Assigns each source type to a specialist research agent
2. Compiles a comprehensive internal research report
3. Transforms it into a clean, structured **HTML guide** you can download

---

## 🏗️ Architecture

```
guide_genie/
│
├── src/                                        # CrewAI agent package
│   └── guide_generator_flow/
│       ├── crews/
│       │   ├── research_crew/                  # Crew 1 — Hierarchical
│       │   │   ├── config/
│       │   │   │   ├── agents.yaml             # Agent definitions
│       │   │   │   └── tasks.yaml              # Task definitions
│       │   │   └── research_crew.py
│       │   └── writing_crew/                   # Crew 2 — Sequential
│       │       ├── config/
│       │       │   ├── agents.yaml
│       │       │   └── tasks.yaml
│       │       └── writing_crew.py
│       └── main.py                             # GuideGeneratorFlow entry point
│
├── backend/                                    # FastAPI REST API
│   ├── main.py                                 # API routes
│   ├── services.py                             # Flow execution + task tracking
│   ├── models.py                               # Pydantic request/response models
│   ├── config.py                               # Settings (reads .env)
│   └── requirements.txt                        # Backend-only dependencies                    
│
├── assets/                                     # Screenshots and media
│   └── screenshot.png
│
├── outputs/                                    # Generated HTML guides
│
├── pyproject.toml                              # Project + CrewAI dependencies
├── uv.lock
└── README.md
```

### Agent Pipeline

```
                        ┌─────────────────────────────────────┐
                        │         CREW 1: Research Crew        │
                        │         (Hierarchical Process)       │
                        │                                      │
  Sources ──────────►  │  Research Manager (orchestrator)     │
                        │       │                              │
                        │  ┌────┴──────────────────────────┐  │
                        │  │  ↓           ↓         ↓     ↓ │  │
                        │  │ YouTube   Web      Arxiv  Docs  │  │
                        │  │Specialist Specialist Spec  Spec  │  │
                        │  └──────────────────────────────┘  │
                        └──────────────┬──────────────────────┘
                                       │ Research Report
                                       ▼
                        ┌─────────────────────────────────────┐
                        │         CREW 2: Writing Crew         │
                        │         (Sequential Process)         │
                        │                                      │
                        │  Technical Writer ──► Content Editor │
                        └──────────────┬──────────────────────┘
                                       │
                                       ▼
                               📄 HTML Guide
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Agent Framework | [CrewAI](https://crewai.com) |
| LLM | OpenAI GPT (configurable) |
| Backend API | FastAPI + Uvicorn |
| Data Validation | Pydantic v2 |
| Package Manager | [uv](https://docs.astral.sh/uv/) |
| Frontend | [Lovable](https://lovable.dev) |

---

## 🛠️ Local Setup

### Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) package manager
- OpenAI API key

### 1. Clone the repository

```bash
git clone https://github.com/FahimS45/AI_agents.git
cd AI_agents/guide_genie
```

### 2. Install dependencies

```bash
# Install all CrewAI + project dependencies from pyproject.toml
uv sync

# Install backend API dependencies
uv pip install -r backend/requirements.txt
```

### 3. Configure environment variables

Open `.env` and fill in your values:

```env
OPENAI_API_KEY=sk-...
MODEL=gpt-5
```

### 4. Start the API server

```bash
uvicorn backend.main:app --reload
```

The API will be live at `http://localhost:8000`
Interactive docs available at `http://localhost:8000/docs`

---

## 📡 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/api/upload-documents` | Upload PDF / TXT / MD files |
| `POST` | `/api/generate-guide` | Start guide generation, returns `task_id` |
| `GET` | `/api/status/{task_id}` | Poll generation progress |
| `GET` | `/api/download/{task_id}` | Download the finished HTML guide |
| `DELETE` | `/api/cleanup/{task_id}` | Remove task and generated files |

### Typical frontend flow

```
1. POST /api/generate-guide   →  { task_id: "abc123" }
2. GET  /api/status/abc123    →  { status: "processing" }   ← poll every 10s
3. GET  /api/status/abc123    →  { status: "completed", download_url: "/api/download/abc123" }
4. GET  /api/download/abc123  →  downloads getting-started-guide-abc123.html
```

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | ✅ | Your OpenAI API key |
| `MODEL` | ✅ | LLM model name (e.g. `gpt-5`) |
| `CORS_ORIGINS` | Optional | Comma-separated allowed origins for CORS |
| `CREWAI_TRACING_ENABLED` | Optional | Enable CrewAI telemetry (`true`/`false`) |

---

## 📂 Supported Input Sources

| Source | Format | Notes |
|--------|--------|-------|
| YouTube | Video URL, Channel URL | Comma-separated. Fetches transcripts automatically. |
| Web pages | Any URL | Docs, blog posts, tutorials |
| Research papers | arXiv URLs or IDs | Comma-separated |
| Documents | PDF, TXT, MD, MDX | Upload via API or directly in the request |

---

## ⏱️ Performance

Guide generation is intentionally thorough — quality over speed. Here's what to expect based on real usage:

**Benchmark: 3 sources (YouTube link + Medium article + PDF) — ~9 minutes**

| Phase | What's happening | Time estimate |
|---|---|---|
| Research Crew | Manager delegates to specialists; each runs RAG queries + LLM extraction | ~5–6 min |
| Writing Crew — Technical Writer | Produces a full 3000–5000 word HTML guide from the research report | ~2 min |
| Writing Crew — Content Editor | Reads the entire guide and rewrites/polishes it end-to-end | ~1–2 min |

### Why it takes this long

- **Hierarchical process** — the Research Manager makes a planning call, delegates to specialists, then compiles all findings before passing to the Writing Crew. That's many LLM calls chained together.
- **Two full long-context passes** — the Technical Writer and Content Editor each process the entire document, not just chunks.
- **`planning=True`** on the research crew adds an extra planning step before any specialist work begins.
- **Source content volume** — a YouTube transcript can be thousands of tokens; a PDF adds more. The more content, the more RAG queries each specialist makes.

### What affects timing most

| Factor | Effect |
|---|---|
| Number of sources | Each additional source adds ~2–3 minutes |
| Document / video length | Longer content → more RAG queries → more time |
| Model choice | Faster models (e.g. `gpt-4o-mini`) reduce time; slower models improve quality |
| `planning=True` | Adds one extra LLM call at the start of the research crew |


---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <sub>Built with ❤️ using CrewAI · FastAPI · OpenAI</sub>
</div>