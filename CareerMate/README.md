# 🧠 CareerMate: Multi-Agent AI Career Advisor

**CareerMate** is a multi-agent AI assistant designed to help users identify missing skills, discover learning paths, and find relevant job opportunities in the tech industry. It uses **RAG (Retrieval-Augmented Generation)** and OpenAI's function-calling capabilities to provide personalized, context-aware responses.

---

## 🚀 Features

- 🔍 **Skill Gap Analysis**  
  Identifies the missing skills required for your target job using a RAG-based knowledge retrieval pipeline.

- 🎓 **Course Recommendation**  
  Recommends high-quality online courses for missing or desired skills (currently based on a curated static dataset).

- 💼 **Job Matching**  
  Finds job listings based on your current skills, preferred location, employment type, and work mode using RAG over a job dataset.

- 🧠 **Multi-Agent System**  
  A smart controller agent routes user queries to three specialized sub-agents:
  - `Skill Gap Specialist`
  - `Job Finder Specialist`
  - `Course Recommender Specialist`

---

## 🧠 Built With

- [LangChain Agents](https://python.langchain.com/)
- [FAISS Vector Store](https://github.com/facebookresearch/faiss)
- [HuggingFace Embeddings](https://huggingface.co/)
- [AsyncOpenAI](https://platform.openai.com/docs/guides/function-calling)
- [RAG (Retrieval-Augmented Generation)](https://huggingface.co/blog/rag)

---

## 📊 Dataset Used

We use the [LinkedIn Job Clean Data](https://www.kaggle.com/datasets/shashankshukla123123/linkedin-job-cleandata/data) from Kaggle, which contains **over 10,000 cleaned job postings**.  
Each listing includes:

- ✅ Job title  
- ✅ Company  
- ✅ Location  
- ✅ Employment type  
- ✅ Required skills  
- ✅ Description  

This dataset is converted into a vector store to enable **hybrid RAG** (retriever + generator) job matching based on user input.

---

## 📂 Project Structure

