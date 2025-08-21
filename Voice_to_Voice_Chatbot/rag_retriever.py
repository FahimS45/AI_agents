from typing import List, Dict
from ddgs import DDGS
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Embedder
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# DuckDuckGo Web Search
def duckduckgo_web_search(query: str, max_results: int = 8) -> List[Dict[str, str]]:
    results = []
    with DDGS() as ddgs:
        for result in ddgs.text(query, max_results=max_results):
            if "body" in result and "href" in result:
                results.append({
                    "text": result["body"],
                    "metadata": {"source": result["href"]}
                })
    return results

# Chunk and embed documents
def chunk_and_embed(texts: List[Dict[str, str]]) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = []
    for item in texts:
        source = item["metadata"]["source"]
        enriched_text = f"{item['text']}"
        chunks = splitter.create_documents([enriched_text], metadatas=[{"source": source}])
        docs.extend(chunks)
    return FAISS.from_documents(docs, embedder)

# Main function: Return top N relevant chunks with source
def retrieve_web_chunks(claim: str, top_k: int = 10) -> List[Dict[str, str]]:
    results = duckduckgo_web_search(claim, max_results=15)
    if not results:
        return []

    vectorstore = chunk_and_embed(results)
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.get_relevant_documents(claim)

    # Return clean format
    return [{"text": doc.page_content.strip(), "source": doc.metadata.get("source", "")} for doc in docs]

print('Loaded!')
