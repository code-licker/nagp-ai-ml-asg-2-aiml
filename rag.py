"""
RAG (Retrieval-Augmented Generation) Module for Singapore Travel Knowledge.
Queries the persistent Chroma vector store and provides semantic similarity search
with metadata attribution (source_title, source_url) for LangChain agents and UI.
"""

import os
import sys
from typing import List, Dict, Any

os.environ["ANONYMIZED_TELEMETRY"] = "False"

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from langchain_core.tools import tool

CHROMA_DIR = os.path.join(current_dir, "chroma_db")

_collection = None


def get_chroma_collection():
    """Lazily loads the persistent Chroma collection."""
    global _collection
    if _collection is None:
        if not os.path.exists(CHROMA_DIR):
            raise FileNotFoundError(
                f"Chroma DB directory '{CHROMA_DIR}' not found. Please run 'python ingest.py' first!"
            )
        client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False)
        )
        ef = DefaultEmbeddingFunction()
        _collection = client.get_collection(
            name="singapore_travel_kb",
            embedding_function=ef
        )
    return _collection


def retrieve_travel_knowledge(query: str, top_k: int = 4) -> List[Dict[str, Any]]:
    """
    Performs semantic similarity search against the Singapore travel knowledge base.
    Returns list of dictionaries containing content, source_title, and source_url.
    """
    collection = get_chroma_collection()
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )

    retrieved = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0] if "distances" in results else [0.0] * len(docs)

    for i in range(len(docs)):
        meta = metas[i] if i < len(metas) else {}
        retrieved.append({
            "content": docs[i],
            "source_title": meta.get("source_title", "Singapore Travel Guide"),
            "source_url": meta.get("source_url", "https://www.visitsingapore.com"),
            "category": meta.get("category", "General"),
            "distance": float(distances[i]) if i < len(distances) else 0.0
        })
    return retrieved


def format_retrieved_context(retrieved: List[Dict[str, Any]]) -> str:
    """Formats retrieved chunks into a clean, cited context string for the LLM."""
    if not retrieved:
        return "No specific travel documents found in knowledge base."

    parts = []
    for i, r in enumerate(retrieved, start=1):
        header = f"--- [Document {i}: {r['source_title']} ({r['source_url']})] ---"
        parts.append(f"{header}\n{r['content']}")
    return "\n\n".join(parts)


@tool
def search_singapore_knowledge(query: str) -> str:
    """
    Search the Singapore travel knowledge base for destination facts, attractions,
    indoor/outdoor activities, cultural precincts (Chinatown, Little India, Marina Bay),
    transportation (MRT/bus), food, and sample itineraries.
    Always use this tool for questions about Singapore attractions, heritage, or activities.
    """
    docs = retrieve_travel_knowledge(query, top_k=4)
    return format_retrieved_context(docs)


if __name__ == "__main__":
    print("Testing RAG retrieval...")
    res = retrieve_travel_knowledge("indoor attractions when it rains", top_k=2)
    print(format_retrieved_context(res))
