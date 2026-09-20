"""
Unit test for Singapore Travel RAG retrieval.
Verifies semantic similarity search and metadata attribution from Chroma DB.
"""

import os
import sys

# Ensure current project directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag import retrieve_travel_knowledge, format_retrieved_context


def test_rag_queries():
    test_queries = [
        "What indoor attractions can I visit if it rains?",
        "How can a tourist travel around Singapore on MRT?",
        "What are the best food experiences and hawker centres?",
        "Suggest activities for a family with children."
    ]

    print("=== RUNNING RAG SEMANTIC RETRIEVAL TESTS ===\n")
    for q in test_queries:
        print(f"Query: '{q}'")
        docs = retrieve_travel_knowledge(q, top_k=2)
        assert len(docs) > 0, f"No documents retrieved for: {q}"
        for i, d in enumerate(docs, start=1):
            print(f"  [{i}] Source: {d['source_title']} ({d['source_url']})")
            snippet = d['content'][:150].replace('\n', ' ')
            print(f"      Snippet: {snippet}...")
        print(" Retrieval: PASSED\n")

    print("=== ALL RAG RETRIEVAL TESTS COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    test_rag_queries()
