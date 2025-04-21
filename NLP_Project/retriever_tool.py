                                                               
from smolagents.tools import Tool
from datetime import datetime
import re

class RetrieverTool(Tool):
    name = "retriever"
    description = "Retrieves semantically relevant past conversations from memory."

    inputs = {
        "query": {
            "type": "string",
            "description": "The user query to search related memory for.",
        }
    }

    output_type = "string"

    def __init__(self, collection, embedding_fn, **kwargs):
        super().__init__(**kwargs)
        self.collection = collection
        self.embedding_fn = embedding_fn


    def forward(self, query: str) -> str:
        print(f"[DEBUG] Retriever query: {query!r}")

        embedding = self.embedding_fn.get_text_embedding(query)
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=10,  # fetch more for filtering
        )

        candidates = results.get("documents", [[]])[0]
        for i, d in enumerate(candidates, 1):
            print(f"[DEBUG] Matching records from DB  {i}. {d!r}")
        if not candidates:
            return None

        query_words = set(query.lower().split())
        scored = []

        for doc in candidates:
            doc_clean = doc.strip().lower()
            if len(doc_clean.split()) < 2 or doc_clean.endswith("?"):
                continue

            doc_words = set(doc_clean.split())
            overlap = len(query_words & doc_words)
            score = overlap / len(query_words)  
            if score >= 0.3:  
                scored.append((score, doc.strip()))

        if not scored:
            return "No relevant memory was found related to this query"

        # Sort and return up to 3 relevant results
        scored.sort(reverse=True)
        top_docs = [doc for _, doc in scored[:3]]
        return "\n\n".join(top_docs)
