                                                               
from smolagents.tools import Tool
from datetime import datetime
import re
import string


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
        distances = results.get("distances", [[]])[0]
        scored_results = sorted(list(zip(distances, candidates)))
        scored_results = [(dist, doc) for dist, doc in scored_results
                          if len(doc.strip().lower()) > 2]

        final = scored_results
        # final = []
        # trans = str.maketrans("","", string.punctuation)
        # query_words = set(query.translate(trans).lower().split())
        #
        # for dist, doc in scored_results:
        #     doc_words = set(doc.strip().translate(trans).lower().split())
        #     overlap = len(query_words & doc_words)
        #     score = overlap / len(query_words)
        #     if score >= 0.3:
        #         final.append((dist, doc.strip()))

        if not final:
            return "No relevant memory was found related to this query."

        print('[DEBUG] Matching records and distances in DB')
        for i, r in enumerate(final, 1):
            print(f"[DEBUG] {i} | Distance {r[0]} | {r[1]!r}")

        return "\n\n".join((doc for _, doc in final[:5]))
