from dataclasses import dataclass
from typing import List, Optional

from langchain_community.vectorstores import FAISS

from src.config import settings


@dataclass
class RetrievedChunk:
    content: str
    source: str
    page: int
    score: float


class Retriever:
    def __init__(self, vector_store: FAISS):
        self.vector_store = vector_store

    def retrieve(self, query: str, k: int = None) -> List[RetrievedChunk]:
        top_k = k or settings.retrieval_k
        results = self.vector_store.similarity_search_with_score(query, k=top_k)

        chunks = []
        for doc, score in results:
            chunks.append(
                RetrievedChunk(
                    content=doc.page_content,
                    source=doc.metadata.get("source", "unknown"),
                    page=doc.metadata.get("page", 0),
                    # FAISS L2 distance — convert to similarity (lower = more similar)
                    score=float(1 / (1 + score)),
                )
            )
        return chunks

    def retrieve_texts(self, query: str, k: int = None) -> List[str]:
        return [c.content for c in self.retrieve(query, k)]

    def retrieve_sources(self, query: str, k: int = None) -> List[str]:
        chunks = self.retrieve(query, k)
        seen: set = set()
        sources = []
        for c in chunks:
            key = f"{c.source}:p{c.page}"
            if key not in seen:
                seen.add(key)
                sources.append(key)
        return sources
