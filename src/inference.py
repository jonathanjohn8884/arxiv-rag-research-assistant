import time
from pathlib import Path
from typing import Any, Dict, Optional

from src.config import settings
from src.ingestion import load_faiss_index
from src.retriever import Retriever
from src.pipeline import build_graph


def query_paper(
    question: str,
    vector_store_path: Optional[str] = None,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Query an ingested arXiv paper using the RAG pipeline.

    Returns:
        {
            "answer": str,
            "confidence": float,
            "sources": List[str],
            "retrieved_chunks": List[str],
            "latency_ms": int,
        }
    """
    start = time.time()

    path = vector_store_path or settings.vector_store_path
    if not Path(path).exists():
        raise FileNotFoundError(
            f"No FAISS index found at '{path}'. "
            "Upload and ingest a PDF first."
        )

    vector_store = load_faiss_index(path)
    retriever = Retriever(vector_store)
    graph = build_graph(retriever)

    initial_state = {
        "question": question,
        "retrieved_chunks": [],
        "answer": "",
        "confidence": 0.0,
        "sources": [],
        "turn_count": 0,
        "is_clarification": False,
    }

    final_state = graph.invoke(initial_state)
    latency_ms = int((time.time() - start) * 1000)

    if debug:
        print(f"[debug] turn_count={final_state['turn_count']}  confidence={final_state['confidence']}")
        print(f"[debug] latency={latency_ms}ms")

    # Deduplicate sources while preserving order
    seen: set = set()
    unique_sources = []
    for s in final_state["sources"]:
        if s not in seen:
            seen.add(s)
            unique_sources.append(s)

    return {
        "answer": final_state["answer"],
        "confidence": final_state["confidence"],
        "sources": unique_sources,
        "retrieved_chunks": final_state["retrieved_chunks"],
        "latency_ms": latency_ms,
    }
