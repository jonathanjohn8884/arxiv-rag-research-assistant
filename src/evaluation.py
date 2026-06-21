"""
RAG evaluation runner — RAGAS-style metrics via direct OpenAI calls.

Computes faithfulness, answer relevancy, context recall, and context
precision without depending on the ragas package (avoids langchain
version conflicts).

Usage:
    python -m src.evaluation --vector_store vector_stores/faiss_index
"""
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

from openai import OpenAI

from src.config import settings
from src.ingestion import load_faiss_index
from src.retriever import Retriever
from src.pipeline import build_graph


# ---------------------------------------------------------------------------
# Synthetic evaluation set — replace with real Q&A pairs for best results
# ---------------------------------------------------------------------------
EVAL_SAMPLES: List[Dict[str, str]] = [
    {
        "question": "What is the main contribution of this paper?",
        "ground_truth": "The paper presents a novel method that improves upon existing baselines.",
    },
    {
        "question": "What datasets were used in the experiments?",
        "ground_truth": "The experiments were conducted on standard benchmark datasets.",
    },
    {
        "question": "What is the proposed model architecture?",
        "ground_truth": "The model uses a transformer-based architecture with attention mechanisms.",
    },
    {
        "question": "What are the key results reported in the paper?",
        "ground_truth": "The proposed method achieves state-of-the-art performance on all evaluated benchmarks.",
    },
    {
        "question": "What are the limitations acknowledged by the authors?",
        "ground_truth": "The authors note computational cost and domain generalization as limitations.",
    },
]


# ---------------------------------------------------------------------------
# LLM-as-judge helpers
# ---------------------------------------------------------------------------

_client: OpenAI = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def _score(prompt: str) -> float:
    """Ask the LLM to return a score between 0 and 1. Returns 0.5 on failure."""
    response = _get_client().chat.completions.create(
        model=settings.llm_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an evaluation judge. "
                    "Respond with a single decimal number between 0 and 1. "
                    "No explanation, just the number."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=10,
    )
    try:
        return round(float(response.choices[0].message.content.strip()), 4)
    except ValueError:
        return 0.5


def score_faithfulness(answer: str, contexts: List[str]) -> float:
    """Is every claim in the answer supported by the retrieved context?"""
    context_text = "\n\n".join(contexts)
    return _score(
        f"Context:\n{context_text}\n\n"
        f"Answer:\n{answer}\n\n"
        "Rate from 0 to 1 how faithfully the answer is grounded in the context. "
        "1 = every statement is directly supported, 0 = answer contains hallucinations."
    )


def score_answer_relevancy(question: str, answer: str) -> float:
    """Does the answer actually address the question?"""
    return _score(
        f"Question: {question}\n\nAnswer: {answer}\n\n"
        "Rate from 0 to 1 how well the answer addresses the question. "
        "1 = directly and completely answers it, 0 = irrelevant or off-topic."
    )


def score_context_recall(question: str, contexts: List[str], ground_truth: str) -> float:
    """Does the retrieved context contain the information needed to answer correctly?"""
    context_text = "\n\n".join(contexts)
    return _score(
        f"Question: {question}\n\n"
        f"Expected answer: {ground_truth}\n\n"
        f"Retrieved context:\n{context_text}\n\n"
        "Rate from 0 to 1 how much of the expected answer can be derived from the context. "
        "1 = all required information is present, 0 = context is missing key information."
    )


def score_context_precision(question: str, contexts: List[str]) -> float:
    """Is the retrieved context relevant to the question (low noise)?"""
    context_text = "\n\n".join(contexts)
    return _score(
        f"Question: {question}\n\nRetrieved context:\n{context_text}\n\n"
        "Rate from 0 to 1 how relevant the retrieved context is to the question. "
        "1 = all context is highly relevant, 0 = context is mostly irrelevant noise."
    )


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    vector_store_path: str = None,
    output_path: str = "results/ragas_scores.json",
) -> Dict[str, float]:
    path = vector_store_path or settings.vector_store_path
    if not Path(path).exists():
        raise FileNotFoundError(
            f"No FAISS index at '{path}'. Ingest a PDF first."
        )

    vector_store = load_faiss_index(path)
    retriever = Retriever(vector_store)
    graph = build_graph(retriever)

    faith_scores, relevancy_scores, recall_scores, precision_scores = [], [], [], []

    print(f"Running {len(EVAL_SAMPLES)} evaluation samples…")
    for i, sample in enumerate(EVAL_SAMPLES, 1):
        print(f"  [{i}/{len(EVAL_SAMPLES)}] {sample['question'][:60]}…")

        state = graph.invoke({
            "question": sample["question"],
            "retrieved_chunks": [],
            "answer": "",
            "confidence": 0.0,
            "sources": [],
            "turn_count": 0,
            "is_clarification": False,
        })

        q = sample["question"]
        a = state["answer"]
        c = state["retrieved_chunks"]
        gt = sample["ground_truth"]

        faith_scores.append(score_faithfulness(a, c))
        relevancy_scores.append(score_answer_relevancy(q, a))
        recall_scores.append(score_context_recall(q, c, gt))
        precision_scores.append(score_context_precision(q, c))

    def avg(lst): return round(sum(lst) / len(lst), 4) if lst else 0.0

    scores: Dict[str, float] = {
        "faithfulness": avg(faith_scores),
        "answer_relevancy": avg(relevancy_scores),
        "context_recall": avg(recall_scores),
        "context_precision": avg(precision_scores),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(scores, f, indent=2)

    print("\n=== Evaluation Results (RAGAS-style) ===")
    for metric, value in scores.items():
        print(f"  {metric:<22}: {value:.4f}")
    print(f"\nSaved to {output_path}")

    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument("--vector_store", default=None, help="Path to FAISS index")
    parser.add_argument("--output", default="results/ragas_scores.json")
    args = parser.parse_args()

    run_evaluation(vector_store_path=args.vector_store, output_path=args.output)
