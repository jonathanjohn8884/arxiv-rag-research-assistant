import os
from typing import List, Optional, Annotated
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langsmith import traceable

from src.config import settings
from src.retriever import Retriever

# Wire LangSmith tracing via environment
os.environ.setdefault("LANGCHAIN_TRACING_V2", settings.langchain_tracing_v2)
if settings.langchain_api_key:
    os.environ.setdefault("LANGCHAIN_API_KEY", settings.langchain_api_key)
os.environ.setdefault("LANGCHAIN_PROJECT", settings.langchain_project)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class QueryState(TypedDict):
    question: str
    retrieved_chunks: List[str]
    answer: str
    confidence: float
    sources: List[str]
    turn_count: int
    is_clarification: bool


def _default_state(question: str) -> QueryState:
    return QueryState(
        question=question,
        retrieved_chunks=[],
        answer="",
        confidence=0.0,
        sources=[],
        turn_count=0,
        is_clarification=False,
    )


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES = """
Example 1:
Context: "The transformer architecture uses self-attention mechanisms to process sequences in parallel."
Question: "What makes transformers fast?"
Answer: "Transformers achieve speed through self-attention, which processes all positions in a sequence simultaneously rather than sequentially."

Example 2:
Context: "We evaluate our method on three benchmarks: SQuAD, TriviaQA, and Natural Questions."
Question: "Which benchmarks were used?"
Answer: "The paper evaluates on SQuAD, TriviaQA, and Natural Questions."
"""


@traceable(run_type="retriever", name="retrieve_documents")
def retrieve_node(state: QueryState, retriever: Retriever) -> QueryState:
    chunks = retriever.retrieve(state["question"], k=settings.retrieval_k)
    return {
        **state,
        "retrieved_chunks": [c.content for c in chunks],
        "sources": [f"{c.source}:p{c.page}" for c in chunks],
        "turn_count": state["turn_count"] + 1,
    }


@traceable(run_type="llm", name="generate_answer")
def generate_node(state: QueryState) -> QueryState:
    llm = ChatOpenAI(
        model=settings.llm_model,
        openai_api_key=settings.openai_api_key,
        temperature=0.0,
    )
    context = "\n\n---\n\n".join(state["retrieved_chunks"])
    messages = [
        SystemMessage(content=(
            "You are an expert research assistant. Answer questions about academic papers "
            "using ONLY the provided context. Be concise and cite specific evidence. "
            "If the answer is not in the context, say so explicitly.\n\n"
            f"Few-shot examples:\n{FEW_SHOT_EXAMPLES}"
        )),
        HumanMessage(content=(
            f"Context:\n{context}\n\n"
            f"Question: {state['question']}\n\n"
            "Answer:"
        )),
    ]
    response = llm.invoke(messages)
    answer = response.content.strip()

    # Simple heuristic confidence: penalise if model expresses uncertainty
    uncertainty_phrases = ["i don't know", "not mentioned", "cannot find", "not in the context"]
    confidence = 0.5 if any(p in answer.lower() for p in uncertainty_phrases) else 0.85

    return {**state, "answer": answer, "confidence": confidence}


@traceable(run_type="chain", name="evaluate_response")
def evaluate_node(state: QueryState) -> QueryState:
    """
    Lightweight faithfulness check without a ground-truth label.
    Checks whether every sentence in the answer is grounded in retrieved chunks.
    Falls back to heuristic scoring when RAGAS can't run inline.
    """
    answer = state["answer"]
    context = " ".join(state["retrieved_chunks"]).lower()

    # Count answer sentences that have at least one keyword in context
    sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 10]
    if not sentences:
        return {**state, "confidence": 0.5}

    grounded = sum(
        1 for s in sentences
        if any(word in context for word in s.lower().split() if len(word) > 4)
    )
    faithfulness = grounded / len(sentences)

    # Blend with existing confidence
    blended = (state["confidence"] + faithfulness) / 2
    is_clarification = blended < 0.7

    return {**state, "confidence": round(blended, 4), "is_clarification": is_clarification}


@traceable(run_type="retriever", name="refine_retrieval")
def refine_node(state: QueryState, retriever: Retriever) -> QueryState:
    """Re-retrieve with an expanded query to improve low-confidence answers."""
    expanded = f"{state['question']} explain in detail methodology results"
    chunks = retriever.retrieve(expanded, k=settings.retrieval_k)
    return {
        **state,
        "retrieved_chunks": [c.content for c in chunks],
        "sources": [f"{c.source}:p{c.page}" for c in chunks],
        "is_clarification": False,
    }


def router_node(state: QueryState) -> str:
    """Conditional edge: decide whether to refine or end."""
    if state["turn_count"] > 5:
        return "end"
    if state["confidence"] < 0.6 and state["is_clarification"]:
        return "refine"
    return "end"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(retriever: Retriever) -> StateGraph:
    # Bind retriever into nodes that need it
    def _retrieve(state): return retrieve_node(state, retriever)
    def _generate(state): return generate_node(state)
    def _evaluate(state): return evaluate_node(state)
    def _refine(state): return refine_node(state, retriever)

    graph = StateGraph(QueryState)
    graph.add_node("retrieve", _retrieve)
    graph.add_node("generate", _generate)
    graph.add_node("evaluate", _evaluate)
    graph.add_node("refine", _refine)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "evaluate")
    graph.add_conditional_edges("evaluate", router_node, {"refine": "refine", "end": END})
    graph.add_edge("refine", "generate")

    return graph.compile()
