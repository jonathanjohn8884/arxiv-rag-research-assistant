# arXiv RAG Research Assistant

A production-grade Retrieval-Augmented Generation (RAG) system for querying arXiv research papers. Built as a portfolio project demonstrating AI/ML engineering expertise.

**Key metrics (evaluated with LLM-as-judge on sample paper):**

| Metric | Score |
|---|---|
| Faithfulness | 0.96 |
| Answer Relevancy | 0.94 |
| Context Recall | 0.40 |
| Context Precision | 0.86 |
| Avg. Latency | ~2–3 s |

---

## Architecture

```
┌─────────────┐     ┌──────────────────────────────────────────────────┐
│  Gradio UI  │────▶│                 LangGraph Agent                   │
│  (port 7860)│     │                                                    │
└─────────────┘     │  START ──▶ retrieve ──▶ generate ──▶ evaluate    │
                    │                                    │               │
                    │              refine ◀──────────────┘ (if conf<0.7)│
                    │                │                                   │
                    │                └──────────────────────────▶ END   │
                    └──────────────────────────────────────────────────┘
                           │                    │
                    ┌──────┴──────┐    ┌────────┴────────┐
                    │  FAISS      │    │  OpenAI API      │
                    │  Vector     │    │  gpt-4o-mini     │
                    │  Store      │    │  text-embed-3-sm │
                    └─────────────┘    └─────────────────┘
                           │
                    ┌──────┴──────┐
                    │  LangSmith  │
                    │  Tracing    │
                    └─────────────┘
```

**Pipeline flow:**

1. PDF uploaded → chunked (500 tokens / 50 overlap) → embedded → FAISS index saved to disk
2. Question → `retrieve_node` fetches top-5 chunks by cosine similarity
3. `generate_node` calls `gpt-4o-mini` with few-shot prompt + retrieved context
4. `evaluate_node` scores faithfulness; if confidence < 0.7, triggers refine loop
5. `refine_node` expands the query and re-retrieves; max 5 iterations
6. Final answer returned with confidence score and source citations

---

## Tech Stack

| Component | Library |
|---|---|
| Agentic workflow | LangGraph |
| LLM & embeddings | OpenAI (`gpt-4o-mini`, `text-embedding-3-small`) |
| Vector store | FAISS (CPU) |
| PDF loading | LangChain + PyPDF |
| Observability | LangSmith |
| Evaluation | LLM-as-judge (OpenAI GPT) |
| UI | Gradio |

---

## Setup

### 1. Clone & install

```bash
git clone https://github.com/jonathanjohn8884/arxiv-rag-research-assistant.git
cd arxiv-rag-research-assistant
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY (and optionally LANGCHAIN_API_KEY)
```

### 3. Run the Gradio app

```bash
python api/main.py
# Open http://localhost:7860
```

Upload any arXiv PDF, click **Load Paper**, then start asking questions.

---

## Python API

```python
from src.ingestion import ingest_pdf
from src.inference import query_paper

# One-time ingestion
ingest_pdf("my_paper.pdf")

# Query
result = query_paper("What is the main contribution of this paper?")
print(result["answer"])
print(f"Confidence: {result['confidence']:.2f}")
print("Sources:", result["sources"])
```

---

## Evaluation

```bash
# Requires an ingested FAISS index
python -m src.evaluation --vector_store vector_stores/faiss_index

# Output: results/ragas_scores.json
```

Actual scores on a sample arXiv paper:
```json
{
  "faithfulness": 0.96,
  "answer_relevancy": 0.94,
  "context_recall": 0.40,
  "context_precision": 0.86
}
```

Metrics are computed using **LLM-as-judge** (GPT evaluates each response directly) rather than the ragas library, avoiding package dependency conflicts while producing equivalent scores.

The script ships with 5 generic Q&A samples. For best results replace `EVAL_SAMPLES` in [src/evaluation.py](src/evaluation.py) with real question/ground-truth pairs from your target paper — this will significantly improve the context recall score.

---

## Docker

```bash
docker build -t arxiv-qa:latest .
docker run --env-file .env -p 7860:7860 arxiv-qa:latest
```

---

## Deploy to HuggingFace Spaces

1. Create a new Space (Gradio SDK, Python 3.10)
2. Push this repo to the Space's git remote
3. Add `OPENAI_API_KEY` as a Space Secret
4. The Space will auto-build and serve the Gradio app

---

## Chunking Strategy

| Parameter | Value | Rationale |
|---|---|---|
| `chunk_size` | 500 tokens | Fits within GPT-4o-mini context; small enough for precise retrieval |
| `chunk_overlap` | 50 tokens | Preserves sentence continuity across chunk boundaries |
| `separators` | `\n\n`, `\n`, `. ` | Respects paragraph and sentence structure in research papers |

---

## Limitations & Future Work

- Evaluation scores depend on paper quality and ingested content; synthetic ground truth inflates some metrics
- No persistent multi-user session management (single FAISS index on disk)
- Confidence is heuristic-based; a dedicated NLI model would be more accurate
- Future: hybrid BM25 + dense retrieval, re-ranker, multi-paper search

---

## References

- [LangGraph docs](https://langchain-ai.github.io/langgraph/)
- [RAGAS paper](https://arxiv.org/abs/2309.15217)
- [OpenAI embeddings guide](https://platform.openai.com/docs/guides/embeddings)
- [FAISS](https://github.com/facebookresearch/faiss)
