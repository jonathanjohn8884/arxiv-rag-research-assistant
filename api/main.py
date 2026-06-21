import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import shutil
import tempfile

import gradio as gr

from src.ingestion import ingest_pdf
from src.inference import query_paper
from src.config import settings

# Runtime vector store path (can be per-session in production)
_VECTOR_STORE_PATH = settings.vector_store_path


def handle_upload(pdf_file) -> str:
    """Ingest an uploaded PDF and build a FAISS index."""
    if pdf_file is None:
        return "No file uploaded."
    try:
        # pdf_file.name is a temp path provided by Gradio
        ingest_pdf(pdf_file, save_path=_VECTOR_STORE_PATH)
        filename = Path(pdf_file).name
        return f"✓ '{filename}' ingested successfully. You can now ask questions."
    except Exception as e:
        return f"Error during ingestion: {e}"


def handle_question(question: str, history: list):
    """Run RAG pipeline and stream answer back to the Gradio chatbot."""
    if not question.strip():
        yield history, "", 0.0, ""
        return

    if not Path(_VECTOR_STORE_PATH).exists():
        history = history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": "Please upload and load a PDF paper first."},
        ]
        yield history, "", 0.0, ""
        return

    try:
        result = query_paper(question, vector_store_path=_VECTOR_STORE_PATH)
    except FileNotFoundError as e:
        history = history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": str(e)},
        ]
        yield history, "", 0.0, ""
        return
    except Exception as e:
        history = history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": f"Error: {e}"},
        ]
        yield history, "", 0.0, ""
        return

    answer = result["answer"]
    confidence = result["confidence"]
    sources = "\n".join(result["sources"]) if result["sources"] else "—"

    history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    yield history, "", confidence, sources


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="arXiv Research Paper Q&A") as demo:
        gr.Markdown(
            """
            # arXiv Research Paper Q&A
            **Powered by LangGraph · OpenAI · FAISS · RAGAS**

            Upload a research paper PDF, then ask questions about it.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                pdf_upload = gr.File(
                    label="Upload PDF",
                    file_types=[".pdf"],
                    type="filepath",
                )
                load_btn = gr.Button("Load Paper", variant="primary")
                status_box = gr.Textbox(
                    label="Status",
                    interactive=False,
                    placeholder="Upload a PDF to get started…",
                )

            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Conversation", height=400)
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="e.g. What is the main contribution of this paper?",
                    lines=2,
                )
                ask_btn = gr.Button("Ask", variant="primary")

        with gr.Row():
            confidence_slider = gr.Slider(
                label="Answer Confidence",
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                interactive=False,
            )
            sources_box = gr.Textbox(
                label="Sources (file : page)",
                interactive=False,
                lines=3,
            )

        gr.Markdown(
            "_Tip: Confidence reflects how well the answer is grounded in retrieved passages._"
        )

        # Wire events
        load_btn.click(
            fn=handle_upload,
            inputs=[pdf_upload],
            outputs=[status_box],
        )

        ask_btn.click(
            fn=handle_question,
            inputs=[question_input, chatbot],
            outputs=[chatbot, question_input, confidence_slider, sources_box],
        )

        question_input.submit(
            fn=handle_question,
            inputs=[question_input, chatbot],
            outputs=[chatbot, question_input, confidence_slider, sources_box],
        )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Soft())
