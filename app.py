from __future__ import annotations

from pathlib import Path

import gradio as gr

from src.config import settings
from src.qa import answer_stream
from src.vectorstore import ingest_pdf

EMPTY_SESSION = {"docs": {}}  


def process_files(files, session, progress=gr.Progress()):
    """Index every uploaded PDF. The File component holds the full current set, so rebuild from it."""
    if not files:
        return {"docs": {}}, "No documents loaded."

    paths = [files] if isinstance(files, (str, Path)) else list(files)
    docs: dict[str, str] = {}
    lines: list[str] = []

    for i, path in enumerate(paths):
        name = Path(getattr(path, "name", path)).name
        progress((i, len(paths)), desc=f"Processing {name}")
        try:
            digest, cached = ingest_pdf(getattr(path, "name", path))
            docs[name] = digest
            lines.append(f"✅ {name}" + (" (cached)" if cached else ""))
        except Exception as e:  
            lines.append(f"❌ {name}: {e}")

    return {"docs": docs}, "\n".join(lines)


def respond(message, history, session):
    digests = list((session or EMPTY_SESSION)["docs"].values())
    try:
        yield from answer_stream(message, history, digests)
    except Exception as e:
        yield f"❌ {e}"


with gr.Blocks(title="PDF Chatbot") as demo:
    gr.Markdown(
        f"# 📄 PDF Chatbot\nAsk questions about your PDFs. Model: `{settings.llm_model}`"
    )
    session = gr.State({"docs": {}})

    with gr.Row():
        files = gr.File(
            label="📤 Upload PDFs",
            file_types=[".pdf"],
            file_count="multiple",
        )
        status = gr.Textbox(label="Status", lines=4, interactive=False)

    files.change(process_files, inputs=[files, session], outputs=[session, status])

    gr.ChatInterface(
        fn=respond,
        type="messages",
        additional_inputs=[session],
        fill_height=True,
    )


if __name__ == "__main__":
    demo.queue().launch(share=settings.share)