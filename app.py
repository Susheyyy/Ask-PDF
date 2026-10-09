from __future__ import annotations

import html
import time
from pathlib import Path

import gradio as gr

from src.config import settings
from src.qa import answer_stream
from src.vectorstore import ingest_pdf, warmup

THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="violet",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_lg,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
)

CSS = """

.gradio-container { max-width: 100% !important; padding: 14px 28px !important; }
footer { display: none !important; }

.app-header {
    display: flex; align-items: center; gap: 14px;
    padding: 14px 22px; margin-bottom: 14px; border-radius: 14px;
    background: linear-gradient(120deg, #6366f1 0%, #8b5cf6 60%, #ec4899 100%);
    box-shadow: 0 6px 20px rgba(99, 102, 241, .25);
}
.app-header .logo { font-size: 1.9rem; line-height: 1; }
.app-header h1 { margin: 0; font-size: 1.35rem; font-weight: 700; color: #fff; }
.app-header p  { margin: 2px 0 0; font-size: .86rem; color: rgba(255,255,255,.9); }
.app-header .model {
    margin-left: auto; padding: 5px 12px; border-radius: 999px;
    font-size: .8rem; font-family: ui-monospace, monospace;
    background: rgba(255,255,255,.2); color: #fff;
}

.side-title {
    margin: 14px 2px 8px; font-size: .78rem; font-weight: 600;
    letter-spacing: .06em; text-transform: uppercase; opacity: .65;
}


.doc-card {
    display: flex; align-items: flex-start; gap: 10px;
    padding: 10px 12px; margin-bottom: 8px; border-radius: 12px;
    border: 1px solid var(--border-color-primary);
    background: var(--background-fill-secondary);
}
.doc-card .icon { font-size: 1.1rem; line-height: 1.4; }
.doc-card .doc-name { font-size: .9rem; font-weight: 600; word-break: break-all; }
.doc-card .doc-note { font-size: .78rem; opacity: .7; margin-top: 2px; }
.doc-card.err { border-color: #ef4444; background: rgba(239, 68, 68, .08); }
.doc-card.err .doc-note { color: #ef4444; opacity: 1; }
.doc-empty {
    padding: 22px 12px; text-align: center; font-size: .88rem; opacity: .6;
    border: 1.5px dashed var(--border-color-primary); border-radius: 12px;
}
.doc-summary { font-size: .82rem; opacity: .75; margin: 2px 2px 8px; }
"""

HEADER = f"""
<div class="app-header">
  <div>
    <h1>Ask PDF</h1>
    <p>Chat with your documents. Answers are grounded in your PDFs, with page citations.</p>
  </div>
</div>
"""


def render_docs(items: list[tuple[str, str, str]]) -> str:
    """items: (kind 'ok'|'err', file name, note)."""
    if not items:
        return '<div class="doc-empty">No documents yet.<br>Upload one or more PDFs to begin.</div>'
    cards = []
    for kind, name, note in items:
        icon = "✅" if kind == "ok" else "⚠️"
        cards.append(
            f'<div class="doc-card {kind}"><div class="icon">{icon}</div><div>'
            f'<div class="doc-name">{html.escape(name)}</div>'
            f'<div class="doc-note">{html.escape(note)}</div></div></div>'
        )
    ready = sum(1 for k, _, _ in items if k == "ok")
    summary = f'<div class="doc-summary">{ready} of {len(items)} ready</div>'
    return summary + "".join(cards)


def process_files(files, session, progress=gr.Progress()):
    """Index every uploaded PDF. The File component holds the full current set, so rebuild from it."""
    if not files:
        return {"docs": {}}, render_docs([])

    paths = [files] if isinstance(files, (str, Path)) else list(files)
    total = len(paths)
    docs: dict[str, str] = {}
    items: list[tuple[str, str, str]] = []

    for i, item in enumerate(paths):
        raw = getattr(item, "name", item)
        name = Path(raw).name

        def on_stage(message: str, fraction: float, i=i, name=name) -> None:
            progress((i + fraction) / total, desc=f"{name}: {message}")

        t0 = time.perf_counter()
        try:
            digest, cached = ingest_pdf(raw, on_stage)
            docs[name] = digest
            note = "Loaded from cache" if cached else f"Indexed in {time.perf_counter() - t0:.0f}s"
            items.append(("ok", name, note))
        except Exception as e:  
            items.append(("err", name, str(e)))

    return {"docs": docs}, render_docs(items)


def add_files(new, current):
    """Append newly picked PDFs to the ones already in the list (skip duplicate names)."""
    merged = list(current or [])
    seen = {Path(getattr(f, "name", f)).name for f in merged}
    for f in ([new] if isinstance(new, (str, Path)) else list(new or [])):
        raw = getattr(f, "name", f)
        if Path(raw).name not in seen:
            merged.append(raw)
            seen.add(Path(raw).name)
    return merged


def respond(message, history, session):
    digests = list((session or {"docs": {}})["docs"].values())
    try:
        yield from answer_stream(message, history, digests)
    except Exception as e:
        yield f"❌ {e}"


with gr.Blocks(title="Ask PDF", theme=THEME, css=CSS) as demo:
    gr.HTML(HEADER)
    session = gr.State({"docs": {}})

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=320):
            gr.HTML('<div class="side-title">Upload</div>')
            add_btn = gr.UploadButton(
                "➕ Add PDFs",
                file_types=[".pdf"],
                file_count="multiple",
                variant="primary",
            )
            files = gr.File(
                show_label=False,
                file_types=[".pdf"],
                file_count="multiple",
                height=150,
            )
            gr.HTML('<div class="side-title">Documents</div>')
            status = gr.HTML(render_docs([]))

        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                type="messages",
                height="74vh",
                show_copy_button=True,
                show_label=False,
                placeholder="### Ask anything about your PDFs\nUpload a document on the left to get started.",
                render=False,
            )
            gr.ChatInterface(
                fn=respond,
                type="messages",
                chatbot=chatbot,
                textbox=gr.Textbox(
                    placeholder="Ask a question about your documents…",
                    container=False,
                    scale=7,
                    lines=1,
                    max_lines=5,
                    render=False,
                ),
                additional_inputs=[session],
            )

    add_btn.upload(add_files, inputs=[add_btn, files], outputs=[files])
    files.change(process_files, inputs=[files, session], outputs=[session, status])


if __name__ == "__main__":
    print("Loading embedding model (the first run downloads it, which can take a minute)...")
    t0 = time.perf_counter()
    warmup()
    print(f"Embedding model ready in {time.perf_counter() - t0:.1f}s. Starting UI...")
    demo.queue().launch(share=settings.share)