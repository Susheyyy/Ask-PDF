from __future__ import annotations

import html
import threading
import time
from pathlib import Path

import gradio as gr

from src.config import settings
from src.qa import answer_stream
from src.vectorstore import ingest_pdf, warmup

THEME = gr.themes.Base(
    primary_hue="pink",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_lg,
    font=[
        gr.themes.GoogleFont("Google Sans Flex"),
        "Google Sans",
        "ui-sans-serif",
        "system-ui",
        "sans-serif",
    ],
    font_mono=["ui-monospace", "SFMono-Regular", "Consolas", "monospace"],
)

FORCE_DARK_JS = "() => { document.body.classList.add('dark'); }"

CSS = """
:root, body, .dark, body.dark, .gradio-container {
    --bg: #0D0D11;
    --surface: #14141A;
    --surface-2: #1B1B23;
    --line: #262630;
    --text: #EDEDF2;
    --muted: #8E8E9B;
    --accent: #F4A3B8;
    --accent-ink: #1B0E13;
    --accent-soft: rgba(244, 163, 184, .10);
    --accent-line: rgba(244, 163, 184, .32);
    --danger: #FF8E9B;

    --body-background-fill: #0D0D11 !important;
    --background-fill-primary: #0D0D11 !important;
    --background-fill-secondary: #14141A !important;
    --block-background-fill: transparent !important;
    --block-border-width: 0px !important;
    --block-border-color: #262630 !important;
    --border-color-primary: #262630 !important;
    --input-background-fill: #14141A !important;
    --input-background-fill-focus: #14141A !important;
    --input-border-color: #262630 !important;
    --input-border-color-focus: rgba(244, 163, 184, .45) !important;
    --input-shadow: none !important;
    --input-shadow-focus: none !important;
    --body-text-color: #EDEDF2 !important;
    --body-text-color-subdued: #8E8E9B !important;
    --block-label-text-color: #8E8E9B !important;
    --button-primary-background-fill: #F4A3B8 !important;
    --button-primary-background-fill-hover: #F8B9C8 !important;
    --button-primary-text-color: #1B0E13 !important;
    --button-primary-border-color: #F4A3B8 !important;
    --button-secondary-background-fill: transparent !important;
    --button-secondary-background-fill-hover: rgba(244, 163, 184, .10) !important;
    --button-secondary-text-color: #F4A3B8 !important;
    --button-secondary-border-color: rgba(244, 163, 184, .32) !important;
    --color-accent: #F4A3B8 !important;
    --color-accent-soft: rgba(244, 163, 184, .10) !important;
    --link-text-color: #F4A3B8 !important;
    --link-text-color-hover: #F8B9C8 !important;
    --link-text-color-active: #F8B9C8 !important;
    --color-text-link: #F4A3B8 !important;
    --loader-color: #F4A3B8 !important;
    --shadow-drop: none !important;
    --shadow-drop-lg: none !important;
}

html, body, .gradio-container { background: var(--bg) !important; color: var(--text); }
.gradio-container { max-width: 100% !important; padding: 0 !important; }
footer { display: none !important; }

* { scrollbar-width: thin; scrollbar-color: #2B2B36 transparent; }
*::-webkit-scrollbar { width: 8px; height: 8px; }
*::-webkit-scrollbar-thumb { background: #2B2B36; border-radius: 8px; }
*::-webkit-scrollbar-track { background: transparent; }

#shell { gap: 0 !important; flex-wrap: nowrap !important; min-height: 100dvh; }
#side {
    flex: 0 0 320px !important; min-width: 320px !important; max-width: 320px;
    padding: 28px 22px !important; gap: 14px !important;
    background: var(--surface); border-right: 1px solid var(--line);
}
#main { flex: 1 1 auto !important; min-width: 0; padding: 22px 40px 24px !important; }
@media (max-width: 860px) {
    #shell { flex-wrap: wrap !important; }
    #side { flex: 1 1 100% !important; max-width: 100%; min-width: 0 !important; }
    #main { padding: 16px !important; }
}

.brand { display: flex; align-items: center; gap: 11px; }
.brand .mark {
    width: 26px; height: 26px; border-radius: 9px; background: var(--accent);
    display: grid; place-items: center;
}
.brand .mark::after { content: ""; width: 9px; height: 9px; border-radius: 50%; background: var(--bg); }
.brand .name { font-size: 1.2rem; font-weight: 600; letter-spacing: -.01em; }
.tagline { margin: 6px 0 0; font-size: .86rem; color: var(--muted); line-height: 1.45; }
.model-line { margin-top: 10px; font-size: .74rem; color: var(--muted); font-family: ui-monospace, Consolas, monospace; }
.model-line b { color: var(--accent); font-weight: 500; }
.label { margin: 10px 0 0; font-size: .72rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }

#add-btn { width: 100%; border-radius: 12px !important; font-weight: 600; }
#files { background: transparent !important; border: 1px dashed var(--line) !important; border-radius: 12px !important; }
#files .file-preview, #files table { font-size: .85rem; }

.doc {
    display: flex; gap: 12px; align-items: flex-start;
    padding: 11px 13px; margin-bottom: 8px; border-radius: 12px;
    background: var(--surface-2); border: 1px solid var(--line);
}
.doc .dot { width: 8px; height: 8px; border-radius: 50%; margin-top: 6px; flex: none; background: var(--muted); }
.doc.ok .dot { background: var(--accent); }
.doc.err { border-color: rgba(255, 142, 155, .45); }
.doc.err .dot { background: var(--danger); }
.doc.err .doc-note { color: var(--danger); }
.doc.busy { border-color: var(--accent-line); }
.doc.busy .dot { background: var(--accent); animation: pulse 1.1s ease-in-out infinite; }
.doc-body { min-width: 0; flex: 1; }
.doc-name { font-size: .9rem; font-weight: 500; word-break: break-word; }
.doc-note { margin-top: 2px; font-size: .76rem; color: var(--muted); }
.bar { height: 3px; margin-top: 9px; border-radius: 3px; background: var(--line); overflow: hidden; }
.bar span { display: block; height: 100%; background: var(--accent); border-radius: 3px; transition: width .25s ease; }
.doc-empty { padding: 20px 12px; text-align: center; font-size: .86rem; color: var(--muted); }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: .3; } }

#chat { background: transparent !important; border: none !important; }
#chat .message-wrap { max-width: 820px; margin: 0 auto; padding-bottom: 12px; }

#chat .bubble.bot-row { max-width: 100% !important; }
#chat .bot, #chat .bot-row .message, #chat .message-bubble-border {
    background: transparent !important; border: none !important; box-shadow: none !important;
}
#chat .bot .message-content, #chat .bot-row .message-content { padding: 4px 2px !important; display: block; }

#chat .user, #chat .user-row .message {
    background: var(--accent-soft) !important;
    border: 1px solid var(--accent-line) !important;
    border-radius: 18px 18px 5px 18px !important;
    color: var(--text) !important;
}
#chat .user .message-content, #chat .user-row .message-content { padding: 9px 15px !important; }

#chat .message-wrap .prose, #chat .message-wrap .md { opacity: 1 !important; font-size: 1rem; line-height: 1.65; }
#chat .message-wrap > div p:not(:first-child),
#chat .message-wrap p:not(:first-child) { margin-top: .7em !important; }
#chat .message-wrap p { margin-bottom: 0 !important; }
#chat .message-wrap ul, #chat .message-wrap ol { margin: .5em 0 !important; padding-left: 1.3em !important; }
#chat .message-wrap li { margin: .25em 0 !important; }
#chat .message-wrap hr { margin: .9em 0 !important; border-color: var(--line) !important; }
#chat .message-wrap em:last-child { color: var(--muted); font-size: .84rem; font-style: normal; }
#chat .message-wrap code { background: var(--surface-2); border-radius: 6px; padding: 1px 6px; }

#chat .placeholder-content { align-items: center; justify-content: center; text-align: center; color: var(--muted); }
#chat .placeholder-content h2 { margin: 0 0 6px; color: var(--text); font-weight: 600; letter-spacing: -.01em; }

#msg textarea {
    background: var(--surface) !important; border: 1px solid var(--line) !important;
    border-radius: 16px !important; padding: 14px 18px !important; font-size: 1rem;
}
#msg textarea:focus { border-color: var(--accent-line) !important; }
#msg textarea::placeholder { color: var(--muted); }
.submit-button { background: var(--accent) !important; color: var(--accent-ink) !important; border-radius: 12px !important; }
.submit-button:hover:not(:disabled) { background: #F8B9C8 !important; }
.submit-button:disabled { opacity: .35; }
"""

SIDEBAR_HEAD = f"""
<div class="brand"><span class="mark"></span><span class="name">Ask PDF</span></div>
<p class="tagline">Ask questions about your documents. Answers cite the pages they came from.</p>
<div class="model-line">model <b>{html.escape(settings.llm_model)}</b></div>
"""

PLACEHOLDER = (
    "## What would you like to know?\n"
    "Add a PDF on the left, then ask anything about it."
)

Item = tuple[str, str, str, float]


def render_docs(items: list[Item]) -> str:
    if not items:
        return '<div class="doc-empty">No documents yet</div>'
    rows = []
    for kind, name, note, frac in items:
        bar = f'<div class="bar"><span style="width:{int(frac * 100)}%"></span></div>' if kind == "busy" else ""
        rows.append(
            f'<div class="doc {kind}"><span class="dot"></span><div class="doc-body">'
            f'<div class="doc-name">{html.escape(name)}</div>'
            f'<div class="doc-note">{html.escape(note)}</div>{bar}</div></div>'
        )
    return "".join(rows)


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


def process_files(files, session):
    """Index each PDF in a worker thread and stream live progress into the sidebar."""
    if not files:
        yield {"docs": {}}, render_docs([])
        return

    paths = [files] if isinstance(files, (str, Path)) else list(files)
    raws = [getattr(p, "name", p) for p in paths]
    names = [Path(r).name for r in raws]

    docs: dict[str, str] = {}
    done: list[Item] = []

    for i, (raw, name) in enumerate(zip(raws, names)):
        state = {"msg": "Starting", "frac": 0.0}
        box: dict = {}

        def work(raw=raw, state=state, box=box) -> None:
            try:
                box["out"] = ingest_pdf(raw, lambda m, f: state.update(msg=m, frac=f))
            except Exception as e:  
                box["err"] = e

        t0 = time.perf_counter()
        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        queued: list[Item] = [("wait", n, "Waiting", 0.0) for n in names[i + 1:]]
        while worker.is_alive():
            busy: Item = ("busy", name, state["msg"], state["frac"])
            yield {"docs": dict(docs)}, render_docs(done + [busy] + queued)
            time.sleep(0.3)
        worker.join()

        if "err" in box:
            done.append(("err", name, str(box["err"]), 1.0))
        else:
            digest, cached = box["out"]
            docs[name] = digest
            note = "Loaded from cache" if cached else f"Indexed in {time.perf_counter() - t0:.0f}s"
            done.append(("ok", name, note, 1.0))

    yield {"docs": docs}, render_docs(done)


def respond(message, history, session):
    digests = list((session or {"docs": {}})["docs"].values())
    try:
        yield from answer_stream(message, history, digests)
    except Exception as e:
        yield f"Something went wrong: {e}"


with gr.Blocks(title="Ask PDF", theme=THEME, css=CSS, js=FORCE_DARK_JS) as demo:
    session = gr.State({"docs": {}})

    with gr.Row(equal_height=False, elem_id="shell"):
        with gr.Column(elem_id="side", scale=0, min_width=320):
            gr.HTML(SIDEBAR_HEAD)
            gr.HTML('<div class="label">Documents</div>')
            add_btn = gr.UploadButton(
                "Add PDFs",
                file_types=[".pdf"],
                file_count="multiple",
                variant="secondary",
                elem_id="add-btn",
            )
            files = gr.File(
                show_label=False,
                file_types=[".pdf"],
                file_count="multiple",
                elem_id="files",
            )
            status = gr.HTML(render_docs([]))

        with gr.Column(elem_id="main", scale=1):
            chatbot = gr.Chatbot(
                type="messages",
                layout="bubble",
                height="calc(100dvh - 128px)",
                show_copy_button=True,
                show_label=False,
                container=False,
                allow_tags=False,
                placeholder=PLACEHOLDER,
                elem_id="chat",
                render=False,
            )
            gr.ChatInterface(
                fn=respond,
                type="messages",
                chatbot=chatbot,
                textbox=gr.Textbox(
                    placeholder="Ask about your documents",
                    container=False,
                    scale=7,
                    lines=1,
                    max_lines=6,
                    elem_id="msg",
                    render=False,
                ),
                additional_inputs=[session],
                show_progress="hidden",
            )

    add_btn.upload(add_files, inputs=[add_btn, files], outputs=[files], show_progress="hidden")
    files.change(
        process_files,
        inputs=[files, session],
        outputs=[session, status],
        show_progress="hidden",
    )


if __name__ == "__main__":
    print("Loading embedding model (the first run downloads it, which can take a minute)...")
    t0 = time.perf_counter()
    warmup()
    print(f"Embedding model ready in {time.perf_counter() - t0:.1f}s. Starting UI...")
    demo.queue().launch(share=settings.share)