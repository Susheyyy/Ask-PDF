from __future__ import annotations

import html
import threading
import time
from pathlib import Path

import gradio as gr

from src.config import settings
from src.qa import answer_stream
from src.vectorstore import ingest_pdf, warmup

MAX_DOCS = settings.max_docs

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
    --bg: #0B0C10;
    --surface: #111318;
    --surface-2: #181C25;
    --line: #22242E;
    --text: #E8EAF0;
    --muted: #858A9E;
    --accent: #F4A3B8;
    --accent-2: #C87FA4;
    --accent-ink: #1A0812;
    --accent-soft: rgba(244, 163, 184, .10);
    --accent-line: rgba(244, 163, 184, .28);
    --danger: #FF7E94;
    --danger-soft: rgba(255, 126, 148, .12);

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

html, body, .gradio-container { background: var(--bg) !important; color: var(--text) !important; }
.gradio-container { max-width: 100% !important; padding: 0 !important; }
footer { display: none !important; }

* { scrollbar-width: thin; scrollbar-color: #22242E transparent; }
*::-webkit-scrollbar { width: 6px; height: 6px; }
*::-webkit-scrollbar-thumb { background: #22242E; border-radius: 8px; }
*::-webkit-scrollbar-track { background: transparent; }

/* Shell */
#shell {
    gap: 0 !important; flex-wrap: nowrap !important;
    min-height: 100dvh; align-items: stretch !important;
}
#side {
    flex: 0 0 300px !important; min-width: 300px !important; max-width: 300px;
    padding: 24px 18px !important;
    background: var(--surface) !important;
    border-right: 1px solid var(--line);
    display: flex !important; flex-direction: column !important;
    gap: 14px !important;
    box-shadow: 2px 0 16px rgba(0,0,0,.35);
}
#main {
    flex: 1 1 auto !important; min-width: 0;
    padding: 0 !important;
    display: flex !important; flex-direction: column !important;
}
@media (max-width: 860px) {
    #shell { flex-wrap: wrap !important; }
    #side { flex: 1 1 100% !important; max-width: 100%; min-width: 0 !important; }
    #main { flex-direction: column !important; }
}

/* Brand */
.brand { display: flex; align-items: center; gap: 11px; flex-shrink: 0; }
.brand .mark {
    width: 28px; height: 28px; border-radius: 9px; flex-shrink: 0;
    background: linear-gradient(135deg, #F4A3B8 0%, #C87FA4 100%);
    display: grid; place-items: center;
    box-shadow: 0 3px 10px rgba(244,163,184,.28);
}
.brand .mark::after { content: ""; width: 10px; height: 10px; border-radius: 50%; background: var(--bg); }
.brand .name { font-size: 1.15rem; font-weight: 700; letter-spacing: -.02em; color: var(--text); }
.tagline { margin: 0; font-size: .82rem; color: var(--muted); line-height: 1.45; }
.label { font-size: .7rem; font-weight: 700; letter-spacing: .09em; text-transform: uppercase; color: var(--muted); }

/* Drop zone */
#drop {
    background: transparent !important;
    border: 1.5px dashed var(--line) !important;
    border-radius: 12px !important;
    min-height: 100px !important;
    transition: border-color .18s ease, background .18s ease;
}
#drop:hover { border-color: var(--accent-line) !important; background: rgba(244,163,184,.05) !important; }
#drop * { color: var(--muted) !important; }

/* Docs header */
#docs-head {
    display: flex !important; align-items: center !important;
    flex-wrap: nowrap !important; margin: 0 !important; padding: 0 !important;
    min-height: 0 !important;
}
#docs-head > * { min-height: 0 !important; }
#docs-head > :first-child { flex: 1 1 auto !important; }
#docs-head #clear-btn, #clear-btn {
    flex: 0 0 auto !important; align-self: center !important;
    width: auto !important; min-width: 0 !important; max-width: max-content !important;
    height: 24px !important; min-height: 0 !important; line-height: 24px !important;
    padding: 0 9px !important; margin: 0 !important;
    font-size: .72rem !important; font-weight: 600 !important; border-radius: 6px !important;
    color: var(--muted) !important;
    border: 1px solid var(--line) !important;
    background: transparent !important;
    transition: color .15s, border-color .15s, background .15s;
}
#docs-head #clear-btn:hover, #clear-btn:hover {
    color: var(--danger) !important;
    border-color: rgba(255,126,148,.3) !important;
    background: rgba(255,126,148,.08) !important;
}

/* Document cards - fix alignment */
.doc-slot {
    position: relative !important;
    display: block !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}
.doc-slot > .block, .doc-slot .block {
    padding: 0 !important;
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    min-height: 0 !important;
}
.doc {
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    gap: 10px !important;
    width: 100% !important;
    box-sizing: border-box !important;
    padding: 10px 40px 10px 12px !important;
    border-radius: 10px !important;
    background: var(--surface-2) !important;
    border: 1px solid var(--line) !important;
    margin-top: 6px !important;
    transition: border-color .15s ease, box-shadow .15s ease;
}
.doc:hover { border-color: var(--accent-line) !important; box-shadow: 0 2px 12px rgba(0,0,0,.3); }
.doc .dot {
    width: 7px; height: 7px; border-radius: 50%;
    flex-shrink: 0; background: var(--muted);
    transition: background .2s;
}
.doc-body { min-width: 0; flex: 1 1 auto; overflow: hidden; }
.doc.ok { border-left: 2.5px solid var(--accent) !important; }
.doc.ok .dot { background: var(--accent); box-shadow: 0 0 6px rgba(244,163,184,.4); }
.doc.err { border-left: 2.5px solid var(--danger) !important; }
.doc.err .dot { background: var(--danger); }
.doc.err .doc-note { color: var(--danger) !important; }
.doc.busy { border-color: var(--accent-line) !important; }
.doc.busy .dot { background: var(--accent); animation: pulse 1s ease-in-out infinite; }
.doc-name {
    font-size: .875rem; font-weight: 500; color: var(--text);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    line-height: 1.4; display: block;
}
.doc-note { margin-top: 2px; font-size: .73rem; color: var(--muted); line-height: 1.3; display: block; }
.bar { height: 2px; margin-top: 7px; border-radius: 2px; background: var(--line); overflow: hidden; }
.bar span { display: block; height: 100%; background: var(--accent); border-radius: 2px; transition: width .3s ease; }
.doc-empty {
    padding: 20px 10px; text-align: center; font-size: .85rem;
    color: var(--muted); border: 1px dashed var(--line);
    border-radius: 10px; line-height: 1.4;
}
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: .2; } }

.doc-slot .rm {
    position: absolute !important; top: 50% !important; right: 8px !important;
    transform: translateY(-50%) !important; z-index: 5;
    width: 24px !important; min-width: 24px !important; height: 24px !important;
    padding: 0 !important; border: none !important; border-radius: 6px !important;
    background: transparent !important; color: var(--muted) !important;
    font-size: 1rem !important; line-height: 24px !important;
    display: flex !important; align-items: center !important; justify-content: center !important;
    transition: color .15s, background .15s;
}
.doc-slot .rm:hover { color: var(--danger) !important; background: rgba(255,126,148,.12) !important; }

/* Chat */
#chat { background: transparent !important; border: none !important; flex: 1 1 auto !important; }
#chat .message-wrap { max-width: 780px; margin: 0 auto; padding: 20px 24px 8px; }

#chat .bubble.bot-row { max-width: 100% !important; }
#chat .bot, #chat .bot-row .message, #chat .message-bubble-border {
    background: transparent !important; border: none !important; box-shadow: none !important;
}
#chat .bot .message-content, #chat .bot-row .message-content { padding: 4px 0 !important; display: block; }

#chat .user, #chat .user-row .message {
    background: var(--surface-2) !important;
    border: 1px solid var(--line) !important;
    border-radius: 16px 16px 4px 16px !important;
    color: var(--text) !important;
}
#chat .user .message-content, #chat .user-row .message-content { padding: 10px 16px !important; }

#chat .message-wrap .prose, #chat .message-wrap .md { opacity: 1 !important; font-size: .97rem; line-height: 1.68; }
#chat .message-wrap > div p:not(:first-child),
#chat .message-wrap p:not(:first-child) { margin-top: .65em !important; }
#chat .message-wrap p { margin-bottom: 0 !important; }
#chat .message-wrap ul, #chat .message-wrap ol { margin: .45em 0 !important; padding-left: 1.4em !important; }
#chat .message-wrap li { margin: .22em 0 !important; }
#chat .message-wrap hr { margin: .9em 0 !important; border-color: var(--line) !important; }
#chat .message-wrap em:last-child { color: var(--muted); font-size: .83rem; font-style: normal; }
#chat .message-wrap code { background: var(--surface-2); border: 1px solid var(--line); border-radius: 5px; padding: 1px 6px; font-size: .88em; }
#chat .message-wrap pre { background: var(--surface-2) !important; border: 1px solid var(--line) !important; border-radius: 10px !important; padding: 12px 16px !important; }

#chat .placeholder-content {
    display: flex !important; flex-direction: column !important;
    align-items: center !important; justify-content: center !important;
    text-align: center !important; color: var(--muted);
}
#chat .placeholder-content h2 { margin: 0 0 8px !important; color: var(--text); font-size: 1.45rem !important; font-weight: 600; letter-spacing: -.02em; }
#chat .placeholder-content p { font-size: .93rem; color: var(--muted); max-width: 340px; line-height: 1.5; margin: 0 !important; }

/* Input bar */
#msg {
    border-top: 1px solid var(--line) !important;
    padding: 12px 24px 16px !important;
    background: var(--surface) !important;
    flex-shrink: 0 !important;
}
#msg > div { max-width: 780px; margin: 0 auto; padding: 0 !important; }
#msg textarea {
    background: var(--bg) !important;
    border: 1px solid var(--line) !important;
    border-radius: 12px !important;
    padding: 11px 16px !important;
    font-size: .97rem !important;
    color: var(--text) !important;
    box-shadow: none !important;
    transition: border-color .18s, box-shadow .18s !important;
    resize: none !important;
    line-height: 1.5 !important;
}
#msg textarea:focus {
    border-color: var(--accent-line) !important;
    box-shadow: 0 0 0 3px rgba(244,163,184,.08) !important;
    outline: none !important;
}
#msg textarea::placeholder { color: var(--muted) !important; }
.submit-button {
    background: var(--accent) !important;
    color: var(--accent-ink) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    border: none !important;
    box-shadow: 0 2px 10px rgba(244,163,184,.22) !important;
    transition: background .15s, box-shadow .15s, transform .1s !important;
}
.submit-button:hover:not(:disabled) {
    background: #F8BAC8 !important;
    box-shadow: 0 4px 16px rgba(244,163,184,.3) !important;
    transform: translateY(-1px) !important;
}
.submit-button:active:not(:disabled) { transform: translateY(0) !important; }
.submit-button:disabled { opacity: .35 !important; }
"""

SIDEBAR_HEAD = """
<div class="brand"><span class="mark"></span><span class="name">Ask PDF</span></div>
<p class="tagline">Intelligent Document Search and Retrieval</p>
"""

EMPTY_HTML = '<div class="doc-empty">No documents yet</div>'

PLACEHOLDER = (
    "## What would you like to know?\n"
    "Add a PDF on the left, then ask anything about it."
)

Item = tuple[str, str, str, float]

def render_doc(item: Item) -> str:
    kind, name, note, frac = item
    bar = f'<div class="bar"><span style="width:{int(frac * 100)}%"></span></div>' if kind == "busy" else ""
    return (
        f'<div class="doc {kind}"><span class="dot"></span><div class="doc-body">'
        f'<div class="doc-name">{html.escape(name)}</div>'
        f'<div class="doc-note">{html.escape(note)}</div>{bar}</div></div>'
    )


def pack(session: dict, items: list[Item]) -> tuple:
    updates: list = [session, gr.update(visible=not items)]
    for i in range(MAX_DOCS):
        if i < len(items):
            updates += [gr.update(visible=True), gr.update(value=render_doc(items[i]))]
        else:
            updates += [gr.update(visible=False), gr.update()]
    return tuple(updates)

def add_files(new, paths):
    merged = list(paths or [])
    seen = {Path(p).name for p in merged}
    skipped_limit = False
    for f in ([new] if isinstance(new, (str, Path)) else list(new or [])):
        raw = str(getattr(f, "name", f))
        name = Path(raw).name
        if name in seen:
            continue
        if len(merged) >= MAX_DOCS:
            skipped_limit = True
            continue
        merged.append(raw)
        seen.add(name)
    if skipped_limit:
        gr.Warning(f"Limit of {MAX_DOCS} documents reached. Remove one to add another.")
    return merged, None


def make_remove(index: int):
    def remove(paths):
        paths = list(paths or [])
        if 0 <= index < len(paths):
            paths.pop(index)
        return paths

    return remove


def process_files(paths, session):

    paths = list(paths or [])
    names = [Path(p).name for p in paths]
    prev = session or {}
    prev_docs, prev_notes, prev_errs = prev.get("docs", {}), prev.get("notes", {}), prev.get("errors", {})

    docs: dict[str, str] = {}
    notes: dict[str, str] = {}
    errs: dict[str, str] = {}
    items: list[Item] = []

    for name in names:
        if name in prev_docs:
            docs[name], notes[name] = prev_docs[name], prev_notes.get(name, "Ready")
            items.append(("ok", name, notes[name], 1.0))
        elif name in prev_errs:
            errs[name] = prev_errs[name]
            items.append(("err", name, errs[name], 1.0))
        else:
            items.append(("wait", name, "Waiting", 0.0))

    def snapshot() -> dict:
        return {"docs": dict(docs), "notes": dict(notes), "errors": dict(errs)}

    yield pack(snapshot(), items)

    for i, (raw, name) in enumerate(zip(paths, names)):
        if items[i][0] != "wait":
            continue

        state = {"msg": "Starting", "frac": 0.0}
        box: dict = {}

        def work(raw=raw, state=state, box=box) -> None:
            try:
                box["out"] = ingest_pdf(raw, lambda m, f: state.update(msg=m, frac=f))
            except Exception as e:  # one bad file must not block the rest
                box["err"] = e

        t0 = time.perf_counter()
        worker = threading.Thread(target=work, daemon=True)
        worker.start()

        last = None
        while worker.is_alive():
            sig = (state["msg"], int(state["frac"] * 100))
            if sig != last:
                items[i] = ("busy", name, state["msg"], state["frac"])
                yield pack(snapshot(), items)
                last = sig
            time.sleep(0.25)
        worker.join()

        if "err" in box:
            errs[name] = str(box["err"])
            items[i] = ("err", name, errs[name], 1.0)
        else:
            digest, cached = box["out"]
            docs[name] = digest
            notes[name] = "Loaded from cache" if cached else f"Indexed in {time.perf_counter() - t0:.0f}s"
            items[i] = ("ok", name, notes[name], 1.0)
        yield pack(snapshot(), items)


def respond(message, history, session):
    digests = list((session or {"docs": {}}).get("docs", {}).values())
    try:
        yield from answer_stream(message, history, digests)
    except Exception as e:
        yield f"Something went wrong: {e}"

with gr.Blocks(title="Ask PDF", theme=THEME, css=CSS, js=FORCE_DARK_JS) as demo:
    session = gr.State({"docs": {}, "notes": {}, "errors": {}})
    paths = gr.State([])  

    with gr.Row(equal_height=False, elem_id="shell"):
        with gr.Column(elem_id="side", scale=0, min_width=300):
            gr.HTML(SIDEBAR_HEAD)
            files = gr.File(
                show_label=False,
                file_types=[".pdf"],
                file_count="multiple",
                elem_id="drop",
                height=100,
            )
            with gr.Row(elem_id="docs-head"):
                gr.HTML('<div class="label">Documents</div>')
                clear_btn = gr.Button("Clear all", size="sm", elem_id="clear-btn")

            empty = gr.HTML(EMPTY_HTML)
            slot_rows: list[gr.Row] = []
            slot_html: list[gr.HTML] = []
            slot_rm: list[gr.Button] = []
            for _ in range(MAX_DOCS):
                with gr.Row(visible=False, elem_classes="doc-slot") as row:
                    slot_html.append(gr.HTML(""))
                    slot_rm.append(gr.Button("\u00d7", elem_classes="rm"))
                slot_rows.append(row)

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

    PROCESS_OUTPUTS = [session, empty]
    for _row, _html in zip(slot_rows, slot_html):
        PROCESS_OUTPUTS += [_row, _html]

    def _process(trigger):
        return trigger.then(
            process_files,
            inputs=[paths, session],
            outputs=PROCESS_OUTPUTS,
            show_progress="hidden",
        )

    _process(files.upload(add_files, inputs=[files, paths], outputs=[paths, files], show_progress="hidden"))
    _process(clear_btn.click(lambda: [], outputs=[paths], show_progress="hidden"))
    for _i, _btn in enumerate(slot_rm):
        _process(_btn.click(make_remove(_i), inputs=[paths], outputs=[paths], show_progress="hidden"))


if __name__ == "__main__":
    print("Loading embedding model (the first run downloads it, which can take a minute)...")
    t0 = time.perf_counter()
    warmup()
    print(f"Embedding model ready in {time.perf_counter() - t0:.1f}s. Starting UI...")
    demo.queue().launch(share=settings.share)