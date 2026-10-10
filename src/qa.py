from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from typing import Any, Iterator

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

from .config import require_api_key, settings
from .vectorstore import search

SOURCES_MARK = "\n\n*Sources: "

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions about the user's PDF documents using ONLY the context below.\n\n"
            "Rules:\n"
            "- If the answer is not in the context, say exactly: I couldn't find this in the document.\n"
            "- Never use outside knowledge and never guess.\n"
            "- Be concise and lead with the answer. Use short paragraphs or flat bullet lists. "
            "No headings, no horizontal rules, and no tables unless the user asks.\n"
            "- Write math and symbols in plain text. Never use LaTeX or $ delimiters.\n"
            "- Cite pages inline like (p. 4), or (file.pdf, p. 4) when several documents are involved.\n"
            "- If the user asks you to shorten, rephrase or reformat your previous answer, "
            "do that using the conversation instead of the context. A one-liner means one sentence.\n\n"
            "Context:\n{context}",
        ),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)


@lru_cache(maxsize=1)
def get_llm() -> ChatGoogleGenerativeAI:
    require_api_key()
    kwargs: dict[str, Any] = {}
    if settings.llm_thinking:
        kwargs["thinking_config"] = {"thinking_level": settings.llm_thinking}
    return ChatGoogleGenerativeAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        **kwargs,
    )


def _text(content: Any) -> str:
    """Normalise LangChain / Gradio message content (str or list of parts) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "".join(parts)
    return str(content or "")


def to_messages(history: list[dict], limit: int = 6) -> list[BaseMessage]:
    """Last few turns as LangChain messages, with the sources footer stripped from answers."""
    messages: list[BaseMessage] = []
    for m in (history or [])[-limit:]:
        text = _text(m.get("content")).split(SOURCES_MARK)[0].strip()
        if not text:
            continue
        if m.get("role") == "assistant":
            messages.append(AIMessage(content=text))
        elif m.get("role") == "user":
            messages.append(HumanMessage(content=text))
    return messages


def retrieval_query(question: str, history: list[dict]) -> str:
    """Short follow-ups ('make it shorter') borrow the previous question for retrieval,
    with no extra LLM call."""
    if history and len(question.split()) < 8:
        for m in reversed(history):
            if m.get("role") == "user":
                return f"{_text(m.get('content'))}\n{question}"
    return question


def format_context(hits: list[tuple[Document, float]]) -> str:
    return "\n\n".join(
        f"[{d.metadata.get('source', '?')}, p. {d.metadata.get('page', '?')}]\n{d.page_content}"
        for d, _ in hits
    )


def format_sources(hits: list[tuple[Document, float]]) -> str:
    pages: dict[str, set[int]] = defaultdict(set)
    for d, _ in hits:
        pages[d.metadata.get("source", "?")].add(int(d.metadata.get("page", 0)))
    if not pages:
        return ""
    parts = [f"{name} p. {', '.join(str(p) for p in sorted(nums))}" for name, nums in pages.items()]
    return SOURCES_MARK + "; ".join(parts) + "*"


def answer_stream(question: str, history: list[dict], digests: list[str]) -> Iterator[str]:
    """Yield the growing answer string; the final yield includes the sources line."""
    if not digests:
        yield "Add a PDF first, then ask your question."
        return

    hits = search(digests, retrieval_query(question, history))
    if not hits:
        yield "I couldn't find anything relevant in the uploaded documents."
        return

    chain = ANSWER_PROMPT | get_llm()
    payload = {
        "context": format_context(hits),
        "history": to_messages(history),
        "question": question,
    }
    answer = ""
    for chunk in chain.stream(payload):
        answer += _text(chunk.content)
        yield answer
    yield answer + format_sources(hits)