from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from typing import Any, Iterator

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from .config import require_api_key, settings
from .vectorstore import search

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions about the user's PDF documents using ONLY the context below.\n"
            "- If the answer is not in the context, say: \"I couldn't find this in the document.\"\n"
            "- Do not use outside knowledge or guess.\n"
            "- Be concise, and mention the page number(s) you relied on, e.g. (p. 4).\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)

CONDENSE_PROMPT = ChatPromptTemplate.from_template(
    "Given the conversation and a follow-up question, rewrite the follow-up as a "
    "standalone question that can be understood without the conversation. "
    "If it is already standalone, return it unchanged. Return ONLY the question.\n\n"
    "Conversation:\n{history}\n\nFollow-up: {question}\n\nStandalone question:"
)


@lru_cache(maxsize=1)
def get_llm() -> ChatGoogleGenerativeAI:
    require_api_key()
    return ChatGoogleGenerativeAI(
        model=settings.llm_model, temperature=settings.llm_temperature
    )


def _text(content: Any) -> str:
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


def condense_question(question: str, history: list[dict], max_turns: int = 6) -> str:
    recent = [m for m in (history or []) if m.get("role") in {"user", "assistant"}][-max_turns:]
    if not recent:
        return question
    transcript = "\n".join(f"{m['role']}: {_text(m.get('content'))}" for m in recent)
    result = (CONDENSE_PROMPT | get_llm()).invoke({"history": transcript, "question": question})
    return _text(result.content).strip() or question


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
    lines = [
        f"- **{name}** - p. {', '.join(str(p) for p in sorted(nums))}"
        for name, nums in pages.items()
    ]
    return "\n\n---\nn**Sources**\n" + "\n".join(lines)


def answer_stream(question: str, history: list[dict], digests: list[str]) -> Iterator[str]:
    if not digests:
        yield "Please upload a PDF first."
        return

    standalone = condense_question(question, history)
    hits = search(digests, standalone)
    if not hits:
        yield "I couldn't find anything relevant in the uploaded documents."
        return

    chain = ANSWER_PROMPT | get_llm()
    answer = ""
    for chunk in chain.stream({"context": format_context(hits), "question": standalone}):
        answer += _text(chunk.content)
        yield answer
    yield answer + format_sources(hits)