"""Generators. Default is extractive (no API key). Optional OpenAI / Anthropic."""
from __future__ import annotations

import json
import os
import re

from rag.config import RAG_GEN_MODEL, RAG_VERIFY_MODEL


def extractive_gen(prompt: str, timeout: float | None = None) -> str:  # noqa: ARG001
    """Deterministic stub: first overlapping sentence of each citable chunk.

    Used when no LLM key is set. Returns no sentences when the question tokens
    barely meet the chunks (traps should abstain).
    """
    qm = re.search(r"Pergunta:\s*(.*?)(?:\n\nTrechos:|\Z)", prompt, re.S)
    qtok = set(re.findall(r"[A-Za-zÀ-ÿ]{4,}", (qm.group(1) if qm else "").lower()))
    trechos = re.findall(
        r'<trecho n="(\d+)"[^>]*>\s*(.*?)\s*</trecho>',
        prompt, re.S,
    )
    frases = []
    for n, corpo in trechos[:4]:
        texto = re.sub(r"\s+", " ", corpo).strip()
        ttok = set(re.findall(r"[A-Za-zÀ-ÿ]{4,}", texto.lower()))
        if qtok and len(qtok & ttok) < max(2, len(qtok) // 4):
            continue
        sent = re.split(r"(?<=[.!?])\s+", texto, maxsplit=1)[0].strip()
        if len(sent) < 20:
            sent = texto[:240].strip()
        cit = sent[:180].strip()
        if not cit:
            continue
        frases.append({"texto": cit, "provas": [{"trecho": int(n), "citacao": cit}]})
    return json.dumps({"frases": frases}, ensure_ascii=False)


def _openai(prompt: str, model: str, timeout: float | None = None) -> str:
    import httpx
    key = os.environ["OPENAI_API_KEY"]
    r = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0},
        timeout=timeout or 60.0,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _anthropic(prompt: str, model: str, timeout: float | None = None) -> str:
    import httpx
    key = os.environ["ANTHROPIC_API_KEY"]
    r = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
        json={"model": model, "max_tokens": 2000, "messages": [{"role": "user", "content": prompt}]},
        timeout=timeout or 60.0,
    )
    r.raise_for_status()
    return r.json()["content"][0]["text"]


def make_generator(name: str | None = None):
    name = (name or RAG_GEN_MODEL or "extractive").lower()
    if name in ("extractive", "stub", ""):
        return extractive_gen
    if name.startswith("gpt") or name.startswith("openai"):
        model = name.split(":", 1)[-1] if ":" in name else os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return lambda p, timeout=None: _openai(p, model, timeout)
    if name.startswith("claude") or name.startswith("anthropic"):
        model = name.split(":", 1)[-1] if ":" in name else os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
        return lambda p, timeout=None: _anthropic(p, model, timeout)
    return extractive_gen


def make_verifier(name: str | None = None):
    name = name if name is not None else RAG_VERIFY_MODEL
    if not name:
        return None
    return make_generator(name)
