"""Faithfulness judge. Lexical fallback is the default (no API key).

Optional LLM via JUDGE_MODEL=openai:gpt-4o-mini / anthropic:claude-haiku-4-5.
Phrase-level: code proof (`rag.proof.check_phrase`) first, then lexical/LLM.
"""
from __future__ import annotations

import re
import unicodedata

from rag.proof import check_phrase


def _sem_acento(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s.lower()) if not unicodedata.combining(c))


def _clean_answer(answer: str) -> str:
    return re.split(r"⚠️", answer or "")[0]


def lexical_supported(answer: str, context: str) -> float:
    """Fraction of 4-grams of the answer that appear in the context. 0..1."""
    ans = re.findall(r"\w+", _sem_acento(_clean_answer(answer)))
    ctx = " ".join(re.findall(r"\w+", _sem_acento(context)))
    if len(ans) < 4:
        return 1.0 if _sem_acento(_clean_answer(answer)) in _sem_acento(context) else 0.0
    grams = [" ".join(ans[i:i + 4]) for i in range(len(ans) - 3)]
    if not grams:
        return 0.0
    return round(sum(1 for g in grams if g in ctx) / len(grams), 3)


def faithfulness(answer: str, primary_context: str, **_kw) -> dict:
    score = lexical_supported(answer, primary_context)
    return {"score": score, "backend": "lexical"}


def judge_phrases(question: str, itens: list[tuple[str, list[str]]], chunks: list[str]) -> list[dict]:
    """One verdict per (frase, provas). Code proof first; lexical on the sentence vs cited chunks."""
    out = []
    for texto, provas in itens:
        frase = {"texto": texto, "provas": provas}
        motivo = check_phrase(frase, chunks)
        if motivo:
            out.append({"texto": texto, "veredito": motivo, "backend": "proof"})
            continue
        citados = [chunks[p["trecho"] - 1] for p in provas if isinstance(p.get("trecho"), int)]
        score = lexical_supported(texto, " ".join(citados))
        veredito = "ok" if score >= 0.4 else "nao_sustenta"
        out.append({"texto": texto, "veredito": veredito, "score": score, "backend": "lexical"})
    return out
