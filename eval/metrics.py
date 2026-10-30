"""Retrieval + citation-safety metrics. Pure Python, no model."""
from __future__ import annotations

import re
import unicodedata

from rag.numeros import numeros_nao_encontrados
from rag.retrieve import is_citavel

_NUM = re.compile(r"\d[\d.,]*\s*%?")
_ABST = re.compile(
    r"não\s+(encontr|verific|const|localiz|h[áa]\s+(essa|informa))|"
    r"não\s+foi\s+possível\s+verificar|não\s+é\s+possível\s+confirmar",
    re.I,
)
_TOKEN = re.compile(r"[a-z]{7,}")
_MOLDURA = set(
    "material trechos trecho verificado verificada fontes resposta pergunta "
    "conforme segundo apenas somente mesmo ainda porque portanto entretanto "
    "inclusive ementa".split()
)


def _vid(doc) -> str:
    return doc.metadata.get("doc_id") or ""


def primary_recall_at_k(docs, gold_id: str) -> float:
    if not gold_id:
        return 0.0
    return 1.0 if any(is_citavel(d) and _vid(d) == gold_id for d in docs) else 0.0


def lesson_recall_at_k(docs, gold_id: str) -> float:
    """1.0 if ANY doc (primary or synthetic) belongs to the gold document."""
    if not gold_id:
        return 0.0
    prefix = re.compile(r"^(ementa|syn-(qa|resumo))-")
    return 1.0 if any(prefix.sub("", _vid(d)) == gold_id or _vid(d) == gold_id for d in docs) else 0.0


def context_precision(docs, gold_id: str) -> float:
    if not docs:
        return 0.0
    rel = sum(1 for d in docs if is_citavel(d) and _vid(d) == gold_id)
    return round(rel / len(docs), 3)


def synthetic_share(docs) -> float:
    if not docs:
        return 0.0
    return round(sum(1 for d in docs if not is_citavel(d)) / len(docs), 3)


def numbers_hallucination(answer: str, primary_context: str) -> dict:
    ans = re.split(r"\n⚠️", answer)[0]
    nums = [n.strip() for n in _NUM.findall(ans)]
    fora = numeros_nao_encontrados(ans, primary_context)
    return {"rate": round(len(fora) / len(nums), 3) if nums else 0.0, "n_total": len(nums), "fora": fora}


def abstained(answer: str) -> bool:
    return bool(_ABST.search(answer or ""))


def _shingles(text: str, n: int = 6) -> set:
    w = re.findall(r"\w+", text.lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def content_recall_at_k(docs, gold_texts: list[str], threshold: float = 0.5) -> float:
    gold = set().union(*map(_shingles, gold_texts)) if gold_texts else set()
    for d in docs:
        if not is_citavel(d):
            continue
        s = _shingles(d.page_content)
        if s and len(s & gold) / len(s) >= threshold:
            return 1.0
    return 0.0


def _sem_acento(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s.lower()) if not unicodedata.combining(c))


def facet_coverage(docs, keywords: list[str]) -> float | None:
    if not keywords:
        return None
    texto = _sem_acento(" ".join(d.page_content for d in docs if is_citavel(d)))
    achou = sum(1 for k in keywords if any(_sem_acento(alt.strip()) in texto for alt in k.split("|")))
    return round(achou / len(keywords), 3)


def answer_facet_coverage(answer: str, keywords: list[str]) -> float | None:
    if not keywords:
        return None
    texto = _sem_acento(re.split(r"⚠️", answer)[0])
    achou = sum(1 for k in keywords if any(_sem_acento(alt.strip()) in texto for alt in k.split("|")))
    return round(achou / len(keywords), 3)


def _tokens(texto: str) -> set:
    return set(_TOKEN.findall(_sem_acento(texto)))


def termos_sem_base(answer: str, ctx_docs, question: str = "") -> list[str]:
    prim = _tokens(" ".join(d.page_content for d in ctx_docs if is_citavel(d)))
    base = prim | _tokens(question) | _MOLDURA
    return sorted(_tokens(re.split(r"⚠️", answer)[0]) - base)


def unsupported_rate(answer: str, ctx_docs, question: str = "") -> float:
    toks = _tokens(re.split(r"⚠️", answer)[0]) - _MOLDURA
    return round(len(termos_sem_base(answer, ctx_docs, question)) / len(toks), 3) if toks else 0.0
