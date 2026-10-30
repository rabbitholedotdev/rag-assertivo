"""In-process retriever over the vendored corpus. No Qdrant, no ONNX, no keys."""
from __future__ import annotations

import math
import re
import unicodedata
from pathlib import Path

from ingest.chunk import ementa_of, load_corpus_file
from rag.documents import Chunk
from rag.retrieve import SINTETICO

CORPUS = Path(__file__).resolve().parent.parent / "corpus" / "texts"


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\W+", " ", s).strip()


def _tokens(s: str) -> set[str]:
    return {t for t in _norm(s).split() if len(t) > 2}


def load_chunks() -> list[Chunk]:
    out = []
    for path in sorted(CORPUS.glob("*.md")):
        parts = load_corpus_file(path)
        em = ementa_of(parts)
        if em:
            parts.append(em)
        for d in parts:
            out.append(Chunk(page_content=d["page_content"], metadata=dict(d["metadata"])))
    return out


class CorpusBackend:
    """Lexical stand-in for hybrid search. Dense ≈ sparse ≈ token overlap."""

    def __init__(self, chunks: list[Chunk] | None = None):
        self.chunks = chunks if chunks is not None else load_chunks()

    def search(
        self,
        query: str,
        k: int,
        *,
        source_types: list[str] | None = None,
        exclude_types: list[str] | None = None,
        mode: str = "hybrid",
        collection: str | None = None,
        exact: bool = False,
    ) -> list[tuple[Chunk, float]]:
        q = _tokens(query)
        scored = []
        for d in self.chunks:
            st = (d.metadata.get("source_type") or "").lower()
            if source_types and st not in {s.lower() for s in source_types}:
                continue
            if exclude_types and st in {s.lower() for s in exclude_types}:
                continue
            dt = _tokens(d.page_content + " " + (d.metadata.get("title") or ""))
            if not q or not dt:
                continue
            overlap = len(q & dt)
            # BM25-ish length penalty so short ementas don't always win.
            score = overlap / (1.0 + math.log(1 + len(dt)))
            if mode == "sparse":
                score = overlap
            scored.append((d, float(score)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(Chunk(page_content=d.page_content, metadata=dict(d.metadata)), s) for d, s in scored[:k]]


def lexical_rerank(question: str, texts: list[str]) -> list[float]:
    q = _tokens(question)
    out = []
    for t in texts:
        dt = _tokens(t)
        out.append(len(q & dt) / max(1, len(q)))
    return out
