"""Hybrid retrieve (dense e5-large + BM25) + cross-encoder rerank + optional split pools.

The Qdrant backend is optional: tests inject a `SearchBackend`. Production uses
`QdrantBackend` from `rag.vectorstore`.
"""
from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Protocol

from rag.config import (
    COLLECTION_NAMES,
    RAG_PENALIDADE_SINTETICO,
    RAG_POOLS_SEPARADOS,
    RAG_RERANK,
)
from rag.documents import Chunk

# Synthetic layer: useful to locate a topic, never citable for numbers/claims.
SINTETICO = ("ementa", "resumo", "qa")

TITULOS_REBAIXADOS = re.compile(os.getenv("TITULOS_REBAIXADOS", r"(?!x)x"), re.I)
PENALIDADE_TITULO = float(os.getenv("PENALIDADE_TITULO", "0.0"))


class SearchBackend(Protocol):
    """One hybrid (or dense/sparse) search against one collection."""

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
        ...


RerankerFn = Callable[[str, list[str]], list[float]]


def _dedup_conteudo(docs: list[Chunk]) -> list[Chunk]:
    """Drop near-duplicate page_content; keep the first. Preserves order."""
    visto, out = set(), []
    for d in docs:
        chave = re.sub(r"\s+", " ", (d.page_content or "")[:200]).strip().lower()
        if chave and chave in visto:
            continue
        visto.add(chave)
        out.append(d)
    return out


def retrieve(
    question: str,
    k: int,
    *,
    source_type: str | list[str] | None = None,
    rerank: bool = True,
    fetch_k: int | None = None,
    priorizar_sintese: bool = True,
    pools_separados: bool | None = None,
    penalidade_sintetico: float | None = None,
    reranker_fn: RerankerFn | None = None,
    collections: list[str] | None = None,
    exact: bool = False,
    trace: dict | None = None,
    backend: SearchBackend | None = None,
    area: str | list[str] | None = None,
) -> list[Chunk]:
    """Hybrid search + rerank.

    - `priorizar_sintese` (default): when the caller did not pin `source_type`,
      inject a dedicated pool of synthetic candidates into the rerank. Without
      this, a large primary index drowns the small synthetic layer.
    - `pools_separados`: the main pool is primary-only (`must_not` synthetic) and
      the synthetic pool is separate; the two rankings merge before rerank.
      Stops a type-biased dense model from crowding primary chunks out of `cand`.
    - `penalidade_sintetico`: subtracted from the rerank score of ementa/resumo/qa.
    - `reranker_fn(question, texts) -> scores`: swap the cross-encoder (eval).
    - `trace`: filled with the set at each stage (denso, esparso, fusao, sintetico,
      pool, rerank, final) plus the effective `config`. Dense-only and sparse-only
      searches run only when `trace` is provided (funnel).
    """
    if backend is None:
        from rag.vectorstore import QdrantBackend
        backend = QdrantBackend()
    if pools_separados is None:
        pools_separados = RAG_POOLS_SEPARADOS
    if penalidade_sintetico is None:
        penalidade_sintetico = RAG_PENALIDADE_SINTETICO

    cols = list(collections or COLLECTION_NAMES)
    use_rerank = bool(rerank and RAG_RERANK) or reranker_fn is not None
    # fetch_k is honored even when this call skips the cross-encoder (LangGraph
    # retrieve node fetches a wide pool; the rerank node scores it).
    cand = fetch_k if fetch_k is not None else (max(k * 5, 30) if use_rerank else k)

    types = _as_list(source_type)
    wide = use_rerank or fetch_k is not None
    separar = bool(pools_separados and priorizar_sintese and not types and wide)
    com_pool_sintetico = bool(priorizar_sintese and not types and wide)
    n = len(cols)
    cand_c = cand if n == 1 else max(-(-cand // n), k)
    cand_s = max(k * 2, 12) if n == 1 else max(-(-(k * 2) // n), 6)

    if trace is not None:
        for etapa in ("denso", "esparso", "fusao", "sintetico"):
            trace[etapa] = []
        trace["config"] = {
            "k": k, "cand": cand_c,
            "cand_sintetico": cand_s if com_pool_sintetico else 0,
            "pools_separados": bool(separar),
            "penalidade_sintetico": penalidade_sintetico,
            "rerank": use_rerank, "exact": exact,
        }

    def _pools(col: str) -> list[tuple[Chunk, float]]:
        exclude = list(SINTETICO) if separar else None
        ps = backend.search(
            question, cand_c,
            source_types=types, exclude_types=exclude,
            mode="hybrid", collection=col, exact=exact,
        )
        ps_s: list[tuple[Chunk, float]] = []
        if com_pool_sintetico:
            ps_s = backend.search(
                question, cand_s,
                source_types=list(SINTETICO), exclude_types=None,
                mode="hybrid", collection=col, exact=exact,
            )
        if trace is not None:
            trace["fusao"] += [d for d, _ in ps]
            trace["sintetico"] += [d for d, _ in ps_s]
            for etapa, modo in (("denso", "dense"), ("esparso", "sparse")):
                trace[etapa] += [
                    d for d, _ in backend.search(
                        question, cand_c,
                        source_types=types, exclude_types=exclude,
                        mode=modo, collection=col, exact=exact,
                    )
                ]
        if n > 1:
            for d, _ in ps + ps_s:
                d.metadata["_colecao"] = col
        return ps + ps_s

    if n == 1:
        pairs = _pools(cols[0])
    else:
        with ThreadPoolExecutor(max_workers=n) as ex:
            res = list(ex.map(_one_or_err, [(_pools, c) for c in cols]))
        erros = [e for _, e in res if e]
        if len(erros) == n:
            raise RuntimeError("search failed on every collection: " + "; ".join(erros))
        pairs = [p for ps, _ in res for p in ps]

    seen, uniq = set(), []
    for d, s in pairs:
        chave = (
            d.metadata.get("_colecao"),
            d.metadata.get("doc_id"),
            d.metadata.get("chunk"),
        )
        if chave in seen:
            continue
        seen.add(chave)
        d.metadata["_score"] = round(float(s), 4)
        uniq.append(d)
    docs = uniq
    if area:
        wanted = {a.lower() for a in _as_list(area)}
        docs = [d for d in docs if (d.metadata.get("area") or "").lower() in wanted]

    for d in docs:
        d.metadata["relevance_score"] = None

    if use_rerank and docs:
        textos = [d.page_content for d in docs]
        if reranker_fn is None:
            from rag.vectorstore import get_reranker
            reranker = get_reranker()
            scores = list(reranker.rerank(question, textos)) if reranker else [0.0] * len(docs)
        else:
            scores = list(reranker_fn(question, textos))
        if penalidade_sintetico:
            scores = [
                sc - penalidade_sintetico
                if (d.metadata.get("source_type") or "").lower() in SINTETICO
                else sc
                for d, sc in zip(docs, scores)
            ]
        if PENALIDADE_TITULO:
            scores = [
                sc - PENALIDADE_TITULO
                if TITULOS_REBAIXADOS.search(d.metadata.get("title") or "")
                else sc
                for d, sc in zip(docs, scores)
            ]
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        out = []
        for d, sc in ranked:
            d.metadata["_rerank"] = round(float(sc), 4)
            d.metadata["relevance_score"] = d.metadata["_rerank"]
            out.append(d)
        ordenados = _dedup_conteudo(out)
    else:
        if n > 1:
            docs.sort(key=lambda d: d.metadata.get("_score") or 0.0, reverse=True)
        ordenados = _dedup_conteudo(docs)

    if trace is not None:
        trace.update(pool=list(docs), rerank=ordenados, final=ordenados[:k])
    return ordenados[:k]


def _as_list(v) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    return list(v)


def _one_or_err(item):
    fn, col = item
    try:
        return fn(col), None
    except Exception as e:  # noqa: BLE001 — one collection down must not kill the rest
        return [], f"{col}: {e}"


def rerank_docs(
    question: str,
    docs: list[Chunk],
    k: int,
    *,
    reranker_fn: RerankerFn | None = None,
    penalidade_sintetico: float | None = None,
) -> list[Chunk]:
    """Reorder an existing candidate list. Used as the LangGraph rerank node."""
    if penalidade_sintetico is None:
        penalidade_sintetico = RAG_PENALIDADE_SINTETICO
    if not docs:
        return []
    if reranker_fn is None:
        try:
            from rag.vectorstore import get_reranker
            reranker = get_reranker()
        except Exception:  # noqa: BLE001
            reranker = None
        if reranker is None:
            return docs[:k]
        scores = list(reranker.rerank(question, [d.page_content for d in docs]))
    else:
        scores = list(reranker_fn(question, [d.page_content for d in docs]))
    if penalidade_sintetico:
        scores = [
            sc - penalidade_sintetico
            if (d.metadata.get("source_type") or "").lower() in SINTETICO
            else sc
            for d, sc in zip(docs, scores)
        ]
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    out = []
    for d, sc in ranked:
        d.metadata["_rerank"] = round(float(sc), 4)
        d.metadata["relevance_score"] = d.metadata["_rerank"]
        out.append(d)
    return _dedup_conteudo(out)[:k]


def is_citavel(doc: Chunk) -> bool:
    return (doc.metadata.get("source_type") or "").lower() not in SINTETICO
