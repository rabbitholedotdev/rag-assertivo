"""LangGraph orchestration: retrieve → rerank → generate → proof-check → format.

Each node is a plain function over `RAGState`. Tests call nodes directly with a
fake backend; production compiles the graph once.

If `langgraph` is not installed, `build_graph()` raises and `run_pipeline()`
falls back to the same node sequence (the stub is explicit, not silent).
"""
from __future__ import annotations

import time
from typing import Any, Callable, TypedDict

from rag.answer import answer as run_answer
from rag.answer import format_text
from rag.config import RAG_DEADLINE_S, RAG_PENALIDADE_SINTETICO, RAG_POOLS_SEPARADOS
from rag.generate import make_generator, make_verifier
from rag.retrieve import rerank_docs, retrieve
from rag.tracing import span


class RAGState(TypedDict, total=False):
    question: str
    k: int
    docs: list
    result: dict
    texto_final: str
    tempos: dict
    backend: Any
    reranker_fn: Callable
    gen: Callable
    verifier: Callable | None
    deadline_s: float


def node_retrieve(state: RAGState) -> dict:
    """Hybrid candidates. Rerank is a separate node so the funnel can see both sets."""
    t0 = time.monotonic()
    with span("retrieve", k=state.get("k", 8)) as bag:
        docs = retrieve(
            state["question"],
            k=state.get("k", 8),
            rerank=False,
            fetch_k=max(state.get("k", 8) * 5, 30),
            pools_separados=RAG_POOLS_SEPARADOS,
            penalidade_sintetico=RAG_PENALIDADE_SINTETICO,
            backend=state.get("backend"),
            reranker_fn=None,
        )
        bag["output"] = {"n": len(docs)}
    return {"docs": docs, "tempos": {**state.get("tempos", {}), "busca_s": round(time.monotonic() - t0, 2)}}


def node_rerank(state: RAGState) -> dict:
    t0 = time.monotonic()
    with span("rerank") as bag:
        fn = state.get("reranker_fn")
        docs = rerank_docs(
            state["question"],
            list(state.get("docs") or []),
            k=state.get("k", 8),
            reranker_fn=fn,
        )
        bag["output"] = {"n": len(docs), "top": [d.metadata.get("title") for d in docs[:3]]}
    tempos = {**state.get("tempos", {}), "rerank_s": round(time.monotonic() - t0, 2)}
    return {"docs": docs, "tempos": tempos}


def node_answer(state: RAGState) -> dict:
    gen = state.get("gen") or make_generator()
    verifier = state["verifier"] if "verifier" in state else make_verifier()
    with span("answer_and_check") as bag:
        result = run_answer(
            state["question"],
            state.get("docs") or [],
            gen,
            verifier,
            deadline_s=state.get("deadline_s", RAG_DEADLINE_S),
        )
        result["tempos"]["busca_s"] = state.get("tempos", {}).get("busca_s", 0.0)
        texto = format_text(result)
        bag["output"] = {"status": result["status"], "n_frases": len(result["frases"])}
    return {"result": result, "texto_final": texto}


def build_graph():
    """Compile retrieve → rerank → answer. Raises if langgraph is missing."""
    from langgraph.graph import END, START, StateGraph

    g = StateGraph(RAGState)
    g.add_node("retrieve", node_retrieve)
    g.add_node("rerank", node_rerank)
    g.add_node("answer", node_answer)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank", "answer")
    g.add_edge("answer", END)
    return g.compile()


_APP = None


def run_pipeline(question: str, k: int = 8, **kw) -> dict:
    """Run the graph. If langgraph is not installed, run the same nodes in order."""
    state: RAGState = {"question": question, "k": k, "tempos": {}, **kw}
    try:
        global _APP
        if _APP is None:
            _APP = build_graph()
        out = _APP.invoke(state)
        return {"status": out["result"]["status"], "result": out["result"], "texto_final": out["texto_final"]}
    except ImportError:
        # Explicit stub: same nodes, no LangGraph runtime.
        state.update(node_retrieve(state))
        state.update(node_rerank(state))
        state.update(node_answer(state))
        return {"status": state["result"]["status"], "result": state["result"], "texto_final": state["texto_final"], "stub": True}
