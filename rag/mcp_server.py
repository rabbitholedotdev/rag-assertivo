"""MCP server: hybrid search + proved answer over the public PT-BR corpus.

Transports:
  MCP_TRANSPORT=stdio  (default, for Claude Desktop / Cursor)
  MCP_TRANSPORT=http   streamable HTTP + GET /healthz

Auth on HTTP is optional. When MCP_AUTH_TOKEN is set (>=16 chars), the client
must send `Authorization: Bearer <token>`. Wrong token → 401.
"""
from __future__ import annotations

import os
import time

from rag.config import (
    COLLECTION_NAME,
    COLLECTION_NAMES,
    MCP_AUTH_TOKEN,
    MCP_HOST,
    MCP_HTTP_PATH,
    MCP_PORT,
    MCP_TRANSPORT,
    RAG_DEADLINE_S,
    RAG_PENALIDADE_SINTETICO,
    RAG_POOLS_SEPARADOS,
)
from rag.generate import make_generator, make_verifier
from rag.graph import run_pipeline
from rag.retrieve import SINTETICO, is_citavel, retrieve

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # tests import tool functions without the SDK
    FastMCP = None  # type: ignore

mcp = FastMCP("rag-assertivo") if FastMCP is not None else None


def _tool(fn):
    if mcp is None:
        return fn
    try:
        return mcp.tool(
            annotations={"title": fn.__name__, "readOnlyHint": True, "openWorldHint": False}
        )(fn)
    except TypeError:
        return mcp.tool()(fn)


@_tool
def search(query: str, k: int = 6, area: str | None = None, source_type: str | None = None) -> dict:
    """Hybrid search (e5-large + BM25) + Jina rerank over the public corpus.

    Each hit has `citavel`. Synthetic hits (`ementa`/`resumo`/`qa`, citavel=false)
    may invent structure — locate the topic with them, cite only citavel=true.
    """
    docs = retrieve(
        query, k=k, area=area, source_type=source_type,
        pools_separados=RAG_POOLS_SEPARADOS,
        penalidade_sintetico=RAG_PENALIDADE_SINTETICO,
    )
    results = []
    for d in docs:
        m = d.metadata
        st = (m.get("source_type") or "").lower()
        item = {
            "title": m.get("title", ""),
            "doc_id": m.get("doc_id", ""),
            "area": m.get("area", ""),
            "article": m.get("article", ""),
            "source_type": st,
            "citavel": is_citavel(d),
            "url": m.get("url", ""),
            "score": m.get("_rerank", m.get("_score", 0)),
            "text": d.page_content,
        }
        if st in SINTETICO:
            item["aviso"] = (
                "Ementa/resumo gerado para navegação — pode conter estrutura inventada. "
                "Cite números e afirmações só de resultados citavel=true."
            )
        results.append(item)
    return {
        "count": len(results),
        "contrato": (
            "Cite números/passos/afirmações APENAS de resultados citavel=true. "
            "Itens citavel=false são ementas — use só para localizar o assunto."
        ),
        "results": results,
    }


@_tool
def answer(question: str, k: int = 8) -> dict:
    """Answer with per-sentence proof. Unverified text never leaves.

    Orchestrated by LangGraph: retrieve → rerank → generate → code-check → format.
    status: ok | nao_encontrado | falha_verificacao.
    """
    out = run_pipeline(question, k=k, gen=make_generator(), verifier=make_verifier(), deadline_s=RAG_DEADLINE_S)
    res = out["result"]
    res["cortadas"] = [{"motivo": c["motivo"]} for c in res["cortadas"]]
    return {**res, "texto_final": out["texto_final"]}


@_tool
def health() -> dict:
    """Is the vector store up? How many points? Distinguishes 'no hit' from 'store down'."""
    t0 = time.time()
    try:
        from rag.vectorstore import get_client
        client = get_client()
        if not client.collection_exists(COLLECTION_NAME):
            return {"ok": False, "collection": COLLECTION_NAME, "error": "collection does not exist"}
        n = client.count(COLLECTION_NAME, exact=False).count
        return {
            "ok": True,
            "collection": COLLECTION_NAME,
            "collections": COLLECTION_NAMES,
            "points": n,
            "latency_ms": round((time.time() - t0) * 1000),
        }
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "collection": COLLECTION_NAME,
            "error": str(e)[:200],
            "latency_ms": round((time.time() - t0) * 1000),
        }


@_tool
def list_sources() -> dict:
    """List documents in the public corpus (from corpus/sources.json)."""
    from pathlib import Path
    import json
    path = Path(__file__).resolve().parent.parent / "corpus" / "sources.json"
    return json.loads(path.read_text(encoding="utf-8"))


def build_http_app():
    """Starlette app: GET /healthz + MCP streamable HTTP at MCP_HTTP_PATH."""
    from starlette.applications import Starlette
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse, Response
    from starlette.routing import Mount, Route

    async def healthz(_request):
        return JSONResponse({"ok": True, "service": "rag-assertivo"})

    class BearerAuth(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            if request.url.path in ("/healthz", "/health"):
                return await call_next(request)
            token = MCP_AUTH_TOKEN
            if not token:
                return await call_next(request)
            got = request.headers.get("authorization") or ""
            if got != f"Bearer {token}":
                return Response("unauthorized", status_code=401)
            return await call_next(request)

    mcp_app = Starlette()
    if mcp is not None:
        for attr in ("streamable_http_app", "http_app", "sse_app"):
            fn = getattr(mcp, attr, None)
            if callable(fn):
                try:
                    mcp_app = fn()
                    break
                except Exception:  # noqa: BLE001
                    continue
    app = Starlette(routes=[
        Route("/healthz", healthz),
        Route("/health", healthz),
        Mount(MCP_HTTP_PATH.rstrip("/") or "/mcp", app=mcp_app),
    ])
    app.add_middleware(BearerAuth)
    return app


def main() -> None:
    if MCP_TRANSPORT == "http":
        import uvicorn
        uvicorn.run(build_http_app(), host=MCP_HOST, port=MCP_PORT)
        return
    if mcp is None:
        raise SystemExit("mcp package is required for stdio transport")
    mcp.run()


if __name__ == "__main__":
    main()
