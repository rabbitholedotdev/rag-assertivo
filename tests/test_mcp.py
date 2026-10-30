from eval.fake_backend import CorpusBackend, lexical_rerank
from rag.mcp_server import list_sources, search


def test_list_sources_public_only():
    data = list_sources()
    assert "documents" in data
    urls = " ".join(d["url"] for d in data["documents"])
    assert "planalto.gov.br" in urls
    assert all("planalto.gov.br" in d["url"] for d in data["documents"])


def test_search_tool_with_injected_backend(monkeypatch):
    import rag.mcp_server as m
    import rag.retrieve as r

    be = CorpusBackend()
    orig = r.retrieve

    def wrapped(query, k=6, **kw):
        kw.setdefault("backend", be)
        kw.setdefault("reranker_fn", lexical_rerank)
        return orig(query, k, **kw)

    monkeypatch.setattr(m, "retrieve", wrapped)
    out = search("dado pessoal", k=4)
    assert out["count"] >= 1
    assert "citavel" in out["results"][0]


def test_healthz_app():
    from rag.mcp_server import build_http_app
    from starlette.testclient import TestClient
    app = build_http_app()
    c = TestClient(app)
    r = c.get("/healthz")
    assert r.status_code == 200
    assert r.json()["ok"] is True
