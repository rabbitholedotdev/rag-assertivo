from eval.fake_backend import CorpusBackend, lexical_rerank
from rag.retrieve import retrieve


def test_hybrid_finds_lgpd_definition():
    docs = retrieve(
        "o que é dado pessoal",
        k=5,
        backend=CorpusBackend(),
        reranker_fn=lexical_rerank,
        priorizar_sintese=False,
    )
    assert any("dado pessoal" in d.page_content.lower() for d in docs)


def test_pools_separados_keeps_primary():
    docs = retrieve(
        "direitos do titular",
        k=6,
        backend=CorpusBackend(),
        reranker_fn=lexical_rerank,
        pools_separados=True,
        penalidade_sintetico=0.5,
    )
    primary = [d for d in docs if (d.metadata.get("source_type") or "") == "lei"]
    assert primary, "split pools must still surface the statute text"


def test_penalidade_sintetico_lowers_ementa():
    be = CorpusBackend()
    q = "princípios da administração pública"

    def identity(_q, texts):
        return [1.0] * len(texts)

    off = retrieve(q, k=8, backend=be, reranker_fn=identity, penalidade_sintetico=0.0, pools_separados=False)
    on = retrieve(q, k=8, backend=be, reranker_fn=identity, penalidade_sintetico=0.9, pools_separados=False)
    # with a flat reranker, penalty is the only signal: ementa should drop
    def ementa_rank(docs):
        for i, d in enumerate(docs):
            if d.metadata.get("source_type") == "ementa":
                return i
        return len(docs)

    assert ementa_rank(on) >= ementa_rank(off)


def test_trace_has_stages():
    trace = {}
    retrieve(
        "saúde é direito de todos",
        k=4,
        backend=CorpusBackend(),
        reranker_fn=lexical_rerank,
        trace=trace,
        fetch_k=12,
    )
    for key in ("denso", "esparso", "fusao", "pool", "rerank", "final", "config"):
        assert key in trace
    assert trace["config"]["k"] == 4
    assert len(trace["final"]) <= 4


def test_multi_collection_survives_one_failure():
    class Boom:
        def search(self, *a, **kw):
            col = kw.get("collection")
            if col == "dead":
                raise RuntimeError("down")
            return CorpusBackend().search(*a, **kw)

    docs = retrieve(
        "dado pessoal",
        k=3,
        backend=Boom(),
        reranker_fn=lexical_rerank,
        collections=["dead", "publico_br"],
        priorizar_sintese=False,
    )
    assert docs
