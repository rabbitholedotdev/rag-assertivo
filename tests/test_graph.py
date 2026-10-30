from eval.fake_backend import CorpusBackend, lexical_rerank
from rag.graph import build_graph, node_answer, node_rerank, node_retrieve, run_pipeline
from rag.generate import extractive_gen


def test_nodes_in_order():
    state = {
        "question": "o que é dado pessoal na LGPD?",
        "k": 4,
        "backend": CorpusBackend(),
        "reranker_fn": lexical_rerank,
        "gen": extractive_gen,
        "verifier": None,
        "tempos": {},
        "deadline_s": 10,
    }
    state.update(node_retrieve(state))
    assert state["docs"]
    state.update(node_rerank(state))
    assert state["docs"]
    state.update(node_answer(state))
    assert state["result"]["status"] in {"ok", "nao_encontrado", "falha_verificacao"}
    assert "texto_final" in state


def test_compiled_graph():
    app = build_graph()
    out = app.invoke({
        "question": "a saúde é direito de quem?",
        "k": 4,
        "backend": CorpusBackend(),
        "reranker_fn": lexical_rerank,
        "gen": extractive_gen,
        "verifier": None,
        "deadline_s": 10,
        "tempos": {},
    })
    assert out["result"]["status"] in {"ok", "nao_encontrado"}
    assert out["texto_final"]


def test_run_pipeline_wrapper():
    out = run_pipeline(
        "princípios da administração pública",
        k=4,
        backend=CorpusBackend(),
        reranker_fn=lexical_rerank,
        gen=extractive_gen,
        verifier=None,
        deadline_s=10,
    )
    assert "texto_final" in out
