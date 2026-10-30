from eval.funnel import aceite, classificar, mcnemar, resumo
from rag.documents import Chunk


def _d(text, **m):
    return Chunk(page_content=text, metadata=m)


def test_classificar_steps():
    assert classificar({"pool": False, "aula": False, "chunk": False, "final": False}, True, "cli") == "0"
    assert classificar({"pool": False, "aula": True, "chunk": False, "final": False}, True, "cli") == "1"
    assert classificar({"pool": False, "aula": True, "chunk": True, "final": False}, True, "cli") == "2"
    assert classificar({"pool": True, "aula": True, "chunk": True, "final": False}, True, "cli") == "3"
    assert classificar({"pool": True, "aula": True, "chunk": True, "final": False}, True, "mcp", cli_final_vivo=True) == "4"
    assert classificar({"pool": True, "aula": True, "chunk": True, "final": True}, True, "cli", resposta_cobre=False) == "5"
    assert classificar({"pool": True, "aula": True, "chunk": True, "final": True}, True, "cli") == "viva"


def test_mcnemar_symmetric():
    assert mcnemar(0, 0) == 1.0
    assert mcnemar(6, 0) < 0.05  # exact two-sided: 2/2^6 = 0.03125
    assert abs(mcnemar(5, 5) - 1.0) < 1e-9


def test_resumo_counts():
    itens = [
        {"colecao": "publico_br", "golden": "legislacao", "caminho": "cli", "degrau": "viva", "marcas": []},
        {"colecao": "publico_br", "golden": "legislacao", "caminho": "cli", "degrau": "2", "marcas": ["nenhum"]},
        {"colecao": "publico_br", "golden": "legislacao", "caminho": "cli", "degrau": None, "marcas": ["sem_facetas"]},
    ]
    r = resumo(itens)
    g = r["publico_br|legislacao|cli"]
    assert g["n"] == 3
    assert g["degraus"]["viva"] == 1
    assert g["degraus"]["2"] == 1


def _pack(ids_viva, caminho="cli", col="A"):
    itens = []
    for i, viva in ids_viva.items():
        itens.append({
            "id": i, "caminho": caminho, "degrau": "viva" if viva else "2",
            "colecao": col, "golden": "g", "marcas": [],
            "config": {"k": 8, "cand": 40, "pools_separados": False},
            "ids_final": ["x"],
        })
    return itens, {"latencia": {caminho: {"p50": 1.0, "p95": 2.0}}}


def test_aceite_rejects_small_gain():
    base = {"A": _pack({"q1": False, "q2": False, "q3": True, "q4": True})}
    # also need mcp path
    for pack in (base,):
        extra, lat = _pack({"q1": False, "q2": False, "q3": True, "q4": True}, caminho="mcp")
        pack["A"][0].extend(extra)
        pack["A"][1]["latencia"]["mcp"] = lat["latencia"]["mcp"]
    novo = {"A": _pack({"q1": True, "q2": False, "q3": True, "q4": True})}
    extra, lat = _pack({"q1": True, "q2": False, "q3": True, "q4": True}, caminho="mcp")
    novo["A"][0].extend(extra)
    novo["A"][1]["latencia"]["mcp"] = lat["latencia"]["mcp"]
    v = aceite(base, novo, "A", None, None)
    ganho = next(c for c in v["criterios"] if c["nome"] == "ganho_visado")
    assert ganho["ok"] is False  # +1 < 3
