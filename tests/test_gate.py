import json

from eval.run_gate import aggregate, check, recusou, run
from rag.answer import NOT_FOUND


def test_recusou_detects_not_found():
    assert recusou(NOT_FOUND)
    assert not recusou("O prazo é de 5 anos.")


def test_aggregate_and_check_pass_on_identical(tmp_path):
    items = [
        {"tipo": "golden", "status": "ok", "facet": 0.8, "latencia_s": 0.2, "frases": [{"veredito": "ok"}]},
        {"tipo": "golden", "status": "ok", "facet": 1.0, "latencia_s": 0.3, "frases": [{"veredito": "ok"}]},
        {"tipo": "armadilha", "status": "nao_encontrado", "abstencao": True, "latencia_s": 0.1, "frases": []},
    ]
    agg = aggregate(items)
    assert agg["taxa_resposta"] == 1.0
    assert agg["armadilha_ok"] == 1.0
    reports = tmp_path
    for tag in ("base", "novo"):
        (reports / f"gate_{tag}.json").write_text(json.dumps({"aggregate": agg, "items": items}))
    v = check("novo", "base", reports=reports)
    assert v["ok"]


def test_check_fails_on_answer_rate_drop(tmp_path):
    base = {"taxa_resposta": 1.0, "faceta": 0.8, "falha_fechada": 0.0}
    novo = {"taxa_resposta": 0.8, "faceta": 0.8, "falha_fechada": 0.0}
    (tmp_path / "gate_base.json").write_text(json.dumps({"aggregate": base}))
    (tmp_path / "gate_novo.json").write_text(json.dumps({"aggregate": novo}))
    v = check("novo", "base", reports=tmp_path)
    assert v["ok"] is False


def test_run_fake_backend_writes_report(tmp_path, monkeypatch):
    import eval.run_gate as g
    monkeypatch.setattr(g, "REPORTS", tmp_path)
    monkeypatch.setattr(g, "GOLDEN", g.ROOT / "golden.jsonl")
    path = run("ci", backend_name="fake", k=4, reports=tmp_path)
    data = json.loads(path.read_text())
    assert data["aggregate"]["n"] >= 20
    assert data["aggregate"]["taxa_resposta"] > 0
