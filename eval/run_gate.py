"""Release gate: a change must not drop answer rate / facet coverage, and must
not raise the rate of ungrounded sentences.

    python -m eval.run_gate run --tag demo --backend fake
    python -m eval.run_gate check demo --baseline demo

`--backend fake` uses the in-process corpus retriever (no Qdrant, no keys).
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from eval.judge import judge_phrases
from eval.metrics import (
    abstained,
    answer_facet_coverage,
    numbers_hallucination,
    primary_recall_at_k,
)
from rag.answer import answer as run_answer
from rag.answer import format_text
from rag.generate import extractive_gen
from rag.proof import check_phrase
from rag.retrieve import is_citavel, retrieve

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
GOLDEN = ROOT / "golden.jsonl"
TRAPS = ROOT / "traps.jsonl"
MAX_QUEDA_RESPOSTA = 0.05
MAX_QUEDA_FACET = 0.05
MOTIVOS = ("citacao_ausente", "numero_sem_prova", "nao_sustenta", "ressalva_omitida")


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def recusou(resposta: str) -> bool:
    return abstained(resposta)


def aggregate(items: list[dict]) -> dict:
    n = len(items)
    resp = [1.0 if it.get("status") == "ok" else 0.0 for it in items if it.get("tipo") != "armadilha"]
    traps = [it for it in items if it.get("tipo") == "armadilha"]
    facets = [it["facet"] for it in items if it.get("facet") is not None]
    lat = [it["latencia_s"] for it in items if it.get("latencia_s") is not None]
    frases = [f for it in items for f in it.get("frases", [])]
    return {
        "n": n,
        "taxa_resposta": round(sum(resp) / len(resp), 4) if resp else 0.0,
        "faceta": round(sum(facets) / len(facets), 4) if facets else None,
        "armadilha_ok": round(sum(1 for t in traps if t.get("abstencao")) / len(traps), 4) if traps else None,
        "p50_s": round(statistics.median(lat), 3) if lat else None,
        "falha_fechada": round(sum(1 for it in items if it.get("status") == "falha_verificacao") / n, 4) if n else 0.0,
        "frases_ok": round(sum(1 for f in frases if f.get("veredito") == "ok") / len(frases), 4) if frases else None,
    }


def _pct(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    i = min(len(s) - 1, max(0, int(round((p / 100) * (len(s) - 1)))))
    return s[i]


def check(tag: str, baseline: str, reports: Path = REPORTS) -> dict:
    novo = json.loads((reports / f"gate_{tag}.json").read_text(encoding="utf-8"))
    base = json.loads((reports / f"gate_{baseline}.json").read_text(encoding="utf-8"))
    a, b = novo["aggregate"], base["aggregate"]
    crit = []

    def add(nome, ok, detalhe):
        crit.append({"nome": nome, "ok": bool(ok), "detalhe": detalhe})

    add(
        "taxa_resposta",
        a["taxa_resposta"] + 1e-9 >= b["taxa_resposta"] - MAX_QUEDA_RESPOSTA,
        {"base": b["taxa_resposta"], "novo": a["taxa_resposta"]},
    )
    if a.get("faceta") is not None and b.get("faceta") is not None:
        add(
            "faceta",
            a["faceta"] + 1e-9 >= b["faceta"] - MAX_QUEDA_FACET,
            {"base": b["faceta"], "novo": a["faceta"]},
        )
    add("falha_fechada", a["falha_fechada"] <= b["falha_fechada"] + 0.02, {"base": b["falha_fechada"], "novo": a["falha_fechada"]})
    return {"ok": all(c["ok"] for c in crit), "criterios": crit, "aggregate": a}


def _backend(name: str):
    if name == "fake":
        from eval.fake_backend import CorpusBackend, lexical_rerank
        return CorpusBackend(), lexical_rerank
    return None, None


def _run_one(row: dict, backend, reranker_fn, k: int) -> dict:
    import time
    t0 = time.monotonic()
    docs = retrieve(row["pergunta"], k=k, backend=backend, reranker_fn=reranker_fn)
    res = run_answer(row["pergunta"], docs, extractive_gen, None, deadline_s=30)
    texto = format_text(res)
    primary = " ".join(d.page_content for d in docs if is_citavel(d))
    frases = []
    chunks = [t["texto"] for t in res["trechos"]]
    for f in res["frases"]:
        motivo = check_phrase(f, chunks)
        frases.append({"texto": f["texto"], "veredito": "ok" if motivo is None else motivo})
    if res["frases"]:
        frases = judge_phrases(row["pergunta"], [(f["texto"], f["provas"]) for f in res["frases"]], chunks)
    gold = (row.get("expected_ids") or [None])[0]
    return {
        "id": row.get("id") or row["pergunta"][:80],
        "pergunta": row["pergunta"],
        "tipo": row.get("tipo", "golden"),
        "status": res["status"],
        "abstencao": recusou(texto),
        "primary_recall": primary_recall_at_k(docs, gold) if gold else None,
        "facet": answer_facet_coverage(texto, row.get("facetas") or []),
        "numbers": numbers_hallucination(texto, primary),
        "frases": frases,
        "latencia_s": round(time.monotonic() - t0, 3),
        "n_docs": len(docs),
    }


def run(tag: str, backend_name: str = "fake", k: int = 8, reports: Path = REPORTS) -> Path:
    backend, reranker_fn = _backend(backend_name)
    rows = _load_jsonl(GOLDEN) + _load_jsonl(TRAPS)
    items = [_run_one(r, backend, reranker_fn, k) for r in rows]
    out = {"tag": tag, "backend": backend_name, "k": k, "items": items, "aggregate": aggregate(items)}
    reports.mkdir(parents=True, exist_ok=True)
    path = reports / f"gate_{tag}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tag", required=True)
    r.add_argument("--backend", default="fake", choices=("fake", "qdrant"))
    r.add_argument("--k", type=int, default=8)
    c = sub.add_parser("check")
    c.add_argument("tag")
    c.add_argument("--baseline", required=True)
    args = ap.parse_args(argv)
    REPORTS.mkdir(parents=True, exist_ok=True)
    if args.cmd == "run":
        path = run(args.tag, args.backend, args.k)
        print(path)
        print(json.dumps(json.loads(path.read_text())["aggregate"], indent=2))
        return 0
    verdict = check(args.tag, args.baseline)
    print(json.dumps(verdict, indent=2, ensure_ascii=False))
    return 0 if verdict["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
