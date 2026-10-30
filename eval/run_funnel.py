"""Run the retrieval funnel on the golden set.

    python -m eval.run_funnel --backend fake --tag demo
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.funnel import L_PADRAO, classificar, evidencia, marcas, resumo
from rag.retrieve import SINTETICO, retrieve

ROOT = Path(__file__).resolve().parent
GOLDEN = ROOT / "golden.jsonl"
REPORTS = ROOT / "reports"

CAMINHOS = {
    "cli": {"k": 15},
    "mcp": {"k": 6},
}


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _ids(docs) -> list[str]:
    return [f"{d.metadata.get('doc_id')}#{d.metadata.get('chunk')}" for d in docs]


def run_one(row: dict, caminho: str, backend, reranker_fn, L: float, cli_final_vivo: bool | None = None) -> dict:
    k = CAMINHOS[caminho]["k"]
    gold_id = (row.get("expected_ids") or [None])[0]
    keywords = row.get("facetas") or []
    trace: dict = {}
    retrieve(row["pergunta"], k=k, backend=backend, reranker_fn=reranker_fn, trace=trace)
    vivo = {
        "aula": bool(gold_id) and any(
            (d.metadata.get("doc_id") or "") == gold_id for d in (trace.get("pool") or [])
        ),
        "chunk": evidencia(
            [d for d in (trace.get("pool") or []) if (d.metadata.get("doc_id") or "") == gold_id] if gold_id else trace.get("pool") or [],
            keywords, L,
        ),
        "pool": evidencia(trace.get("pool") or [], keywords, L),
        "final": evidencia(trace.get("final") or [], keywords, L),
    }
    tem_ouro = bool(gold_id)
    n_chunks_ouro = sum(1 for d in (trace.get("pool") or []) if (d.metadata.get("doc_id") or "") == gold_id) if gold_id else None
    n_sint = sum(1 for d in (trace.get("fusao") or []) if (d.metadata.get("source_type") or "").lower() in SINTETICO)
    degrau = None if len(keywords) < 3 else classificar(vivo, tem_ouro, caminho, cli_final_vivo=cli_final_vivo)
    return {
        "id": row.get("id") or row["pergunta"][:80],
        "colecao": "publico_br",
        "golden": "legislacao",
        "caminho": caminho,
        "degrau": degrau,
        "marcas": ["sem_facetas"] if degrau is None else marcas(
            degrau, vivo, tem_ouro,
            {m: evidencia(trace.get(m) or [], keywords, L) for m in ("denso", "esparso", "fusao")},
            n_sint, n_chunks_ouro,
        ),
        "config": trace.get("config") or {"k": k},
        "ids_final": _ids(trace.get("final") or []),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="fake")
    ap.add_argument("--tag", default="demo")
    ap.add_argument("--L", type=float, default=L_PADRAO)
    args = ap.parse_args(argv)
    if args.backend == "fake":
        from eval.fake_backend import CorpusBackend, lexical_rerank
        backend, reranker_fn = CorpusBackend(), lexical_rerank
    else:
        backend, reranker_fn = None, None
    rows = [r for r in _load_jsonl(GOLDEN) if r.get("tipo") != "armadilha"]
    itens = []
    cli_vivo = {}
    for row in rows:
        cli = run_one(row, "cli", backend, reranker_fn, args.L)
        itens.append(cli)
        cli_vivo[cli["id"]] = cli["degrau"] == "viva"
    for row in rows:
        it = run_one(
            row, "mcp", backend, reranker_fn, args.L,
            cli_final_vivo=cli_vivo.get(row.get("id") or row["pergunta"][:80]),
        )
        itens.append(it)
    summary = resumo(itens)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"funil_{args.tag}.jsonl").write_text(
        "\n".join(json.dumps(i, ensure_ascii=False) for i in itens) + "\n", encoding="utf-8"
    )
    (REPORTS / f"funil_{args.tag}_resumo.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
