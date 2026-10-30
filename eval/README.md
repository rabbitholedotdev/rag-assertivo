# eval/ — citation-safety harness

Measures the real pipeline (`retrieve` + proved answer) on a golden set. The
question is not only "did it find the article?" but "did it cite something that
is not in the source?".

## Flow

1. Curated golden: `eval/golden.jsonl` (public Brazilian law).
2. Traps: `eval/traps.jsonl` — questions whose answer is **not** in the corpus.
   The correct behaviour is to abstain (`nao_encontrado`).
3. Frozen split: `eval/split.json` (`calibracao` vs `teste`). Tune on
   calibration only.
4. Run:

```bash
python -m eval.run_gate run --tag demo --backend fake
python -m eval.run_gate check demo --baseline demo
python -m eval.run_funnel --backend fake --tag demo
```

`--backend fake` is the CI path: lexical stand-in over `corpus/texts/`, no
Qdrant, no API key. `--backend qdrant` hits the real hybrid index.

## Metrics

| metric | meaning |
|---|---|
| `primary_recall@k` | gold document in the **citable** hits |
| `lesson_recall@k` | gold document in any hit (including ementa) |
| `content_recall@k` | 6-gram overlap with the gold text (survives republish) |
| `synthetic_share` | fraction of top-k that is ementa/resumo/qa |
| `numbers_hallucination_rate` | numbers in the answer missing from the primary source |
| `faithfulness` | lexical (default) or LLM judge |
| trap abstention | trap questions must not be answered from memory |

## Gate

`run_gate check` exits 0 only if answer rate and facet coverage do not drop
more than 5 points versus the tagged baseline, and fail-closed rate does not
rise. Phrase verdicts: `citacao_ausente`, `numero_sem_prova`, `nao_sustenta`,
`ressalva_omitida`.

## Funnel

Each question dies in at most one step (sum with `viva` = 100% of questions
that have ≥ 3 facets):

| step | dies because |
|---|---|
| 0 | gold document has no evidence |
| 1 | evidence is split across chunks |
| 2 | outside the candidate pool |
| 3 | rerank dropped it from top-k |
| 4 | CLI found it, MCP `k` cut it |
| 5 | generator ignored the evidence |
| viva | evidence reached the answer |

Acceptance for a retrieval change is `eval.funnel.aceite`: McNemar on `viva`,
latency +10% (or +25% if gain ≥ 6 per path), MCP and CLI must share k/pool,
ungrounded-sentence rate must not rise.

Production numbers (anonymized private corpus) live in the root README — they
are **not** from this public golden set.
