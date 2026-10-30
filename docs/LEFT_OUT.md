# What this public tree does **not** contain

Sanitized extract of a private production RAG. The private git history, `.git`,
and secrets were never copied. This file is the explicit leftover checklist.

## Intentionally omitted

- Personal hostnames, laptops, and DNS (home lab, tunnels, managed PaaS hosts).
- Personal / client collection names and the eval goldens built on paid-course
  transcripts.
- Credential stores, secret-reference URLs, deploy sudoers, and CI jobs that
  SSH into a private box.
- Cloudflare Access service-token wiring and token-in-URL MCP auth as used in
  production.
- Member-area scrapers, LMS loops, YouTube/Vimeo transcript ingest, paid-course
  reindex scripts.
- Late-chunking BGE-M3 encoder, Modal GPU embedding/rerank apps, Claude Code
  CLI as the generator (`claude -p`).
- Incremental ingest ledger bound to a private lesson catalogue.
- Raw eval reports (JSONL of private questions, judge caches, latency dumps).
- Brand assets (logo/favicon) and the production GitHub container image name.
- HyDE, query decomposition against a law-school prompt, and the full
  40-file eval matrix (`run_matrix*.sh`, `run_k_sweep`, `run_embed_ab`, …).
- The private repo name and internal spec paths.

## What was kept (rewritten)

Hybrid retrieve (multilingual-e5-large + BM25 on Qdrant) + Jina reranker v2,
split pools / synthetic penalty, phrase-level proof, number guard, MCP server,
eval golden/traps/judge/gate/funnel, Dockerfile + compose + CI, a **public**
PT-BR legislative corpus, LangGraph retrieve→rerank→answer→check, optional
Langfuse/LangSmith tracing.

Production funnel numbers in the README are **anonymized** (corpus A / B) and
come from the private eval; they are not measurements of `corpus/texts/`.
