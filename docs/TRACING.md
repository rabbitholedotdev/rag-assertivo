# Tracing

The graph in `rag/graph.py` is retrieve → rerank → answer (generate + code
proof + optional LLM verify). Each node is wrapped with `rag.tracing.span`.

Nothing is sent anywhere unless you export keys. Tests stay offline.

## Langfuse

```bash
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_HOST=https://cloud.langfuse.com   # or your host
pip install langfuse
```

`rag.tracing.span` opens a Langfuse span per node (`retrieve`, `rerank`,
`answer_and_check`). Call `rag.tracing.flush()` at process exit if you run a
batch eval.

Self-host: set `LANGFUSE_HOST` to that origin. Do not put a personal hostname
in the repo; it belongs in your `.env`.

## LangSmith

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=lsv2_...
export LANGCHAIN_PROJECT=rag-assertivo
```

LangGraph's compiled app (`build_graph()`) emits runs to LangSmith when those
variables are set. The node functions themselves do not import `langsmith`.

## What you should see

One trace per `answer` / `run_pipeline` call, three spans:

1. `retrieve` — hybrid candidates (dense + BM25, optional split pools)
2. `rerank` — Jina cross-encoder (or the injected `reranker_fn`)
3. `answer_and_check` — JSON generation, `rag.proof.check_phrase`, optional verifier

Fail-closed: a tracing exception is swallowed. An answer never depends on the
telemetry backend being up.
