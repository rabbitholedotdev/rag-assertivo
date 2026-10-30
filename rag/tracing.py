"""Optional tracing. Langfuse takes precedence; LangSmith is the fallback.

Neither backend is required. When no keys are set this module is a no-op, so
tests and `docker compose up` stay offline.

Environment
-----------
Langfuse:
  LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY
  LANGFUSE_HOST (default https://cloud.langfuse.com)

LangSmith:
  LANGCHAIN_TRACING_V2=true
  LANGCHAIN_API_KEY
  LANGCHAIN_PROJECT=rag-assertivo
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator


def _langfuse():
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return None
    try:
        from langfuse import get_client
        return get_client()
    except Exception:  # noqa: BLE001
        return None


def tracing_enabled() -> bool:
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        return True
    return os.getenv("LANGCHAIN_TRACING_V2", "").lower() in ("1", "true", "yes")


@contextmanager
def span(name: str, **attrs: Any) -> Iterator[dict]:
    """Record one graph node. Yields a dict the caller may fill with outputs."""
    bag: dict = {"name": name, **attrs}
    lf = _langfuse()
    if lf is not None:
        try:
            with lf.start_as_current_span(name=name, metadata=attrs) as s:
                yield bag
                if bag.get("output") is not None:
                    s.update(output=str(bag["output"])[:2000])
                return
        except Exception:  # noqa: BLE001 — tracing must never break the answer
            pass
    # LangSmith: langchain/langgraph pick up LANGCHAIN_TRACING_V2 on their own.
    # This contextmanager still exists so node code is identical either way.
    yield bag


def flush() -> None:
    lf = _langfuse()
    if lf is not None:
        try:
            lf.flush()
        except Exception:  # noqa: BLE001
            pass
