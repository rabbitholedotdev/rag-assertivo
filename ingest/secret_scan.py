"""Refuse to ingest a file that looks like it contains credentials.

Ported from the private pipeline (the production ingest must never upsert an
API key that leaked into a source document). Patterns are generic: no product
names, no personal hosts.
"""
from __future__ import annotations

import re

_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    re.compile(r"eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),  # JWT
    re.compile(r"(?i)api[_-]?key\s*[:=]\s*['\"][^'\"]{12,}"),
    re.compile(r"(?i)password\s*[:=]\s*['\"][^'\"]{8,}"),
]


def findings(text: str) -> list[str]:
    hits = []
    for p in _PATTERNS:
        if p.search(text):
            hits.append(p.pattern)
    return hits


def assert_clean(text: str, name: str = "document") -> None:
    hits = findings(text)
    if hits:
        raise ValueError(f"{name}: refused ingest, looks like a secret ({len(hits)} pattern(s))")
