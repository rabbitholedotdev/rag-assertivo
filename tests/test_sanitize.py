"""Refuse to ship personal hosts, paid-course leftovers, or credential-shaped strings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "models", "data", ".pytest_cache"}
SKIP_SUFFIX = {".png", ".ico", ".pyc"}

# Personal / private-production leftovers that must never land in this public tree.
FORBIDDEN = [
    "1password",
    "op://",
    "coolify",
    "joaopedrocoelho",
    "joaopedro",
    "macbook-pro-de-joao",
    "nutror",
    "curseduca",
    "pedro_rag",
    "ofm_rag",
    "examos_rag",
    "natanael",
    "xmserver",
    "attivo-novo",
    "attivorag",
    "fanxpass",
    "ragxandy",
    "claude_code_oauth",
    "unilucro",
    "app.nutror.com",
]


def _files():
    for p in ROOT.rglob("*"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.suffix in SKIP_SUFFIX or not p.is_file():
            continue
        if p.name == "test_sanitize.py" or p.name == "LEFT_OUT.md":
            continue
        yield p


def test_no_private_production_leftovers():
    hits = []
    for path in _files():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
        except OSError:
            continue
        for needle in FORBIDDEN:
            if needle in text:
                hits.append(f"{path.relative_to(ROOT)}: {needle}")
    assert hits == [], "sanitization leak:\n" + "\n".join(hits)


def test_corpus_is_planalto():
    import json
    src = json.loads((ROOT / "corpus" / "sources.json").read_text())
    for d in src["documents"]:
        assert "planalto.gov.br" in d["url"]
