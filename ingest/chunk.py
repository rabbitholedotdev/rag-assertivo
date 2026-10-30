"""Split a law file into article-sized chunks. Overlap is the previous heading."""
from __future__ import annotations

import re
from pathlib import Path

FRONT = re.compile(r"^---\n(.*?)\n---\n", re.S)
ART = re.compile(r"(?=^Art\.?\s+\d+)", re.M | re.I)


def parse_front_matter(text: str) -> tuple[dict, str]:
    m = FRONT.match(text)
    if not m:
        return {}, text
    meta = {}
    for line in m.group(1).splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        meta[k.strip()] = v.strip().strip('"').strip("'")
    return meta, text[m.end():]


def split_articles(body: str) -> list[tuple[str, str]]:
    """Return [(article_label, text), ...]. Preamble (before Art. 1) is kept as 'preambulo'."""
    parts = ART.split(body)
    out = []
    if parts and parts[0].strip():
        head = parts[0].strip()
        if head:
            out.append(("preambulo", head))
        parts = parts[1:]
    for p in parts:
        p = p.strip()
        if not p:
            continue
        m = re.match(r"Art\.?\s+(\d+[A-Za-zº°-]*)", p, re.I)
        label = f"art-{m.group(1)}" if m else "art"
        out.append((label, p))
    return out or [("corpo", body.strip())]


def load_corpus_file(path: Path) -> list[dict]:
    raw = path.read_text(encoding="utf-8")
    meta, body = parse_front_matter(raw)
    docs = []
    for i, (label, text) in enumerate(split_articles(body)):
        docs.append({
            "page_content": text,
            "metadata": {
                **meta,
                "doc_id": meta.get("doc_id") or path.stem,
                "article": label,
                "chunk": i,
                "source_type": meta.get("source_type") or "lei",
                "title": meta.get("title") or path.stem,
                "area": meta.get("area") or "legislacao",
                "url": meta.get("url") or "",
                "path": str(path.name),
            },
        })
    return docs


def ementa_of(docs: list[dict]) -> dict | None:
    """One synthetic navigation card per file: first 400 chars, not citable."""
    if not docs:
        return None
    primary = [d for d in docs if d["metadata"].get("source_type") not in ("ementa", "resumo", "qa")]
    if not primary:
        return None
    base = primary[0]["metadata"]
    blob = " ".join(d["page_content"] for d in primary)[:400]
    return {
        "page_content": blob,
        "metadata": {
            **base,
            "source_type": "ementa",
            "doc_id": f"ementa-{base['doc_id']}",
            "chunk": 0,
            "article": "ementa",
        },
    }
