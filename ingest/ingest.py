"""Reproducible ingest of the vendored public PT-BR corpus into Qdrant.

    python -m ingest.ingest              # upsert corpus/texts/*.md
    python -m ingest.ingest --reset      # drop collection first

Never fetches member areas, paid courses, or private transcripts.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ingest.chunk import ementa_of, load_corpus_file
from ingest.secret_scan import assert_clean
from rag.config import COLLECTION_NAME

CORPUS = Path(__file__).resolve().parent.parent / "corpus" / "texts"


def load_all(root: Path = CORPUS) -> list[dict]:
    docs = []
    for path in sorted(root.glob("*.md")):
        assert_clean(path.read_text(encoding="utf-8"), path.name)
        parts = load_corpus_file(path)
        em = ementa_of(parts)
        docs.extend(parts)
        if em:
            docs.append(em)
    return docs


def upsert(docs: list[dict], collection: str = COLLECTION_NAME) -> int:
    from langchain_core.documents import Document
    from rag.vectorstore import ensure_collection, get_vectorstore

    ensure_collection(collection)
    vs = get_vectorstore(create=False, collection=collection)
    lc = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in docs]
    ids = [
        f"{d['metadata']['doc_id']}::{d['metadata']['chunk']}::{d['metadata']['source_type']}"
        for d in docs
    ]
    vs.add_documents(lc, ids=ids)
    return len(docs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reset", action="store_true", help="drop the collection before upsert")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    docs = load_all()
    print(f"{len(docs)} chunks from {CORPUS}")
    if args.dry_run:
        for d in docs:
            m = d["metadata"]
            print(f"  {m['doc_id']:20} {m['source_type']:8} {m['article']:12} {len(d['page_content']):5}c")
        return 0
    if args.reset:
        from rag.vectorstore import get_client  # noqa: WPS433 — optional heavy import
        c = get_client()
        if c.collection_exists(COLLECTION_NAME):
            c.delete_collection(COLLECTION_NAME)
            print(f"dropped {COLLECTION_NAME}")
    n = upsert(docs)
    print(f"upserted {n} points into {COLLECTION_NAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
