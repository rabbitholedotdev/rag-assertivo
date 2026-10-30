"""Best-effort refresh of official HTML from planalto.gov.br.

The vendored files under corpus/texts/ remain the source of truth. This script
does not overwrite them unless --write is passed, and it never touches anything
behind a login.

    python -m ingest.fetch_planalto --dry-run
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT / "corpus" / "sources.json"


def _strip_html(html: str) -> str:
    html = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    html = re.sub(r"(?is)<style.*?>.*?</style>", " ", html)
    html = re.sub(r"(?is)<[^>]+>", " ", html)
    html = re.sub(r"&nbsp;", " ", html)
    html = re.sub(r"&amp;", "&", html)
    html = re.sub(r"&quot;", '"', html)
    html = re.sub(r"\s+", " ", html)
    return html.strip()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", default=True)
    ap.add_argument("--write", action="store_true", help="actually GET the URLs (network)")
    args = ap.parse_args(argv)
    docs = json.loads(SOURCES.read_text(encoding="utf-8"))["documents"]
    for d in docs:
        print(f"{d['doc_id']:20} {d['url']}")
        if not args.write:
            continue
        try:
            import httpx
            r = httpx.get(d["url"], timeout=30.0, follow_redirects=True)
            r.raise_for_status()
            text = _strip_html(r.text)
            print(f"  fetched {len(text)} chars (not written; copy into corpus/texts/ by hand)")
        except Exception as e:  # noqa: BLE001
            print(f"  skip: {e}")
    if not args.write:
        print("dry-run only. Pass --write to GET. Vendored corpus/texts/*.md is still the ingest source.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
