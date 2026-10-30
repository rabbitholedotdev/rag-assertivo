"""CLI: python -m rag search|answer|health"""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rag")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=6)
    a = sub.add_parser("answer")
    a.add_argument("question")
    a.add_argument("-k", type=int, default=8)
    sub.add_parser("health")
    sub.add_parser("sources")
    args = ap.parse_args(argv)

    if args.cmd == "search":
        from rag.mcp_server import search
        print(json.dumps(search(args.query, k=args.k), ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "answer":
        from rag.mcp_server import answer
        print(json.dumps(answer(args.question, k=args.k), ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "health":
        from rag.mcp_server import health
        print(json.dumps(health(), ensure_ascii=False, indent=2))
        return 0
    from rag.mcp_server import list_sources
    print(json.dumps(list_sources(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
