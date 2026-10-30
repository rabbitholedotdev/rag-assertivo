"""Bootstrap CI over run_eval / run_gate reports — is a delta real or noise?

    python -m eval.bootstrap_ci report_a.json report_b.json --metric primary_recall
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st

N_BOOT = 5000


def load_per(path: str) -> list[dict]:
    data = json.loads(open(path, encoding="utf-8").read())
    return data.get("per") or data.get("items") or []


def metric_value(x: dict, m: str) -> float:
    if m == "faithfulness":
        v = x.get("faithfulness")
        return v["score"] if isinstance(v, dict) else float(v or 0)
    if m == "numbers_hallucination_rate":
        v = x.get("numbers") or x.get("numbers_hallucination") or {}
        return v.get("rate", 0.0) if isinstance(v, dict) else float(v or 0)
    return float(x.get(m) or 0)


def boot_mean_ci(xs: list[float], n_boot: int = N_BOOT, alpha: float = 0.05) -> tuple[float, float, float]:
    n = len(xs)
    means = []
    for _ in range(n_boot):
        sample = [xs[random.randrange(n)] for _ in range(n)]
        means.append(st.mean(sample))
    means.sort()
    lo = means[int(n_boot * alpha / 2)]
    hi = means[int(n_boot * (1 - alpha / 2))]
    return round(st.mean(xs), 3), round(lo, 3), round(hi, 3)


def boot_diff_ci(a: list[float], b: list[float], n_boot: int = N_BOOT, alpha: float = 0.05) -> tuple[float, float, float]:
    n = len(a)
    diffs_obs = [x - y for x, y in zip(a, b)]
    boots = []
    for _ in range(n_boot):
        idx = [random.randrange(n) for _ in range(n)]
        boots.append(st.mean([diffs_obs[i] for i in idx]))
    boots.sort()
    lo = boots[int(n_boot * alpha / 2)]
    hi = boots[int(n_boot * (1 - alpha / 2))]
    return round(st.mean(diffs_obs), 3), round(lo, 3), round(hi, 3)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("report_a")
    ap.add_argument("report_b")
    ap.add_argument("--metric", nargs="+", default=["primary_recall", "lesson_recall", "content_recall"])
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args(argv)

    per_a, per_b = load_per(args.report_a), load_per(args.report_b)
    ids_a, ids_b = [x.get("id") or x.get("pergunta") for x in per_a], [x.get("id") or x.get("pergunta") for x in per_b]
    if ids_a != ids_b:
        raise SystemExit(f"reports not pairable ({len(ids_a)} vs {len(ids_b)} ids)")

    print(f"n_perguntas={len(ids_a)}\n")
    for m in args.metric:
        xs_a = [metric_value(x, m) for x in per_a]
        xs_b = [metric_value(x, m) for x in per_b]
        mean_a, lo_a, hi_a = boot_mean_ci(xs_a, n_boot=args.n_boot)
        mean_b, lo_b, hi_b = boot_mean_ci(xs_b, n_boot=args.n_boot)
        d, dlo, dhi = boot_diff_ci(xs_a, xs_b, n_boot=args.n_boot)
        sig = "REAL (CI misses 0)" if (dlo > 0 or dhi < 0) else "noise (CI crosses 0)"
        print(f"{m}:")
        print(f"  {args.label_a}: {mean_a}  IC95 [{lo_a}, {hi_a}]")
        print(f"  {args.label_b}: {mean_b}  IC95 [{lo_b}, {hi_b}]")
        print(f"  diff ({args.label_a}-{args.label_b}): {d}  IC95 [{dlo}, {dhi}]  -> {sig}\n")


if __name__ == "__main__":
    main()
