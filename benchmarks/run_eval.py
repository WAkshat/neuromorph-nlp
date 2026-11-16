"""Claim-level detection benchmark.

Each benchmark sentence is submitted as its own document. The system's prediction is
"flag" if it raises at least one issue, otherwise "ok". Gold labels are in data/*.tsv.

    python benchmarks/run_eval.py --system baseline --split test
    python benchmarks/run_eval.py --system current  --split dev --errors
    python benchmarks/run_eval.py --system current  --split test --verifiers overclaim
    python benchmarks/run_eval.py --system current  --split test --nli
"""

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

from common import load_orchestrator, load_split


def prf(gold: list[bool], pred: list[bool]) -> dict:
    tp = sum(g and p for g, p in zip(gold, pred))
    fp = sum((not g) and p for g, p in zip(gold, pred))
    fn = sum(g and (not p) for g, p in zip(gold, pred))
    tn = len(gold) - tp - fp - fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision,
            "recall": recall, "f1": f1, "accuracy": (tp + tn) / len(gold)}


def bootstrap_f1_ci(gold, pred, n_boot=2000, seed=0) -> tuple[float, float]:
    """95% percentile bootstrap CI over benchmark items (the systems are deterministic,
    so the only uncertainty is which sentences happen to be in the benchmark)."""
    rng = random.Random(seed)
    n = len(gold)
    scores = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        scores.append(prf([gold[i] for i in idx], [pred[i] for i in idx])["f1"])
    scores.sort()
    return scores[int(0.025 * n_boot)], scores[int(0.975 * n_boot)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", choices=["baseline", "current"], required=True)
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--verifiers", help="current only: comma-separated subset, e.g. 'overclaim,constraints'")
    ap.add_argument("--nli", action="store_true", help="current only: enable the optional NLI verifier")
    ap.add_argument("--no-negation", action="store_true", help="current only: ablate negation/contrast handling")
    ap.add_argument("--errors", action="store_true", help="print misclassified items")
    ap.add_argument("--out", help="write the result JSON here")
    args = ap.parse_args()

    config = {}
    if args.verifiers is not None:
        config["verifiers"] = tuple(v for v in args.verifiers.split(",") if v)
    if args.nli:
        config["nli"] = True
    if args.no_negation:
        config["negation_aware"] = False
    orch = load_orchestrator(args.system, **config)
    items = load_split(args.split)

    orch.process_document(items[0]["text"])  # warm-up (imports, lazy model load)
    gold, pred, rows = [], [], []
    start = time.perf_counter()
    for it in items:
        result = orch.process_document(it["text"])
        issues = result.get("potential_issues") or []
        gold.append(it["label"] == "flag")
        pred.append(bool(issues))
        rows.append({**it, "pred": "flag" if issues else "ok",
                     "issue_types": sorted({i["type"] for i in issues})})
    elapsed = time.perf_counter() - start

    by_cat = defaultdict(lambda: [0, 0])  # category -> [flagged, total]
    for r in rows:
        by_cat[r["category"]][0] += r["pred"] == "flag"
        by_cat[r["category"]][1] += 1

    overall = prf(gold, pred)
    lo, hi = bootstrap_f1_ci(gold, pred)
    summary = {
        "system": args.system, "split": args.split, "config": config, "n": len(items),
        **overall, "f1_ci95": [lo, hi],
        "flag_rate_by_category": {c: {"flagged": f, "total": t} for c, (f, t) in sorted(by_cat.items())},
        "mean_latency_ms": 1000 * elapsed / len(items),
    }

    print(f"{args.system} / {args.split} / config={config or 'default'}  (n={len(items)})")
    print(f"  precision {overall['precision']:.3f}  recall {overall['recall']:.3f}  "
          f"F1 {overall['f1']:.3f} [95% CI {lo:.3f}-{hi:.3f}]  accuracy {overall['accuracy']:.3f}")
    print(f"  TP {overall['tp']}  FP {overall['fp']}  FN {overall['fn']}  TN {overall['tn']}"
          f"   mean latency {summary['mean_latency_ms']:.2f} ms/claim")
    print("  flagged per category (gold 'flag' categories should be high, 'ok' ones low):")
    for c, (f, t) in sorted(by_cat.items()):
        print(f"    {c:18s} {f:2d}/{t}")
    if args.errors:
        for r in rows:
            if r["pred"] != r["label"]:
                print(f"  [{'FP' if r['pred'] == 'flag' else 'FN'}] {r['id']} {r['issue_types']} {r['text']}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"summary": summary, "items": rows}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
