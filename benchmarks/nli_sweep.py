"""Threshold sweep for the optional NLI verifier, run on its own (no rule verifiers).

For every claim in a split it records the two NLI scores the verifier uses:
  c = max P(contradiction | premise = retrieved fact, hypothesis = claim)
  m = max P(entailment    | premise = claim, hypothesis = known myth)
and prints precision / recall of "flag if c >= tc or m >= tm" over a threshold grid.
Thresholds must be chosen on dev, never on test.

    python benchmarks/nli_sweep.py --split dev
"""

import argparse
import sys
import time

from common import ROOT, load_split

sys.path.insert(0, str(ROOT / "backend"))
from agents.verifiers import NLIVerifier  # noqa: E402
from knowledge import neuromorphic_kb as kb  # noqa: E402
from textutils import analyze  # noqa: E402

GRID = (0.5, 0.8, 0.9, 0.95, 0.99)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--model", default="cross-encoder/nli-MiniLM2-L6-H768")
    args = ap.parse_args()

    v = NLIVerifier(args.model, myth_threshold=0.0, contradiction_threshold=0.0)
    rows = []
    items = load_split(args.split)
    t0 = time.perf_counter()
    for it in items:
        a = analyze(0, it["text"])
        ev = kb.search_knowledge(it["text"], 3)
        p = v._probs([(e["content"], a.text) for e in ev] + [(a.text, m.claim) for m in kb.MYTHS])
        c = float(p[:len(ev), v.contradiction].max()) if ev else 0.0
        m = float(p[len(ev):, v.entailment].max())
        rows.append((it, a.denial, c, m))
    print(f"{args.model} on {args.split}: {1000 * (time.perf_counter() - t0) / len(items):.1f} ms/claim")

    def report(name, pred):
        gold = [it["label"] == "flag" for it, *_ in rows]
        tp = sum(g and p for g, p in zip(gold, pred))
        fp = sum((not g) and p for g, p in zip(gold, pred))
        fn = sum(g and (not p) for g, p in zip(gold, pred))
        print(f"  {name:28s} TP={tp:2d} FP={fp:2d} FN={fn:2d}  P={tp / max(tp + fp, 1):.2f} R={tp / (tp + fn):.2f}")

    print("contradiction arm only (flag if c >= tc):")
    for tc in GRID:
        report(f"tc={tc}", [(not d) and c >= tc for _, d, c, _ in rows])
    print("myth-entailment arm only (flag if m >= tm):")
    for tm in GRID:
        report(f"tm={tm}", [(not d) and m >= tm for _, d, _, m in rows])
    print("correct ('ok') claims with c >= 0.9:")
    for it, d, c, m in rows:
        if it["label"] == "ok" and not d and c >= 0.9:
            print(f"  c={c:.3f} {it['id']} {it['text']}")


if __name__ == "__main__":
    main()
