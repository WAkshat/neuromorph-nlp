"""Document-level scaling benchmark: latency, peak Python memory, and coverage.

Builds synthetic documents of N sentences by cycling through the dev+test benchmark
sentences (so gold labels are known for every sentence), then measures:
  * wall time (median of --repeats runs)
  * peak traced Python allocations (tracemalloc; not RSS, not energy)
  * coverage: how many of the N sentences the system actually examined
  * recall on the gold-"flag" sentences inside the document

    python benchmarks/perf.py --system baseline
    python benchmarks/perf.py --system current --out benchmarks/results/perf_current.json
"""

import argparse
import json
import statistics
import time
import tracemalloc
from pathlib import Path

from common import load_orchestrator, load_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["baseline", "current"], required=True)
    ap.add_argument("--sizes", default="10,100,1000,5000")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out")
    args = ap.parse_args()

    orch = load_orchestrator(args.system)
    pool = load_split("dev") + load_split("test")
    rows = []
    print(f"{args.system}: sentences | median ms | peak MiB | examined | flagged-gold recall")
    for n in (int(s) for s in args.sizes.split(",")):
        sents = [pool[i % len(pool)] for i in range(n)]
        doc = " ".join(s["text"] for s in sents)
        gold_flag = {s["text"] for s in sents if s["label"] == "flag"}

        orch.process_document(doc)  # warm-up
        times = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            result = orch.process_document(doc)
            times.append(time.perf_counter() - t0)
        tracemalloc.start()
        orch.process_document(doc)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        # Distinct sentences, because the pool repeats once n exceeds its size.
        examined = len(set(result["extracted_claims"]))
        flagged = {i["claim"] for i in result["potential_issues"]}
        row = {
            "sentences": n, "distinct_sentences": len({s["text"] for s in sents}),
            "median_ms": 1000 * statistics.median(times),
            "peak_mib": peak / 2**20,
            "distinct_examined": examined,
            "gold_flag_recall": len(flagged & gold_flag) / len(gold_flag) if gold_flag else None,
        }
        rows.append(row)
        print(f"  {n:6d} | {row['median_ms']:9.2f} | {row['peak_mib']:8.3f} | "
              f"{examined}/{row['distinct_sentences']} | {row['gold_flag_recall']:.3f}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"system": args.system, "rows": rows}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
