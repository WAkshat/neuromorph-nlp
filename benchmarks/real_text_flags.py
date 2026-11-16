"""Out-of-distribution check on prose nobody on this project wrote.

Fetches the plain text of Wikipedia articles (pinned by revision id when given),
submits every sentence of at least 50 characters as its own document, and lists the
sentences the system flags. There are no gold labels: the output is meant to be read
by a human, and docs/BENCHMARKS.md records one such reading.

    python benchmarks/real_text_flags.py --system current --set heldout
    python benchmarks/real_text_flags.py --system baseline --set tuning
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from common import ROOT, load_orchestrator

# title -> revision id the recorded results were produced from (None = latest).
# "tuning" articles were read during development and motivated one lexicon change;
# "heldout" articles were run once, after that change, and never used for tuning.
ARTICLE_SETS = {
    "tuning": {
        "Neuromorphic computing": 1381885409,
        "Spiking neural network": 1384675387,
        "Memristor": 1385826726,
    },
    "heldout": {
        "Event camera": 1362464116,
        "Hebbian theory": 1382839520,
        "Biological neuron model": 1378207583,
        "Neural coding": 1374681090,
        "Cognitive computer": 1384319408,
    },
}


def fetch(title: str, revid: int | None) -> tuple[int, str]:
    """Fetch (and cache under benchmarks/.cache/) the plain-text extract of one article."""
    cache = Path(__file__).parent / ".cache" / f"{title.replace(' ', '_')}_{revid or 'latest'}.json"
    if cache.exists():
        return tuple(json.loads(cache.read_text(encoding="utf-8")))
    params = {"action": "query", "format": "json", "prop": "extracts|revisions", "explaintext": 1,
              "rvprop": "ids", "redirects": 1}
    if revid:
        params["revids"] = revid
    else:
        params["titles"] = title
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "claims-reviewer-benchmark/1.0"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                page = next(iter(json.load(resp)["query"]["pages"].values()))
            break
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == 4:
                raise
            time.sleep(10 * (attempt + 1))      # rate limited: back off and retry
    result = (page["revisions"][0]["revid"], page["extract"])
    cache.parent.mkdir(exist_ok=True)
    cache.write_text(json.dumps(result), encoding="utf-8")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["baseline", "current"], required=True)
    ap.add_argument("--set", choices=list(ARTICLE_SETS), required=True)
    ap.add_argument("--out")
    args = ap.parse_args()
    ARTICLES = ARTICLE_SETS[args.set]

    orch = load_orchestrator(args.system)
    sys.path.insert(0, str(ROOT / "backend"))   # for the baseline run: only textutils is used from here
    from textutils import split_sentences

    report = []
    total = flagged_total = 0
    for title, revid in ARTICLES.items():
        revid, text = fetch(title, revid)
        sentences = [s for s in split_sentences(text) if len(s) >= 50 and not s.startswith("==")]
        flagged = []
        for s in sentences:
            issues = orch.process_document(s).get("potential_issues") or []
            if issues:
                flagged.append({"sentence": s, "types": sorted({i["type"] for i in issues}),
                                "explanations": [i["explanation"] for i in issues]})
        total += len(sentences)
        flagged_total += len(flagged)
        report.append({"title": title, "revid": revid, "sentences": len(sentences), "flagged": flagged})
        print(f"{title} (rev {revid}): {len(flagged)}/{len(sentences)} sentences flagged")
    print(f"{args.system}: {flagged_total}/{total} sentences flagged ({100 * flagged_total / total:.1f}%)")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
