"""Retrieval ablation: BM25 vs the original substring-overlap scorer, on the same KB.

Gold evidence is silver-standard: for every benchmark claim (dev + test) where the
constraint verifier finds a contradiction, the fact that constraint cites is taken
as the fact a retriever should surface. Reports recall@k of that fact, plus how
many facts each scorer returns for sentences that have nothing to do with the KB.

    python benchmarks/retrieval_eval.py
"""

import sys

from common import ROOT, load_split

sys.path.insert(0, str(ROOT / "backend"))
from agents.verifiers import check_constraints  # noqa: E402
from knowledge import neuromorphic_kb as kb  # noqa: E402
from textutils import analyze  # noqa: E402

OFF_TOPIC = [
    "The committee is going to meet on Tuesday to discuss the annual budget.",
    "Rainfall in Paris was unusually heavy during the first week of the month.",
    "She has been learning to play the violin for more than ten years now.",
    "The recipe uses two cups of flour and is baked for forty minutes.",
    "Traffic on the motorway was slow because of an accident near the bridge.",
]


def legacy_search(query: str, top_k: int) -> list[str]:
    """The original scorer (d80de35 search_knowledge), applied to the current facts:
    count query words that occur as substrings of 'key: text', stopwords included."""
    scored = []
    for f in kb.FACTS:
        text = f"{f.id.split('.')[-1]}: {f.text}".lower()
        score = sum(1 for term in query.lower().split() if term in text)
        if score > 0:
            scored.append((0 if score >= 2 else 1, f.id))
    scored.sort(key=lambda x: x[0])     # stable: ties stay in KB order, as in the original
    return [fid for _, fid in scored[:top_k]]


def bm25_search(query: str, top_k: int) -> list[str]:
    return [h["id"] for h in kb.search_knowledge(query, top_k)]


def main():
    gold = []
    for it in load_split("dev") + load_split("test"):
        for issue in check_constraints(analyze(0, it["text"])):
            gold.append((it["text"], issue["evidence"][0]["id"]))
    print(f"{len(gold)} (claim, cited fact) pairs from constraint findings on dev+test")
    for name, search in (("legacy substring overlap", legacy_search), ("BM25", bm25_search)):
        r1 = sum(fid in search(q, 1) for q, fid in gold) / len(gold)
        r3 = sum(fid in search(q, 3) for q, fid in gold) / len(gold)
        noise = sum(len(search(q, 3)) for q in OFF_TOPIC) / len(OFF_TOPIC)
        print(f"  {name:26s} recall@1 {r1:.3f}  recall@3 {r3:.3f}  "
              f"facts returned per off-topic sentence {noise:.1f}")


if __name__ == "__main__":
    main()
