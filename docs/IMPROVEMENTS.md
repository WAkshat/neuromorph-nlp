# Improvements

Every meaningful change relative to `d80de35`, with the reason and the evidence. Defect
numbers (D1...) refer to `ARCHITECTURE_AUDIT.md`; measurements are in `BENCHMARKS.md`.

## Summary

| Area | Before | After |
|---|---|---|
| Detection F1, held-out | 0.400 | 0.971 (see caveats in BENCHMARKS section 3) |
| False alarms on real prose | 6.8% of sentences | 1.8% of sentences |
| Document coverage | first 12 sentences | whole document, truncation reported |
| Findings | no evidence | each factual finding cites a fact and its source |
| Tests | 0 | 90 |
| Python (backend) | 859 lines | about 1300 lines, plus tests and benchmarks |

## Priority 1: correctness and reliability

### 1. Polarity-aware checking (fixes D1, D7)
- **Files**: `backend/textutils.py` (new), `backend/agents/verifiers.py` (new).
- **Change**: sentences are split into clauses at contrast markers ("whereas", "unlike",
  "but", ";"). Each clause is tested for negation cues before, or just after, the matched
  property; sentences framed as denial ("it is a myth that", "misleading") flip polarity.
  Absolute terms are exempt when a negation cue immediately precedes them ("not always",
  "does not guarantee", "not every").
- **Why**: the original flagged statements that *agree* with its own knowledge base.
- **Evidence**: correct negations flagged 7/10 to 0/10; switching the mechanism off
  restores 10 false positives (ablation).
- **Limit**: no syntactic parse. Scope is a clause plus a token window, so long-range or
  double negation ("it is not untrue that...") is handled wrongly.

### 2. No silent truncation (fixes D4, D5)
- **Files**: `backend/agents/viewer.py`, `recall.py`, `orchestrator.py`.
- **Change**: every sentence of 3+ words is a candidate; the 12-claim and 8-claim caps are
  gone. A configurable `max_claims` (default 5000) remains as a safety bound, and when it
  bites the trace says so in capitals and sets `truncated: true`.
- **Evidence**: recall on gold errors inside a 1000-sentence document 0.114 to 0.957.
- **Cost**: long documents take proportionally longer (126 ms per 1000 sentences).

### 3. Sentence segmentation (fixes D6)
- **File**: `backend/textutils.py`.
- **Change**: does not break after "et al.", "e.g.", "i.e.", "approx.", "Fig.", initials;
  blank lines end a sentence; single newlines are soft wraps (PDF output).
- **Limit**: an abbreviation list, not a trained segmenter.

### 4. API layer (fixes D9, D10, D11, D12, D13, D14, D16, D17)
- **Files**: `backend/main.py`, `backend/services/document_processor.py`.
- **Changes**:
  - `/api/review-text` takes a validated pydantic model; wrong types give 422, not 500.
  - Too-short text is HTTP 400 with a message, from one threshold (20 characters) defined
    once; no more 200-with-error-body.
  - Uploads are read into memory (10 MB cap, 413 beyond it) and never written to disk.
  - Extraction failures raise `ExtractionError` and become 422; they are no longer
    returned as document text.
  - `.doc` removed from the accepted types (python-docx cannot read it).
  - Endpoints are plain `def`, so FastAPI runs them in its threadpool.
  - CORS no longer combines `*` with credentials; `start.sh` and `python main.py` bind
    `127.0.0.1` instead of `0.0.0.0` (the server has no authentication).
  - One structured log line per review.
- **Removed**: `extract_text_from_bytes` and the uploads directory (both replaced).

### 5. XSS in the built-in UI (fixes D15)
- **File**: `backend/static/index.html`.
- **Change**: every value that originates from the document or server passes through an
  `esc()` helper before `innerHTML`. The React frontend was already safe (JSX escapes).
- **Verified** by running the page script in Node with a stub DOM against an `<img onerror>` payload in every field (output is escaped). Not exercised in a real browser, and there is no automated UI test.

### 6. Honest numbers (fixes D18, D20)
- **Files**: `backend/agents/base.py`, both UIs.
- **Change**: hard-coded confidences removed (`confidence` is `null` unless a verifier
  measured something; only NLI findings carry one). The report distinguishes
  `claims_flagged` from `issues_found`.

### 7. Knowledge base corrections (fixes D19)
- Loihi publication venue corrected; TrueNorth and Loihi figures stated with sources.
- A constraint I added and then found wrong was removed: it treated all memristors as
  non-volatile, and volatile memristors exist. Found by the real-prose check.

## Priority 2: core algorithm

### 8. Structured, cited knowledge base
- **File**: `backend/knowledge/neuromorphic_kb.py`.
- **Change**: 29 `Fact` records with citations; 8 `Myth` records stated as the false
  assertion plus its correction; 21 attribute constraints, 6 numeric constraints and 5
  attribution constraints, each pointing at the fact it follows from.
- **Why**: so that knowledge is data, findings can cite sources, and the Checker has
  something to check against (D2).
- **Invariant test**: no sentence of any fact may trigger a constraint
  (`test_knowledge_base_does_not_contradict_itself`).
- **Limit**: 29 facts is a small, hand-curated KB. Coverage is the system's ceiling.

### 9. Constraint verifier
- **File**: `backend/agents/verifiers.py`.
- **Change**: three checks, all data-driven:
  - *attribute polarity* ("TrueNorth ... on-chip learning" asserted positively);
  - *numeric* (neuron, synapse and core counts within a factor of 2 of documented values,
    only when the clause names exactly one chip);
  - *attribution* ("IBM's Loihi", "developed by", "X introduced the Y").
- **Evidence**: factual errors caught 2/14 to 14/14 on the held-out split.
- **Limit**: 2 of 13 false alarms on held-out Wikipedia prose came from attribute
  constraints and 1 from the numeric check (it credited the human brain's synapse count to
  TrueNorth because both were in one clause).

### 10. Overclaim verifier
- **Change**: a lexicon of absolute terms, with universal quantifiers restricted to scope
  nouns ("every workload", not "every timestep") and totality adverbs restricted to
  outcome words ("completely eliminates", not "fully spin-polarized").
- **Evidence**: 12/12 overclaims on the held-out split.
- **Limit, and it is the main one**: it cannot tell an overclaim from a true universal
  statement. 10 of the 13 false alarms on held-out Wikipedia prose are this verifier
  firing on sentences like "a correlation matrix is always positive-definite".

### 11. BM25 retrieval (fixes D8)
- **Change**: Okapi BM25 over stopword-filtered tokens replaces substring counting.
- **Evidence**: recall@3 of the cited fact 0.149 to 0.894.

### 12. Optional NLI verifier
- **Change**: `NLIVerifier`, enabled with `PipelineConfig(nli=True)` or `REVIEWER_NLI=1`.
- **Evidence**: it did **not** improve results on this benchmark and its contradiction arm
  produced confident false positives. It is therefore off by default and the contradiction
  arm needs an explicit threshold. It is kept because the experiment is reproducible and
  the obvious fix (sentence-level premises, a stronger model) is cheap to try.

## Priority 3: architecture

### 13. Every stage is load-bearing (fixes D2, D3)
- The Thinker produces `ClaimAnalysis` objects consumed by Recall and both rule verifiers.
- Recall's per-claim evidence is attached to findings and is the NLI premise set.
- The Synthesizer builds recommendations from the issue types found, not a fixed list.

### 14. Configuration and independent verifiers
- `PipelineConfig` (frozen dataclass): verifiers, negation handling, top-k, claim cap, NLI.
- Verifiers are plain functions over a `ClaimAnalysis`, so each can be switched off; this
  is what makes the ablation table possible.

### 15. Naming
- The classes are still called agents and the UI still shows five of them, because the UI
  and API are built around that. The documentation calls the design what it is: a staged
  pipeline over a shared blackboard.

## Priority 4: performance

Profiled, not optimised. Per-sentence cost is 0.21 ms (original 0.16 to 0.22 ms); 57% in
the Checker, 21% in Recall, 16% in the Thinker, mostly regex matching. Whole-document
time went up because the whole document is now read. See BENCHMARKS section 8.

## Priority 5: developer and research experience

- **Tests** (`tests/`, 90 cases): text utilities, each verifier (with sentences that are
  not in the benchmark), KB invariants, pipeline behaviour and edge cases, HTTP API.
- **Benchmarks** (`benchmarks/`): labelled dev/test splits, detection harness with
  bootstrap intervals and ablation flags, retrieval ablation, scaling benchmark, NLI sweep,
  real-prose check, and one script that regenerates everything.
- **Docs**: this file, the audit, the proposal, the benchmark report, future research.

## Known limitations of the result

1. The benchmark and the system share an author. See BENCHMARKS section 3.
2. About 1.8% of ordinary sentences are flagged, and on the prose tested none of those
   flags was a real problem.
3. Paraphrased errors with no matching rule are missed (2 of 9 on the held-out split).
4. The knowledge base has 29 facts. Anything outside it cannot be fact-checked, and the
   report says so rather than implying the document is correct.
5. English only; one sentence at a time; no coreference ("it", "the chip").
6. Nothing here is neuromorphic and nothing here is multi-agent in the research sense.
