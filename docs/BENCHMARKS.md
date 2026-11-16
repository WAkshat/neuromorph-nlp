# Benchmarks

Every number here was produced by `bash benchmarks/reproduce.sh --nli` on the final code and
is stored under `benchmarks/results/`. Nothing is estimated. Things that were not measured
are listed in section 9.

## 1. Headline

| | Original (`d80de35`) | Revised |
|---|---|---|
| Held-out test F1 (70 labelled claims) | 0.400 | **0.971** |
| Precision / recall | 0.480 / 0.343 | 1.000 / 0.943 |
| Correct denials of a myth wrongly flagged | 7 of 10 | 0 of 10 |
| Factual errors caught | 2 of 14 | 14 of 14 |
| Sentences examined in a 1000-sentence document | 12 | all |
| Sentences flagged in real Wikipedia prose (held-out articles) | 50 of 736 (6.8%) | 13 of 736 (1.8%) |

**Read section 3 before quoting the F1.** The benchmark sentences were written by the same
author as the system. The real-prose row is the more trustworthy indicator, and it shows
that the revised system still raises false alarms on roughly 1 sentence in 55.

## 2. Environment

| | |
|---|---|
| CPU | AMD Ryzen 7 9700X (8 cores / 16 threads), single-threaded workload |
| RAM | 32 GB |
| OS | Windows 11 Pro 10.0.26200 |
| Python | 3.12.10 |
| Libraries | fastapi 0.141.1, pypdf 4.3.1, python-docx 1.2.0; for NLI only: torch 2.14.0+cpu, transformers 4.57.6 |
| GPU | none used |
| Baseline | imported from `git archive d80de35 backend` by `benchmarks/common.py`, so it is always the untouched original |

Both systems are deterministic: the same input gives the same flags. There are therefore no
random seeds to vary and no run-to-run variance in the detection metrics. The uncertainty
that does exist is *which sentences are in the benchmark*, reported as a 95% bootstrap
interval over items (2000 resamples, seed 0). Latencies vary by a few percent between runs.

## 3. The claim benchmark and its limits

`benchmarks/data/dev.tsv` and `test.tsv`: 70 sentences each, 35 that should be flagged and
35 that should not.

| Category | Gold | n per split | What it tests |
|---|---|---|---|
| overclaim | flag | 12 | absolute / universal wording |
| factual_error | flag | 14 | wrong property, number or attribution |
| paraphrased_myth | flag | 9 | a known myth with none of the trigger words |
| correct_fact | ok | 12 | true statements about the same entities |
| correct_negation | ok | 10 | true statements that *deny* a myth, using its vocabulary |
| hedged | ok | 8 | properly qualified claims |
| neutral | ok | 5 | prose that makes no checkable claim |

Protocol: each sentence is submitted as its own document; prediction is "flag" if any issue
is raised. Both splits were committed (`2e2b0bb`) before any backend change. Development
used `dev` only. The implementation was committed (`ddde84f`) before `test` was first run.

Limits, stated plainly:

1. **Same author for data and system.** I wrote the test sentences and, later the same day,
   the rules. Committing the split first prevents tuning on test results; it cannot prevent
   the rules from reflecting the kinds of errors I thought to write. The test F1 is an upper
   bound on what to expect from text written by someone else.
2. **Small.** 70 items per split; the intervals are wide (see below).
3. **Single-sentence claims** with one issue each. Real documents spread a claim over
   several sentences and use pronouns.
4. **One post-hoc revision.** After the first test run, the real-prose check (section 6)
   showed the overclaim lexicon was too broad, and one KB constraint was wrong (it asserted
   all memristors are non-volatile). Both were changed. The default-configuration test
   result was identical before and after (F1 0.971, same two misses); the overclaim-only
   ablation lost one true positive (recall 0.457 to 0.429). The numbers below are after the
   revision.

## 4. Detection results

### Original system

| Split | Precision | Recall | F1 [95% CI] | Accuracy | TP / FP / FN / TN |
|---|---|---|---|---|---|
| dev | 0.480 | 0.343 | 0.400 [0.237, 0.551] | 0.486 | 12 / 13 / 23 / 22 |
| test | 0.480 | 0.343 | 0.400 [0.235, 0.545] | 0.486 | 12 / 13 / 23 / 22 |

(The two splits give the same totals by coincidence; the per-category counts differ.)

### Revised system, default configuration (rule verifiers only)

| Split | Precision | Recall | F1 [95% CI] | Accuracy | TP / FP / FN / TN |
|---|---|---|---|---|---|
| dev (tuned on) | 1.000 | 0.971 | 0.986 [0.949, 1.000] | 0.986 | 34 / 0 / 1 / 35 |
| test (held out) | 1.000 | 0.943 | 0.971 [0.921, 1.000] | 0.971 | 33 / 0 / 2 / 35 |

### Per category, held-out test (sentences flagged / total)

| Category | Should be | Original | Revised |
|---|---|---|---|
| overclaim | high | 8/12 | 12/12 |
| factual_error | high | 2/14 | 14/14 |
| paraphrased_myth | high | 2/9 | 7/9 |
| correct_fact | low | 1/12 | 0/12 |
| correct_negation | low | **7/10** | 0/10 |
| hedged | low | 4/8 | 0/8 |
| neutral | low | 1/5 | 0/5 |

The two test misses are both paraphrases with no rule and no trigger word:

- "Online learning on the device is a standard feature across the whole range of neuromorphic processors."
- "Training spiking networks directly is cheap and needs far less computation than training an equivalent ANN."

## 5. Ablations (held-out test)

| Configuration | Precision | Recall | F1 [95% CI] | FP | ms/claim |
|---|---|---|---|---|---|
| **Default: overclaim + constraints** | 1.000 | 0.943 | 0.971 [0.921, 1.000] | 0 | 0.21 |
| overclaim verifier only | 1.000 | 0.429 | 0.600 [0.408, 0.750] | 0 | 0.11 |
| constraints verifier only | 1.000 | 0.629 | 0.772 [0.640, 0.880] | 0 | 0.21 |
| default, negation handling off | 0.767 | 0.943 | 0.846 [0.754, 0.923] | **10** | 0.20 |
| default + NLI myth arm | 1.000 | 0.943 | 0.971 [0.921, 1.000] | 0 | 63.8 |
| default + NLI myth + contradiction (T = 0.95) | 0.944 | 0.971 | 0.958 [0.904, 1.000] | 2 | 136.1 |
| NLI myth arm only | 1.000 | 0.114 | 0.205 [0.050, 0.368] | 0 | 63.8 |

What each row shows:

- **The two rule verifiers are complementary.** Neither reaches 0.8 F1 alone.
- **Negation handling is where the precision comes from.** Switching it off brings back
  10 false positives, 9 of them on correct negations: the original's main failure.
- **NLI does not pay for itself here.** The myth arm adds nothing over the rules at 300
  times the latency. The contradiction arm finds one more true positive and adds two false
  positives ("TrueNorth does not support on-chip learning" is judged to *contradict* the
  knowledge base). That is a negative result for this model and this premise format, not
  for NLI in general; see `FUTURE_RESEARCH.md`.

### NLI threshold sweep on dev (NLI alone, no rules)

`benchmarks/results/nli_sweep_dev.txt`; model `cross-encoder/nli-MiniLM2-L6-H768`, 135 ms/claim.

| Contradiction threshold | TP | FP | FN | | Myth-entailment threshold | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|
| 0.50 | 23 | 10 | 12 | | 0.50 | 6 | 0 | 29 |
| 0.80 | 21 | 4 | 14 | | 0.80 | 5 | 0 | 30 |
| 0.90 | 15 | 4 | 20 | | 0.90 | 5 | 0 | 30 |
| 0.95 | 13 | 3 | 22 | | 0.95 | 3 | 0 | 32 |
| 0.99 | 5 | 2 | 30 | | 0.99 | 0 | 0 | 35 |

False positives of the contradiction arm survive even a 0.99 threshold, so no threshold
makes it safe; that is why it is disabled unless requested.

## 6. Real prose: false alarms on text nobody here wrote

`benchmarks/real_text_flags.py` fetches Wikipedia articles (revision ids pinned in the
script), submits every sentence of 50+ characters, and lists what is flagged. There are no
gold labels; I read every sentence the revised system flagged.

| Article set | Sentences | Original flags | Revised flags |
|---|---|---|---|
| Tuning set, first run (before the lexicon revision) | 444 | 15 (3.4%) | 9 (2.0%) |
| Tuning set, after the revision | 444 | 15 (3.4%) | 4 (0.9%) |
| **Held-out set, run once after the revision** | 736 | 50 (6.8%) | **13 (1.8%)** |

Tuning set: *Neuromorphic computing*, *Spiking neural network*, *Memristor*.
Held-out set: *Event camera*, *Hebbian theory*, *Biological neuron model*, *Neural coding*,
*Cognitive computer*.

My reading of the 13 held-out flags: **none is a real overclaim or error.**

| Kind | Count | Example |
|---|---|---|
| Mathematical or physical universals | 7 | "a correlation matrix is always a positive-definite matrix" |
| Ordinary use of an absolute word | 3 | "people who have never played the piano"; "a perfect insulator" |
| Attribute constraint misfire | 2 | "temporal codes ... while maintaining the same mean firing rate" read as "temporal coding is the mean rate" |
| Numeric constraint misfire | 1 | "the human brain's 800 trillion synapses, and behind IBM's TrueNorth": the number belongs to the brain, not the chip |

What this establishes and what it does not:

- **Established**: the false-alarm rate on encyclopedic prose is about 1.8% of sentences,
  down from 6.8% for the original (a 3.8-fold reduction), and the benchmark's precision of
  1.00 does not carry over to text written by other people.
- **Not established**: recall on real text. These articles contain few or no overclaims,
  so they cannot show whether real overclaims are caught. I did not read the original's 50
  flags, so its count is a rate, not a judged precision.

The held-out findings were deliberately *not* used to change the system; fixing them and
re-measuring on the same articles would turn the held-out set into a tuning set.

## 7. Retrieval

`benchmarks/retrieval_eval.py`. Silver labels: for the 47 benchmark claims (dev + test) on
which a constraint fires, the fact that constraint cites should be retrieved. Both scorers
run over the same (revised) knowledge base, so this isolates the ranking function.

| Scorer | Recall@1 | Recall@3 | Facts returned per off-topic sentence (of 3) |
|---|---|---|---|
| Original substring overlap | 0.021 | 0.149 | 3.0 |
| BM25 | 0.617 | 0.894 | 1.2 |

BM25 still returns about one fact for sentences unrelated to the field (shared common
words such as "learning" or "time"). There is no score threshold yet.

## 8. Latency, memory, coverage

`benchmarks/perf.py`: synthetic documents made by cycling the 140 benchmark sentences.
Median of 5 runs; memory is peak traced Python allocation (`tracemalloc`), not process RSS.

| Sentences | Original ms | Original examined | Original recall on gold errors | Revised ms | Revised examined | Revised recall | Revised peak MiB |
|---|---|---|---|---|---|---|---|
| 10 | 1.0 | 10/10 | 0.800 | 1.4 | 10/10 | 1.000 | 0.02 |
| 100 | 1.3 | 12/100 | 0.123 | 13.3 | 100/100 | 0.985 | 0.22 |
| 1000 | 3.1 | 12 | 0.114 | 126.0 | all | 0.957 | 2.11 |
| 5000 | 10.8 | 12 | 0.114 | 628.4 | all | 0.957 | 10.4 |

- The original is faster on long documents **because it stops after 12 sentences**. Per
  sentence actually examined the two are close: 0.16 to 0.22 ms for the original, 0.21 ms
  for the revised system.
- The revised system is linear in document length: about 0.13 ms and 2.1 KiB per sentence.
  A 5000-sentence document (a long thesis) takes 0.6 s.
- A `cProfile` run on a 2000-sentence document attributes 57% of the time to the Checker,
  21% to Recall, 16% to the Thinker and 6% to the Viewer; most of it is `re.search`. No optimisation
  was attempted: at 0.2 ms per sentence the pipeline is not the bottleneck of any
  realistic use, and PDF text extraction dominates for uploaded files.

**This is a regression in one sense and should be read as one:** a 1000-sentence document
that took 3 ms now takes 126 ms. The 3 ms figure bought an answer about 1.2% of the document.

## 9. Not measured

| Quantity | Status |
|---|---|
| Energy consumption | **Unmeasured.** No power instrumentation was available. Nothing in this project supports an energy-efficiency claim, and operation counts are not reported as a proxy. |
| Process RSS / peak resident memory | Unmeasured; only traced Python allocations. |
| Recall on real documents | Unmeasured; no real documents with labelled errors exist for this domain. |
| Precision of the original on real prose | Count only; flags not read. |
| Concurrent throughput of the HTTP server | Unmeasured. |
| PDF / DOCX extraction time | Unmeasured. |
| Inter-annotator agreement on the benchmark labels | Not applicable: one annotator. This is a limitation. |
| NLI on GPU, or other NLI models | Unmeasured. |

## 10. Reproduce

```bash
pip install -r backend/requirements.txt -r requirements-dev.txt
bash benchmarks/reproduce.sh            # everything except NLI; needs network for Wikipedia
bash benchmarks/reproduce.sh --nli      # adds the NLI rows (pip install torch transformers)
```

Wikipedia text is cached in `benchmarks/.cache/` (git-ignored) after the first fetch.
