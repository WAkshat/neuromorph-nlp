# Future research

Everything in sections 2 to 4 is a **hypothesis or a proposal**. Only section 1 is
established, and only to the extent stated there.

## 1. What is established

| Result | Evidence | Strength |
|---|---|---|
| The original flags correct denials of myths | 7/10 on held-out, reproduced by hand | Strong: deterministic, mechanism understood |
| Negation handling removes those false positives | ablation, 10 FP to 0 | Strong on this benchmark |
| Constraints catch entity/number/attribution errors the KB encodes | 14/14 held-out | Moderate: benchmark and rules share an author |
| BM25 retrieves the cited fact far more often than substring overlap | recall@3 0.89 vs 0.15 | Moderate: silver labels |
| A small MNLI cross-encoder with multi-sentence premises is not a usable contradiction detector here | dev sweep, test run | Moderate: one model, one premise format |
| About 1.8% of ordinary encyclopedic sentences are flagged, none correctly | 736 held-out sentences, read by one person | Moderate: one reader, one genre |

## 2. Outstanding technical limitations

1. **Universal statements that are true.** The overclaim verifier cannot distinguish
   "SNNs always win" from "a correlation matrix is always positive-definite". This is the
   largest source of false alarms and it is semantic, not lexical.
2. **Clause-level entity binding.** The numeric check credited "800 trillion synapses" to
   TrueNorth because both appeared in one clause. Constraints need to know which noun
   phrase a property attaches to.
3. **Paraphrase.** Rules miss restatements without trigger words.
4. **KB coverage.** 29 facts. No process for growing or reviewing it.
5. **No discourse.** One sentence at a time; pronouns and claims spanning sentences are
   invisible.
6. **Evaluation.** One annotator, 140 sentences, same author as the system.

## 3. Experiments worth running next, in priority order

### 3.1 An independent, larger benchmark  (prerequisite for everything else)
- **What**: 500+ sentences from real neuromorphic papers, preprints, press releases and
  student essays, labelled by two people who have not seen the rules, with agreement
  reported (Cohen's kappa).
- **Why first**: every other experiment below is uninterpretable on 140 self-written items.
- **Expected**: F1 well below 0.97. The size of the drop is the real measure of the system.

### 3.2 Dependency-parse scope for negation and entity binding
- **Hypothesis**: replacing clause-and-window heuristics with dependency scope (negation
  attached to the governing verb; numbers attached to their head noun's subject) removes
  the constraint misfires in BENCHMARKS section 6 without losing recall.
- **How**: spaCy `en_core_web_sm`; compare constraint-verifier false alarms on the
  held-out Wikipedia set and F1 on the benchmark.
- **Cost**: one dependency, roughly 10 ms per sentence.

### 3.3 NLI with sentence-level premises and a stronger model
- **Hypothesis**: the contradiction arm failed because premises were multi-sentence facts
  mentioning several entities. Splitting facts into single-sentence premises (the
  granularity finding of Laban et al., SummaC, TACL 2022) and requiring the premise and
  claim to share a KB entity will remove the false positives seen on dev.
- **How**: rerun `benchmarks/nli_sweep.py` with sentence-level premises; then try a model
  trained on FEVER and ANLI in addition to MNLI.
- **Success criterion**: at least +2 true positives on paraphrased myths with 0 new false
  positives on dev, before touching test.

### 3.4 Separating overclaims from true universals
- **Hypothesis**: a small supervised classifier over the sentences the lexicon fires on
  (features: subject type, whether the predicate is evaluative or comparative, presence of
  a performance term) can halve false alarms.
- **Needs**: 3.1 first, for training data. A logistic-regression baseline should come
  before anything larger.

### 3.5 LLM verification as a second opinion
- **Hypothesis**: an LLM given the claim plus retrieved facts, asked for a verdict with a
  quoted supporting span, catches paraphrases and handles true universals, at the cost of
  latency, money and its own errors.
- **How**: run it only on claims the rules do not decide; measure against 3.1; report
  disagreement with the rules rather than letting it overrule them silently.
- **Related work**: multi-agent debate (Du et al., arXiv:2305.14325, 2023). Whether
  several LLM agents beat one on this task is an open empirical question, not a given.

### 3.6 Retrieval score threshold
- BM25 returns about one fact for off-topic sentences. Calibrate a minimum score on 3.1.

## 4. The neuromorphic question

The honest position is that this project's task gives a spiking network nothing to
exploit, so none was built (`ARCHITECTURE_PROPOSAL.md`, candidate E). If the aim is to make
the project genuinely neuromorphic rather than about neuromorphic computing, these are the
experiments that would produce a real answer instead of a label:

### 4.1 Does a spiking classifier match an ANN on overclaim detection?
- **Setup**: sentence embeddings (a frozen MiniLM encoder) as input; compare (a) logistic
  regression, (b) a 2-layer MLP, (c) a 2-layer LIF network trained with surrogate gradients
  (Neftci, Mostafa & Zenke, 2019) using rate or latency encoding over T timesteps. snnTorch
  (Eshraghian et al., Proc. IEEE 2023) or Norse.
- **Report**: accuracy over at least 5 seeds with variance; mean spikes per inference;
  accuracy as a function of T.
- **What a positive result would mean**: parity with the MLP at low spike counts would be
  a small, honest result about text classification with SNNs. It would not make the
  reviewer better.
- **What must not be claimed**: energy efficiency. Spike counts on a CPU simulation are a
  proxy. A claim needs measurement on neuromorphic hardware (Loihi 2 via Lava) or at
  minimum a stated energy model with its assumptions.
- **Prior work to position against**: Lv et al., "Spiking Convolutional Neural Networks
  for Text Classification" (ICLR 2023); Zhu et al., "SpikeGPT" (arXiv:2302.13939).

### 4.2 Event-driven streaming review
- A reviewer that processes text as it is typed, updating only the state touched by each
  new token, is the nearest thing to an event-driven formulation of this task. Whether a
  spiking implementation offers anything over an incremental conventional one is unknown.

### 4.3 If multi-agent research is the aim
- The task needs a reason for agents to disagree. A defensible setup: independent
  verifiers with different knowledge sources and known error profiles, combined by a
  learned or weighted vote, evaluated against the best single verifier. The verifier
  interface in `agents/verifiers.py` already supports this; what is missing is verifiers
  that are both independent and individually useful (today one of three is not).

## 5. Potential contributions, stated at their true size

- **A small diagnostic benchmark** for claim checkers in this domain whose distinctive
  feature is the *correct-negation* category, which exposes polarity-blind keyword
  checkers. Useful as a unit test for such tools; too small and too single-authored to be
  a research benchmark as it stands.
- **A negative result**: an off-the-shelf small NLI cross-encoder, used naively with
  multi-sentence premises, is harmful as a contradiction detector for domain facts. This is
  consistent with what the summarisation-consistency literature reports about granularity.
- **A worked example** of auditing a "multi-agent / neuromorphic" project and replacing
  the labels with measurements.

None of these is novel research. The engineering is sound; the science is a baseline.
