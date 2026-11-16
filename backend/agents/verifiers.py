"""Verifiers: independent checks the Checker runs over each analysed claim.

Each verifier is a function `(ClaimAnalysis) -> list[issue dict]`. They share nothing
but the knowledge base, so they can be enabled, disabled and benchmarked separately
(see benchmarks/run_eval.py --verifiers).

  overclaim    absolute / universal language that is not negated or denied
  constraints  contradictions with machine-checkable KB constraints
               (entity attributes, numeric values, attribution)
  nli          optional: a pretrained NLI cross-encoder compares the claim with the
               retrieved facts (contradiction) and with known myths (entailment)
"""

import re

from knowledge import neuromorphic_kb as kb
from textutils import ClaimAnalysis, negated_after, negated_before

# --- overclaim ---------------------------------------------------------------------

# Nouns that make "all / every / any X" a claim about scope rather than a description
# ("every timestep" is fine, "every workload" is a universal claim).
_SCOPE = (r"(?:case|task|workload|application|benchmark|condition|platform|problem|scenario|situation|"
          r"dataset|device|chip|model|network|architecture|hardware|system|setting|domain)s?")
# Absolutist vocabulary, adapted to technical writing from the absolutist-word
# dictionary of Al-Mosaiwi & Johnstone (Clin. Psychol. Sci. 6(4), 2018).
ABSOLUTE_PATTERNS: list[tuple[str, str]] = [
    ("always", r"\balways\b"),
    ("never", r"\bnever\b"),
    ("guarantee", r"\bguarantee[sd]?\b"),
    ("universal quantifier", rf"\b(?:all|every|any)\s+(?:\w+\s+){{0,2}}?{_SCOPE}\b|\bin\s+all\b"),
    ("universally", r"\buniversally\b|\bacross\s+the\s+board\b|\bwithout\s+exception\b|\bregardless\s+of\b"
                    r"|\bwhatever\b|\bwhichever\b|\bwhatsoever\b"),
    # "built entirely from", "composed completely of" describe composition, not scope.
    ("totality adverb", r"\b(?:completely|entirely|totally|absolutely|perfectly)\b(?!\s+(?:from|of|by|in|on|with)\b)|\bfully\b(?![- ](?:connected|convolutional))"),
    ("perfect", r"\bperfect\b|\bflawless\w*|\b100\s?%|\bunlimited\b|\binfinite\b"),
    ("impossible", r"\bimpossible\b"),
    ("zero cost", r"\b(?:zero|no)\s+(?:\w+\s+)?(?:energy|power)\b"),
    ("solves everything", r"\bsolve[sd]?\s+all\b"),
]
_ABSOLUTE_RES = [(name, re.compile(pat, re.IGNORECASE)) for name, pat in ABSOLUTE_PATTERNS]


def find_absolutes(analysis: ClaimAnalysis) -> list[dict]:
    """Absolute terms in the claim, each marked negated or not. Called by the Thinker."""
    found = []
    for clause in analysis.clauses:
        for name, rx in _ABSOLUTE_RES:
            for m in rx.finditer(clause):
                # "not always", "does not guarantee", "not every chip": cue right before the term.
                negated = analysis.negation_aware and (
                    analysis.denial or negated_before(clause, m.start(), max_gap_tokens=1))
                found.append({"kind": name, "term": m.group(0), "negated": negated})
    return found


def check_overclaim(analysis: ClaimAnalysis) -> list[dict]:
    terms = [a["term"] for a in analysis.absolutes if not a["negated"]]
    if not terms:
        return []
    return [{
        "type": "overclaim_absolute_language",
        "severity": "medium",
        "verifier": "overclaim",
        "claim": analysis.text,
        "claim_index": analysis.index,
        "matched_myth": None,
        "explanation": (
            f"Unqualified absolute language ({', '.join(repr(t) for t in dict.fromkeys(terms))}). "
            "Universal claims are rarely supported by evidence in this field."
        ),
        "suggestion": "State the conditions, benchmark or workload under which the claim holds.",
        "evidence": [],
    }]


# --- KB constraints ----------------------------------------------------------------

_HEDGE = re.compile(
    r"\b(?:can|may|might|could|some|sometimes|certain|often|typically|usually|many|several|"
    r"our|we|this\s+(?:work|paper|model|study)|reported|in\s+principle)\b", re.IGNORECASE)

_NUM_WORDS = {"a": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
              "eight": 8, "nine": 9, "ten": 10, "hundred": 100}
_SCALE = {"thousand": 1e3, "k": 1e3, "million": 1e6, "m": 1e6, "billion": 1e9, "bn": 1e9, "trillion": 1e12}
_QUANTITY = re.compile(
    r"(?<!loihi )\b(\d[\d,]*(?:\.\d+)?|" + "|".join(_NUM_WORDS) + r")\s*(" + "|".join(_SCALE) + r")?\s+"
    r"(?:\w+\s+){0,2}?(neurons|synapses|cores)\b", re.IGNORECASE)

_CREATION_VERB = r"(?:developed|designed|built|created|made|introduced|invented|produced|coined|proposed|released)"
_NAME = r"(?:[A-Z][\w.&-]*(?:\s+(?:of\s+)?[A-Z][\w.&-]*){0,3})"


def _evidence(fact_id: str) -> list[dict]:
    f = kb.get_fact(fact_id)
    return [{"id": f.id, "content": f.text, "citation": f.citation}]


def _issue(analysis: ClaimAnalysis, type_: str, explanation: str, fact_id: str, constraint_id: str) -> dict:
    return {
        "type": type_,
        "severity": "high",
        "verifier": "constraints",
        "claim": analysis.text,
        "claim_index": analysis.index,
        "matched_myth": None,
        "explanation": explanation,
        "suggestion": f"Check against: {kb.get_fact(fact_id).citation}.",
        "evidence": _evidence(fact_id),
        "constraint": constraint_id,
    }


def _asserts(clause: str, attribute: str, negation_aware: bool) -> tuple[bool, bool]:
    """(attribute mentioned, attribute affirmed) for one clause."""
    mentioned = affirmed = False
    for m in re.finditer(attribute, clause, re.IGNORECASE):
        mentioned = True
        if not (negation_aware and (negated_before(clause, m.start()) or negated_after(clause, m.end()))):
            affirmed = True
    return mentioned, affirmed


def _check_attributes(analysis: ClaimAnalysis) -> list[dict]:
    issues = []
    for c in kb.ATTRIBUTE_CONSTRAINTS:
        if c.context and not re.search(c.context, analysis.text, re.IGNORECASE):
            continue
        for clause in analysis.clauses:
            if not re.search(c.entity, clause, re.IGNORECASE):
                continue
            if c.generic and _HEDGE.search(clause):
                continue
            mentioned, affirmed = _asserts(clause, c.attribute, analysis.negation_aware)
            if not mentioned:
                continue
            claim_says_holds = affirmed != analysis.denial      # a denial frame flips polarity
            if claim_says_holds != c.holds:
                issues.append(_issue(analysis, "contradicts_knowledge_base", c.message, c.fact_id, c.id))
                break
    return issues


def _parse_quantity(number: str, scale: str | None) -> float | None:
    n = number.lower()
    if n in _NUM_WORDS and not scale:
        return None     # "one of the neurons", "a few cores": not a count
    value = float(_NUM_WORDS[n]) if n in _NUM_WORDS else float(n.replace(",", ""))
    return value * _SCALE.get((scale or "").lower(), 1)


def _check_numbers(analysis: ClaimAnalysis) -> list[dict]:
    if analysis.denial:
        return []
    issues = []
    for clause in analysis.clauses:
        quantities = [(v, m.group(3).lower(), m.group(0)) for m in _QUANTITY.finditer(clause)
                      if (v := _parse_quantity(m.group(1), m.group(2))) is not None]
        if not quantities:
            continue
        # A number is only attributable when the clause is about exactly one known chip.
        chips = {c.entity for c in kb.NUMERIC_CONSTRAINTS if re.search(c.entity, clause, re.IGNORECASE)}
        if len(chips) != 1:
            continue
        for c in kb.NUMERIC_CONSTRAINTS:
            if c.entity not in chips or (c.exclude and re.search(c.exclude, clause, re.IGNORECASE)):
                continue
            for value, noun, span in quantities:
                if noun == c.noun and not any(0.5 <= value / a <= 2 for a in c.accepted):
                    accepted = " or ".join(f"{a:,.0f}" for a in c.accepted)
                    issues.append(_issue(
                        analysis, "numeric_mismatch",
                        f"'{span}' does not match the documented figure (about {accepted} {c.noun}).",
                        c.fact_id, c.id))
    return issues


def _check_attribution(analysis: ClaimAnalysis) -> list[dict]:
    if analysis.denial:
        return []
    issues = []
    for c in kb.ATTRIBUTION_CONSTRAINTS:
        ent = c.entity
        for clause in analysis.clauses:
            agents = []
            # "IBM's Loihi", "Intel TrueNorth"
            agents += [m.group(1) for m in re.finditer(rf"\b({kb.KNOWN_ORGS})(?:['’]s)?\s+{ent}\b", clause)]
            # "Loihi ... was designed by IBM", "SpiNNaker ... built at Heidelberg University"
            agents += [m.group(1) for m in re.finditer(
                rf"{ent}\b[^.;]{{0,80}}?\b{_CREATION_VERB}\s+(?:\w+\s+){{0,2}}?(?:by|at)\s+(?:the\s+)?({_NAME})", clause)]
            # "Wolfgang Maass introduced the Loihi chip"
            agents += [m.group(1) for m in re.finditer(
                rf"({_NAME})(?:\s+et\s+al\.?)?\s+{_CREATION_VERB}\s+(?:the\s+)?(?:\w+\s+){{0,2}}?{ent}\b", clause)]
            wrong = [a for a in agents
                     if not re.search(c.accepted, a, re.IGNORECASE)
                     and not re.match(r"(?:The|A|An|Our|We|This|These|Researchers|Engineers|Scientists)\b", a)]
            if wrong:
                issues.append(_issue(
                    analysis, "misattribution",
                    f"The claim attributes this to '{wrong[0]}'; the knowledge base attributes it to {c.owner}.",
                    c.fact_id, c.id))
                break
    return issues


def check_constraints(analysis: ClaimAnalysis) -> list[dict]:
    return _check_attributes(analysis) + _check_numbers(analysis) + _check_attribution(analysis)


# --- optional NLI verifier ---------------------------------------------------------

class NLIVerifier:
    """Natural-language-inference check with a pretrained cross-encoder.

    Two uses of one model:
      * P(contradiction | premise = retrieved fact, hypothesis = claim)  -> contradicts KB
      * P(entailment    | premise = claim, hypothesis = known myth)      -> restates a myth
    Requires `torch` and `transformers` and downloads the model on first use, which is
    why it is opt-in. On the dev split the contradiction arm produced high-confidence
    false positives on correct statements (docs/BENCHMARKS.md), so it is disabled
    unless a threshold is given.
    """

    def __init__(self, model_name: str, myth_threshold: float, contradiction_threshold: float | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.contradiction, self.entailment = labels["contradiction"], labels["entailment"]
        self.contradiction_threshold = contradiction_threshold
        self.myth_threshold = myth_threshold

    def _probs(self, pairs: list[tuple[str, str]]):
        with self.torch.no_grad():
            enc = self.tokenizer([p for p, _ in pairs], [h for _, h in pairs],
                                 padding=True, truncation=True, max_length=256, return_tensors="pt")
            return self.model(**enc).logits.softmax(-1)

    def check(self, analysis: ClaimAnalysis, evidence: list[dict]) -> list[dict]:
        if analysis.denial:
            return []       # "it is a myth that X" entails X for an NLI model; skip rather than misfire
        if self.contradiction_threshold is None:
            evidence = []
        pairs = [(e["content"], analysis.text) for e in evidence] + [(analysis.text, m.claim) for m in kb.MYTHS]
        probs = self._probs(pairs)
        issues = []
        fact_p = probs[:len(evidence), self.contradiction]
        if len(evidence) and float(fact_p.max()) >= self.contradiction_threshold:
            e = evidence[int(fact_p.argmax())]
            issues.append({
                "type": "nli_contradiction", "severity": "high", "verifier": "nli",
                "claim": analysis.text, "claim_index": analysis.index, "matched_myth": None,
                "explanation": f"An NLI model judges this claim to contradict the knowledge base: \"{e['content']}\"",
                "suggestion": f"Check against: {e['citation']}.",
                "evidence": [{"id": e["id"], "content": e["content"], "citation": e["citation"]}],
                "confidence": round(float(fact_p.max()), 3),
            })
        myth_p = probs[len(evidence):, self.entailment]
        if float(myth_p.max()) >= self.myth_threshold:
            myth = kb.MYTHS[int(myth_p.argmax())]
            issues.append({
                "type": "possible_common_mistake", "severity": "high", "verifier": "nli",
                "claim": analysis.text, "claim_index": analysis.index,
                "matched_myth": f"{myth.claim} - {myth.correction}",
                "explanation": f"An NLI model judges this claim to restate a known misconception. {myth.correction}",
                "suggestion": f"Check against: {kb.get_fact(myth.fact_id).citation}.",
                "evidence": _evidence(myth.fact_id),
                "confidence": round(float(myth_p.max()), 3),
            })
        return issues
