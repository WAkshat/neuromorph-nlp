"""Deterministic text analysis shared by the agents: tokenising, sentence and clause
segmentation, and negation / denial detection.

Everything here is rule-based on purpose: it is fast, inspectable, and has no model
dependency. Its known ceiling is documented in docs/FUTURE_RESEARCH.md (no syntactic
parse, so negation scope is approximated by clause boundaries and token windows).
"""

import re
from dataclasses import dataclass, field

STOPWORDS = frozenset("""
a an the and or but if then than that this these those is are was were be been being
am do does did doing have has had having of in on at to for from by with as into over
under about between through during it its they them their there here we our you your
he she his her i me my not no nor so such can could may might will would shall should
must also very more most much many some any all each both other which who whom whose
what when where why how only just up out off again further once own same too s t
""".split())

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercased word tokens; hyphenated words are kept whole AND split, so that
    'on-chip' matches both 'on-chip' and 'chip'."""
    out = []
    for tok in _TOKEN_RE.findall(text.lower()):
        out.append(tok)
        if "-" in tok:
            out.extend(p for p in tok.split("-") if p)
    return out


def content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 1]


# --- sentence segmentation ---------------------------------------------------------

_ABBREV_END = re.compile(
    r"(?:\b(?i:et al|e\.g|i\.e|approx|fig|figs|eq|eqs|vs|cf|dr|prof|ref|refs)|\b[A-Z])\.$"
)
_SENT_BOUNDARY = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]*[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences without breaking on common abbreviations.

    Blank lines always end a sentence (headings, list items); single newlines are
    treated as soft wraps, which is what PDF extraction produces.
    """
    sentences = []
    for block in re.split(r"\n\s*\n", text):
        block = re.sub(r"\s+", " ", block).strip()
        if not block:
            continue
        current = ""
        for part in _SENT_BOUNDARY.split(block):
            current = f"{current} {part}" if current else part
            if not _ABBREV_END.search(current):
                sentences.append(current)
                current = ""
        if current:
            sentences.append(current)
    return sentences


# --- clauses, negation, denial -----------------------------------------------------

# Words that start a new clause whose polarity is independent of the previous one.
_CLAUSE_SPLIT = re.compile(
    r";|:\s|,?\s+\b(?:whereas|while|but|although|though|however|yet|unlike|instead of|"
    r"rather than|in contrast to|as opposed to)\b",
    re.IGNORECASE,
)
# "Unlike X, Y ..." / "Whereas X does A, Y ..." at the start of a sentence.
_LEADING_SUBORDINATE = re.compile(
    r"^\s*(?:whereas|while|although|though|unlike|in contrast to|as opposed to)\b([^,]*),\s*(.+)$",
    re.IGNORECASE | re.DOTALL,
)
_NEGATION = re.compile(
    r"\b(?:not|no|never|none|neither|nor|cannot|without|lacks?|lacking|few|rarely|seldom|"
    r"hardly|far from|fails? to|unable|ignor\w+|discard\w+)\b|n['’]t\b",
    re.IGNORECASE,
)
# Sentence-level framing that marks the embedded proposition as false.
_DENIAL = re.compile(
    r"\b(?:myths?|misconceptions?|incorrect(?:ly)?|false(?:ly)?|wrong(?:ly)?|mistaken(?:ly)?|"
    r"misleading|untrue|fallacy|contrary to|overstate[sd]?|overstat\w+|exaggerat\w+)\b",
    re.IGNORECASE,
)


@dataclass
class ClaimAnalysis:
    """What the Thinker knows about one claim; consumed by Recall and the Checker."""
    index: int
    text: str
    clauses: list[str]
    denial: bool                       # sentence frames its content as false ("it is a myth that ...")
    absolutes: list[dict] = field(default_factory=list)   # filled by the Thinker
    negation_aware: bool = True


def split_clauses(sentence: str) -> list[str]:
    m = _LEADING_SUBORDINATE.match(sentence)
    parts = [m.group(1), m.group(2)] if m else [sentence]
    clauses = []
    for part in parts:
        clauses.extend(c.strip() for c in _CLAUSE_SPLIT.split(part) if c and c.strip())
    return clauses


def analyze(index: int, text: str, negation_aware: bool = True) -> ClaimAnalysis:
    """With negation_aware=False the sentence is one always-affirmative clause: this
    is the ablation that reproduces the original system's polarity blindness."""
    if not negation_aware:
        return ClaimAnalysis(index, text, [text], denial=False, negation_aware=False)
    return ClaimAnalysis(index, text, split_clauses(text), denial=bool(_DENIAL.search(text)))


def negated_before(clause: str, start: int, max_gap_tokens: int | None = None) -> bool:
    """Is there a negation cue in `clause` before character offset `start`?

    max_gap_tokens limits how many words may sit between the cue and `start`
    ("not always" yes, "not receiving spikes burns no power" no).
    """
    for m in _NEGATION.finditer(clause[:start]):
        if max_gap_tokens is None:
            return True
        if len(clause[m.end():start].split()) <= max_gap_tokens:
            return True
    return False


def negated_after(clause: str, end: int, max_gap_tokens: int = 2) -> bool:
    """Negation cue shortly after offset `end` ("on-chip learning is not supported")."""
    m = _NEGATION.search(clause[end:])
    return bool(m) and len(clause[end:end + m.start()].split()) <= max_gap_tokens
