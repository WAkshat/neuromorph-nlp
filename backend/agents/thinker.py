"""Thinker - linguistic analysis of each claim.

Produces one ClaimAnalysis per claim (clauses, denial framing, absolute terms with
their negation status). Recall and the Checker both consume it, so
unlike the original Thinker its output actually changes the result.
"""

from textutils import analyze

from .base import AgentMessage, AgentRole, BaseAgent, SharedMemory
from .verifiers import find_absolutes


class ThinkerAgent(BaseAgent):
    role = AgentRole.THINKER

    def __init__(self, negation_aware: bool = True):
        self.negation_aware = negation_aware

    def run(self, memory: SharedMemory) -> AgentMessage:
        memory.analyses = []
        for i, claim in enumerate(memory.extracted_claims):
            a = analyze(i, claim, self.negation_aware)
            a.absolutes = find_absolutes(a)
            memory.analyses.append(a)

        risky = [a for a in memory.analyses if any(not x["negated"] for x in a.absolutes)]
        negated = sum(1 for a in memory.analyses if a.denial or any(x["negated"] for x in a.absolutes))
        return self._emit(
            memory,
            f"Analysed {len(memory.analyses)} claims: {len(risky)} use unqualified absolute language.",
            f"{negated} claims negate or deny an absolute statement and will not be treated as overclaims.",
            claim_count=len(memory.analyses),
            risk_areas=[{"claim_index": a.index, "claim": a.text[:150],
                         "risk_terms": [x["term"] for x in a.absolutes if not x["negated"]]}
                        for a in risky[:50]],
        )
