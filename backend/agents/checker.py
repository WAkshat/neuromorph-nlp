"""Checker - runs the enabled verifiers over every analysed claim."""

from collections import Counter

from .base import AgentMessage, AgentRole, BaseAgent, SharedMemory
from .verifiers import check_constraints, check_overclaim

RULE_VERIFIERS = {"overclaim": check_overclaim, "constraints": check_constraints}


class CheckerAgent(BaseAgent):
    role = AgentRole.CHECKER

    def __init__(self, verifiers: tuple[str, ...] = ("overclaim", "constraints"), nli=None):
        unknown = set(verifiers) - set(RULE_VERIFIERS)
        if unknown:
            raise ValueError(f"unknown verifiers {sorted(unknown)}; choose from {sorted(RULE_VERIFIERS)}")
        self.verifiers = tuple(verifiers)
        self.nli = nli      # an agents.verifiers.NLIVerifier, or None

    def run(self, memory: SharedMemory) -> AgentMessage:
        issues = []
        for analysis, evidence in zip(memory.analyses, memory.evidence):
            found = [i for name in self.verifiers for i in RULE_VERIFIERS[name](analysis)]
            if self.nli is not None:
                # NLI only adds findings the rules did not already make from the same fact.
                rule_facts = {e["id"] for i in found for e in i["evidence"]}
                found += [i for i in self.nli.check(analysis, evidence)
                          if not rule_facts & {e["id"] for e in i["evidence"]}]
            issues.extend(found)

        memory.potential_issues = issues
        by_type = Counter(i["type"] for i in issues)
        active = list(self.verifiers) + (["nli"] if self.nli is not None else [])
        return self._emit(
            memory,
            f"Found {len(issues)} potential issues in {len({i['claim_index'] for i in issues})} claims.",
            f"Verifiers run: {', '.join(active) or 'none'}. By type: {dict(by_type) or 'nothing flagged'}. "
            "A claim with no finding is not thereby verified: only contradictions the knowledge base "
            "can express are detectable.",
            issue_count=len(issues),
            high_severity=sum(1 for i in issues if i["severity"] == "high"),
            issues_by_type=dict(by_type),
            verifiers=active,
        )
