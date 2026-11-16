"""Synthesizer - turns the findings into the final report."""

from .base import AgentMessage, AgentRole, BaseAgent, SharedMemory

RECOMMENDATIONS = {
    "overclaim_absolute_language":
        "Replace absolute language (always / never / guarantees / all cases) with the conditions under which the claim holds.",
    "contradicts_knowledge_base":
        "Correct the statements that contradict documented facts; each finding cites its source.",
    "numeric_mismatch":
        "Re-check the reported neuron / synapse / core counts against the original chip papers.",
    "misattribution":
        "Fix the attribution of chips and terms to the right organisation or author.",
    "nli_contradiction":
        "Review the claims an NLI model judged to contradict the knowledge base (model-based, verify manually).",
    "possible_common_mistake":
        "Rephrase claims that restate a known misconception about neuromorphic computing.",
}


class SynthesizerAgent(BaseAgent):
    role = AgentRole.SYNTHESIZER

    def run(self, memory: SharedMemory) -> AgentMessage:
        issues = memory.potential_issues
        high = sum(1 for i in issues if i["severity"] == "high")
        medium = sum(1 for i in issues if i["severity"] == "medium")
        flagged = {i["claim_index"] for i in issues}
        n_claims = len(memory.extracted_claims)

        if issues:
            assessment = (f"{len(flagged)} of {n_claims} claims were flagged "
                          f"({len(issues)} findings: {high} high, {medium} medium severity).")
            recommendations = [text for type_, text in RECOMMENDATIONS.items()
                               if any(i["type"] == type_ for i in issues)]
        else:
            assessment = (f"None of the {n_claims} claims were flagged. No overclaim wording and no "
                          "contradiction with the built-in knowledge base was detected; this does not "
                          "mean the document is correct.")
            recommendations = ["Have a domain expert verify quantitative results and claims outside the knowledge base."]

        memory.final_report = {
            "overall_assessment": assessment,
            "claim_count": n_claims,
            "claims_flagged": len(flagged),
            "issues_found": len(issues),
            "high_severity": high,
            "medium_severity": medium,
            "knowledge_items_used": len(memory.retrieved_knowledge),
            "recommendations": recommendations,
            "detailed_issues": issues,
            "claim_verdicts": [
                {"claim_index": a.index,
                 "verdict": "flagged" if a.index in flagged else "no_issue_found",
                 "evidence_ids": [e["id"] for e in ev]}
                for a, ev in zip(memory.analyses, memory.evidence)
            ],
            "agent_collaboration_note": (
                "Sequential pipeline with a shared blackboard: Viewer (segmentation) -> Thinker (negation and "
                "absolute-language analysis) -> Recall (BM25 evidence) -> Checker (verifiers) -> Synthesizer."
            ),
        }
        return self._emit(memory, f"Synthesis complete. {assessment}",
                          "Recommendations are derived from the issue types actually found.",
                          report_keys=list(memory.final_report))
