"""Recall - retrieves knowledge-base evidence for every claim (BM25)."""

from knowledge.neuromorphic_kb import search_knowledge

from .base import AgentMessage, AgentRole, BaseAgent, SharedMemory


class RecallAgent(BaseAgent):
    role = AgentRole.RECALL

    def __init__(self, top_k: int = 3):
        self.top_k = top_k

    def run(self, memory: SharedMemory) -> AgentMessage:
        memory.evidence = [search_knowledge(a.text, self.top_k) for a in memory.analyses]
        union: dict[str, dict] = {}
        for hits in memory.evidence:
            for h in hits:
                if h["id"] not in union or h["score"] > union[h["id"]]["score"]:
                    union[h["id"]] = h
        memory.retrieved_knowledge = sorted(union.values(), key=lambda h: -h["score"])

        with_evidence = sum(1 for hits in memory.evidence if hits)
        return self._emit(
            memory,
            f"Retrieved evidence for {with_evidence} of {len(memory.evidence)} claims "
            f"({len(memory.retrieved_knowledge)} distinct knowledge-base facts).",
            f"BM25 retrieval, top {self.top_k} facts per claim. Claims with no lexical overlap "
            "with the knowledge base get no evidence and cannot be fact-checked.",
            retrieved_count=len(memory.retrieved_knowledge),
            claims_with_evidence=with_evidence,
            knowledge_preview=[h["id"] for h in memory.retrieved_knowledge[:5]],
        )
