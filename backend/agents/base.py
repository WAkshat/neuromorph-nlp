"""Base classes and shared state for the review pipeline."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from textutils import ClaimAnalysis


class AgentRole(str, Enum):
    VIEWER = "viewer"            # segments the document into candidate claims
    THINKER = "thinker"          # linguistic analysis: clauses, negation, absolute terms
    RECALL = "recall"            # retrieves evidence per claim
    CHECKER = "checker"          # runs the verifiers
    SYNTHESIZER = "synthesizer"  # produces the report


@dataclass
class AgentMessage:
    role: AgentRole
    content: str
    thought: str = ""
    # Only set when an agent has a measured quantity to report; None otherwise.
    # (The original hard-coded 0.95 / 0.88 / 0.92 / 0.85 / 0.9 here.)
    confidence: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict = field(default_factory=dict)


@dataclass
class SharedMemory:
    """Blackboard the stages read from and write to. Each field has one writer."""
    document_id: str
    original_text: str
    extracted_claims: list[str] = field(default_factory=list)          # Viewer
    analyses: list[ClaimAnalysis] = field(default_factory=list)        # Thinker
    evidence: list[list[dict]] = field(default_factory=list)           # Recall (per claim)
    retrieved_knowledge: list[dict] = field(default_factory=list)      # Recall (deduplicated union)
    potential_issues: list[dict] = field(default_factory=list)         # Checker
    agent_trace: list[AgentMessage] = field(default_factory=list)
    final_report: Optional[dict] = None                                # Synthesizer

    def add_trace(self, msg: AgentMessage):
        self.agent_trace.append(msg)

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "extracted_claims": self.extracted_claims,
            "retrieved_knowledge": self.retrieved_knowledge,
            "potential_issues": self.potential_issues,
            "agent_trace": [
                {
                    "role": m.role.value,
                    "content": m.content,
                    "thought": m.thought,
                    "confidence": m.confidence,
                    "timestamp": m.timestamp,
                    "metadata": m.metadata,
                }
                for m in self.agent_trace
            ],
            "final_report": self.final_report,
        }


class BaseAgent:
    role: AgentRole

    def run(self, memory: SharedMemory) -> AgentMessage:
        raise NotImplementedError

    def _emit(self, memory: SharedMemory, content: str, thought: str, **metadata) -> AgentMessage:
        msg = AgentMessage(role=self.role, content=content,
                           thought=f"[{self.role.value.upper()}] {thought}", metadata=metadata)
        memory.add_trace(msg)
        return msg
