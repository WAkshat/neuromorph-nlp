"""Orchestrator - builds the pipeline from a config and runs it over a document."""

import uuid
from dataclasses import dataclass

from .base import SharedMemory
from .checker import CheckerAgent
from .recall import RecallAgent
from .synthesizer import SynthesizerAgent
from .thinker import ThinkerAgent
from .viewer import ViewerAgent

MIN_DOCUMENT_CHARS = 20


class DocumentTooShort(ValueError):
    pass


@dataclass(frozen=True)
class PipelineConfig:
    verifiers: tuple[str, ...] = ("overclaim", "constraints")
    negation_aware: bool = True       # False reproduces the original polarity-blind behaviour (ablation)
    top_k: int = 3                    # evidence facts retrieved per claim
    max_claims: int = 5000
    nli: bool = False                 # optional model-based verifier; needs torch + transformers
    nli_model: str = "cross-encoder/nli-MiniLM2-L6-H768"
    nli_myth_threshold: float = 0.9
    nli_contradiction_threshold: float | None = None   # None = contradiction arm off (see docs/BENCHMARKS.md)


class MultiAgentOrchestrator:
    """Stateless after construction, so one instance can serve concurrent requests."""

    def __init__(self, config: PipelineConfig = PipelineConfig()):
        self.config = config
        nli = None
        if config.nli:
            from .verifiers import NLIVerifier
            nli = NLIVerifier(config.nli_model, config.nli_myth_threshold, config.nli_contradiction_threshold)
        self.agents = [
            ViewerAgent(config.max_claims),
            ThinkerAgent(config.negation_aware),
            RecallAgent(config.top_k),
            CheckerAgent(config.verifiers, nli),
            SynthesizerAgent(),
        ]

    def process_document(self, text: str) -> dict:
        """Run viewer -> thinker -> recall -> checker -> synthesizer."""
        if not isinstance(text, str) or len(text.strip()) < MIN_DOCUMENT_CHARS:
            raise DocumentTooShort(f"Document must contain at least {MIN_DOCUMENT_CHARS} characters of text.")
        memory = SharedMemory(document_id=str(uuid.uuid4()), original_text=text.strip())
        for agent in self.agents:
            agent.run(memory)
        return memory.to_dict()
