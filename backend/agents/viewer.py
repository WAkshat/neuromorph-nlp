"""Viewer - segments the document into candidate claims (sentences)."""

from textutils import split_sentences, tokenize

from .base import AgentMessage, AgentRole, BaseAgent, SharedMemory

MIN_WORDS = 3     # drops headings ("Related Work") but keeps "SNNs always win."


class ViewerAgent(BaseAgent):
    role = AgentRole.VIEWER

    def __init__(self, max_claims: int = 5000):
        self.max_claims = max_claims

    def run(self, memory: SharedMemory) -> AgentMessage:
        sentences = split_sentences(memory.original_text)
        # Every sentence with enough words is a candidate. Deciding which ones are
        # risky is the Thinker's and Checker's job, not a keyword filter here.
        claims = [s for s in sentences if len(tokenize(s)) >= MIN_WORDS]
        truncated = len(claims) > self.max_claims
        memory.extracted_claims = claims[: self.max_claims]

        n_words = len(memory.original_text.split())
        thought = (f"Document has ~{n_words} words in {len(sentences)} sentences; "
                   f"{len(memory.extracted_claims)} are long enough to be checked.")
        if truncated:
            thought += f" Truncated to the first {self.max_claims} (max_claims); the rest was NOT reviewed."
        return self._emit(
            memory,
            f"Extracted {len(memory.extracted_claims)} candidate claims.",
            thought,
            word_count=n_words, sentence_count=len(sentences),
            claim_count=len(memory.extracted_claims), truncated=truncated,
            claims_preview=memory.extracted_claims[:3],
        )
