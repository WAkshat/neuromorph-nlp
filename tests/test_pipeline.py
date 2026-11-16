import csv
from pathlib import Path

import pytest

from agents.orchestrator import DocumentTooShort, MultiAgentOrchestrator, PipelineConfig

README_EXAMPLE = (
    "Neuromorphic computing always replaces GPUs because spiking neural networks are more accurate "
    "and use zero energy when idle. Intel Loihi and IBM TrueNorth both support rich on-chip learning "
    "with STDP. SNNs guarantee better energy efficiency in all cases."
)
DATA = Path(__file__).resolve().parent.parent / "benchmarks" / "data"


@pytest.fixture(scope="module")
def orch():
    return MultiAgentOrchestrator()


def test_readme_example_flags_all_three_sentences(orch):
    r = orch.process_document(README_EXAMPLE)
    assert r["final_report"]["claims_flagged"] == 3
    assert {i["type"] for i in r["potential_issues"]} == {"overclaim_absolute_language", "contradicts_knowledge_base"}
    assert [m["role"] for m in r["agent_trace"]] == ["viewer", "thinker", "recall", "checker", "synthesizer"]


def test_response_keeps_the_original_api_shape(orch):
    r = orch.process_document(README_EXAMPLE)
    assert {"document_id", "extracted_claims", "retrieved_knowledge", "potential_issues",
            "agent_trace", "final_report"} <= set(r)
    assert {"overall_assessment", "claim_count", "issues_found", "high_severity", "medium_severity",
            "knowledge_items_used", "recommendations", "detailed_issues",
            "agent_collaboration_note"} <= set(r["final_report"])
    for issue in r["potential_issues"]:
        assert {"type", "severity", "claim", "matched_myth", "explanation", "suggestion"} <= set(issue)


def test_clean_document_is_not_flagged(orch):
    r = orch.process_document(
        "TrueNorth does not support on-chip learning. SNNs are not always more accurate than ANNs. "
        "For sparse workloads, neuromorphic chips can be more energy efficient than GPUs.")
    assert r["potential_issues"] == []
    assert "does not mean the document is correct" in r["final_report"]["overall_assessment"]


def test_long_documents_are_checked_to_the_end(orch):
    """The original examined only the first 12 sentences."""
    filler = "The membrane potential is updated once per simulation step. " * 300
    r = orch.process_document(filler + "TrueNorth learns with on-chip plasticity during deployment.")
    assert len(r["extracted_claims"]) == 301
    assert r["potential_issues"][-1]["claim_index"] == 300


def test_max_claims_truncation_is_reported_not_silent():
    r = MultiAgentOrchestrator(PipelineConfig(max_claims=2)).process_document(README_EXAMPLE)
    assert len(r["extracted_claims"]) == 2
    assert r["agent_trace"][0]["metadata"]["truncated"] is True
    assert "NOT reviewed" in r["agent_trace"][0]["thought"]


def test_pipeline_is_deterministic(orch):
    def strip(r):
        r = dict(r, document_id=None)
        r["agent_trace"] = [dict(m, timestamp=None) for m in r["agent_trace"]]
        return r
    assert strip(orch.process_document(README_EXAMPLE)) == strip(orch.process_document(README_EXAMPLE))


def test_no_hard_coded_confidences(orch):
    assert all(m["confidence"] is None for m in orch.process_document(README_EXAMPLE)["agent_trace"])


@pytest.mark.parametrize("bad", ["", "   ", "too short", None, 42])
def test_too_short_or_non_text_input_raises(orch, bad):
    with pytest.raises(DocumentTooShort):
        orch.process_document(bad)


def test_unknown_verifier_rejected():
    with pytest.raises(ValueError):
        MultiAgentOrchestrator(PipelineConfig(verifiers=("overclaim", "astrology")))


def test_disabling_verifiers_removes_their_findings():
    r = MultiAgentOrchestrator(PipelineConfig(verifiers=("overclaim",))).process_document(README_EXAMPLE)
    assert {i["type"] for i in r["potential_issues"]} == {"overclaim_absolute_language"}
    assert MultiAgentOrchestrator(PipelineConfig(verifiers=())).process_document(README_EXAMPLE)["potential_issues"] == []


@pytest.mark.parametrize("text", [
    "🧠🧠🧠 ünïcödé spiking neurons ñ and emoji everywhere in this sentence 🚀.",
    "a " * 5000,
    "No punctuation at all just a very long run of words about neurons " * 50,
    "....... !!!! ???? ...... .... .... ....",
])
def test_odd_inputs_do_not_crash(orch, text):
    assert orch.process_document(text)["final_report"] is not None


def test_benchmark_splits_are_well_formed_and_disjoint():
    def load(name):
        with open(DATA / name, encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    dev, test = load("dev.tsv"), load("test.tsv")
    for rows in (dev, test):
        assert len({r["id"] for r in rows}) == len(rows) == 70
        assert {r["label"] for r in rows} == {"flag", "ok"}
        assert sum(r["label"] == "flag" for r in rows) == 35
    assert not {r["text"] for r in dev} & {r["text"] for r in test}
