"""Behavioural tests for the rule verifiers. None of these sentences come from the
benchmark splits, so they do not leak dev/test items into development."""

import pytest

from agents.verifiers import check_constraints, check_overclaim, find_absolutes
from textutils import analyze


def run(text, negation_aware=True):
    a = analyze(0, text, negation_aware)
    a.absolutes = find_absolutes(a)
    return check_overclaim(a) + check_constraints(a)


def types(text, **kw):
    return sorted({i["type"] for i in run(text, **kw)})


# --- overclaim ---------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "This approach always wins against dense baselines.",
    "The method guarantees convergence for any network.",
    "Our chip consumes zero power between events.",
    "The accelerator is completely immune to noise.",
])
def test_overclaim_flagged(text):
    assert types(text) == ["overclaim_absolute_language"]


@pytest.mark.parametrize("text", [
    "This approach does not always win against dense baselines.",
    "Convergence is not guaranteed for deep spiking networks.",
    "It is a misconception that such chips consume zero power.",
    "The membrane potential decays at every timestep of the simulation.",
    "The readout is a fully connected layer with ten outputs.",
    "The board is built entirely from commodity parts.",
])
def test_overclaim_not_flagged(text):
    assert types(text) == []


def test_negation_ablation_restores_the_original_false_positive():
    text = "This approach does not always win against dense baselines."
    assert types(text) == []
    assert types(text, negation_aware=False) == ["overclaim_absolute_language"]


# --- attribute constraints ---------------------------------------------------------

@pytest.mark.parametrize("text", [
    "TrueNorth learns with on-chip plasticity during deployment.",
    "Loihi does not support on-chip learning of any kind.",
    "The SpiNNaker machine is an analog computer.",
    "STDP is a supervised rule driven by a global error.",
    "Event cameras produce full frames like any other sensor.",
    "The spike function is differentiable, which makes training easy.",
    "It is a myth that TrueNorth lacks on-chip learning.",
])
def test_contradictions_flagged(text):
    assert "contradicts_knowledge_base" in types(text)


@pytest.mark.parametrize("text", [
    "TrueNorth does not provide on-chip learning.",
    "On-chip learning is not available on TrueNorth.",
    "Loihi supports on-chip learning through programmable rules.",
    "Unlike TrueNorth, Loihi supports on-chip learning.",
    "Loihi supports on-chip learning, whereas TrueNorth does not.",
    "The spike function is non-differentiable.",
    "STDP is a local unsupervised rule.",
    "Our SNN outperforms the ANN baseline on this dataset.",
    "It is a myth that TrueNorth supports on-chip learning.",
])
def test_consistent_statements_not_flagged(text):
    assert "contradicts_knowledge_base" not in types(text)


def test_issue_carries_cited_evidence():
    issue = run("TrueNorth learns with on-chip plasticity during deployment.")[0]
    assert issue["severity"] == "high"
    assert issue["evidence"][0]["id"] == "hardware.ibm_truenorth"
    assert "Merolla" in issue["evidence"][0]["citation"]


# --- numeric constraints -----------------------------------------------------------

@pytest.mark.parametrize("text,flagged", [
    ("TrueNorth has 1 million neurons.", False),
    ("TrueNorth has 1,000,000 neurons and 256 million synapses.", False),
    ("TrueNorth has 5 million neurons.", True),
    ("TrueNorth has 256 thousand synapses.", True),
    ("Loihi has about 130k neurons.", False),
    ("Loihi 2 neurons can be programmed with custom dynamics.", False),   # the "2" is not a count
    ("One of the neurons on TrueNorth was probed.", False),
    ("TrueNorth has a million neurons while Loihi has 130,000 neurons.", False),
    ("Compared with TrueNorth and Loihi, the new chip has 5 billion neurons.", False),  # two chips: ambiguous
])
def test_numeric(text, flagged):
    assert ("numeric_mismatch" in types(text)) is flagged


# --- attribution -------------------------------------------------------------------

@pytest.mark.parametrize("text,flagged", [
    ("Loihi was developed by Intel Labs.", False),
    ("Loihi was developed by Qualcomm.", True),
    ("Google's TrueNorth chip is widely cited.", True),
    ("IBM TrueNorth and Intel Loihi are the best-known chips.", False),
    ("Davies et al. introduced Loihi.", False),
    ("The team developed Loihi over several years.", False),
    ("Yann LeCun introduced the SpiNNaker platform.", True),
])
def test_attribution(text, flagged):
    assert ("misattribution" in types(text)) is flagged
