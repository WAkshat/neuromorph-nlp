import re

from agents.verifiers import check_constraints, check_overclaim, find_absolutes
from knowledge import neuromorphic_kb as kb
from textutils import analyze, split_sentences


def test_fact_ids_unique_and_cited():
    ids = [f.id for f in kb.FACTS]
    assert len(ids) == len(set(ids))
    assert all(f.citation.strip() and f.text.strip() for f in kb.FACTS)


def test_every_constraint_and_myth_points_at_an_existing_fact():
    for c in kb.ATTRIBUTE_CONSTRAINTS + kb.NUMERIC_CONSTRAINTS + kb.ATTRIBUTION_CONSTRAINTS:
        kb.get_fact(c.fact_id)
    for m in kb.MYTHS:
        kb.get_fact(m.fact_id)


def test_constraint_patterns_compile():
    for c in kb.ATTRIBUTE_CONSTRAINTS:
        re.compile(c.entity), re.compile(c.attribute)
        if c.context:
            re.compile(c.context)


def test_knowledge_base_does_not_contradict_itself():
    """Invariant: the constraint verifier must accept every sentence of every fact.
    (Catches a constraint that is wrong, or too loose, the moment it is added.)"""
    for fact in kb.FACTS:
        for i, sentence in enumerate(split_sentences(fact.text)):
            a = analyze(i, sentence)
            assert check_constraints(a) == [], (fact.id, sentence)


def test_myth_corrections_are_not_flagged():
    for m in kb.MYTHS:
        a = analyze(0, m.correction)
        a.absolutes = find_absolutes(a)
        assert check_constraints(a) + check_overclaim(a) == [], m.id


def test_retrieval_ranks_the_relevant_fact_first():
    assert kb.search_knowledge("How many neurons does TrueNorth have?")[0]["id"] == "hardware.ibm_truenorth"
    assert kb.search_knowledge("surrogate gradient training of SNNs")[0]["id"] == "algorithms_and_training.surrogate_gradient"
    assert kb.search_knowledge("event cameras and frames")[0]["id"] == "sensors.event_camera"


def test_retrieval_returns_nothing_for_off_topic_or_stopword_queries():
    # The original substring matcher returned five facts for both of these.
    assert kb.search_knowledge("the of a is") == []
    assert kb.search_knowledge("rainfall in Paris was heavy yesterday") == []


def test_retrieval_respects_top_k_and_score_order():
    hits = kb.search_knowledge("Loihi on-chip learning plasticity neurons", top_k=4)
    assert len(hits) == 4
    assert [h["score"] for h in hits] == sorted((h["score"] for h in hits), reverse=True)
