from textutils import analyze, content_tokens, negated_after, negated_before, split_clauses, split_sentences


def test_sentences_not_split_on_abbreviations():
    text = "Davies et al. introduced Loihi in 2018. It has approx. 130,000 neurons, i.e. far fewer than TrueNorth."
    assert split_sentences(text) == [
        "Davies et al. introduced Loihi in 2018.",
        "It has approx. 130,000 neurons, i.e. far fewer than TrueNorth.",
    ]


def test_sentences_blank_line_is_boundary_and_newline_is_soft_wrap():
    text = "Introduction\n\nSpiking networks use\ndiscrete events. They are sparse."
    assert split_sentences(text) == ["Introduction", "Spiking networks use discrete events.", "They are sparse."]


def test_sentences_decimal_numbers_stay_intact():
    assert split_sentences("The chip draws 0.5 mW. That is low.") == ["The chip draws 0.5 mW.", "That is low."]


def test_sentences_empty_input():
    assert split_sentences("") == [] and split_sentences("  \n\n ") == []


def test_clauses_split_on_contrast():
    assert split_clauses("Loihi supports on-chip learning, whereas TrueNorth does not.") == [
        "Loihi supports on-chip learning", "TrueNorth does not."]
    assert split_clauses("Unlike Loihi, TrueNorth offers no plasticity.") == ["Loihi", "TrueNorth offers no plasticity."]


def test_negation_windows():
    s = "SNNs are not always better"
    assert negated_before(s, s.index("always"), max_gap_tokens=1)
    s = "A chip that is not receiving spikes burns no power"
    assert not negated_before(s, s.index("no power"), max_gap_tokens=1)
    s = "on-chip learning is not supported"
    assert negated_after(s, s.index(" is"))


def test_denial_frame_and_ablation():
    text = "It is a myth that idle chips use zero energy."
    assert analyze(0, text).denial
    off = analyze(0, text, negation_aware=False)
    assert not off.denial and off.clauses == [text]


def test_tokens_drop_stopwords_and_split_hyphens():
    toks = content_tokens("This is the on-chip learning of Loihi")
    assert "is" not in toks and "the" not in toks
    assert {"on-chip", "chip", "learning", "loihi"} <= set(toks)
