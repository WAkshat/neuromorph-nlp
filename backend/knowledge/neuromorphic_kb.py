"""Curated knowledge base for neuromorphic-computing claims.

Three kinds of records, all plain data:

* FACTS        - short statements with a citation. Retrieved (BM25) as evidence.
* MYTHS        - common false claims, each stated as the false assertion itself plus
                 its correction. Used by the optional NLI verifier and for reporting.
* CONSTRAINTS  - machine-checkable consequences of the facts (entity/attribute
                 polarity, numeric values, attribution). Used by the Checker.

Adding knowledge means adding records here; no agent code changes.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass

from textutils import content_tokens


@dataclass(frozen=True)
class Fact:
    id: str
    text: str
    citation: str


@dataclass(frozen=True)
class Myth:
    id: str
    claim: str          # the false assertion, stated plainly
    correction: str
    fact_id: str


FACTS: list[Fact] = [
    Fact("core_concepts.neuromorphic_computing",
         "Neuromorphic computing is a brain-inspired approach that uses spiking neural networks, "
         "event-driven processing and specialised hardware, aiming at energy-efficient real-time "
         "processing for sensory and cognitive tasks.",
         "Mead, Proc. IEEE 78(10), 1990; Schuman et al., Nat. Comput. Sci. 2, 2022"),
    Fact("core_concepts.spiking_neural_networks",
         "Spiking neural networks (SNNs) communicate with discrete spikes rather than continuous "
         "activations, and spike timing can carry information. Common neuron models are leaky "
         "integrate-and-fire (LIF), adaptive exponential (AdEx) and Hodgkin-Huxley.",
         "Maass, Neural Networks 10(9), 1997; Gerstner & Kistler, Spiking Neuron Models, 2002"),
    Fact("core_concepts.event_driven",
         "Event-driven processing computes only when spikes arrive, so sparse activity lowers "
         "dynamic power compared with always-on processing. Static power and infrastructure "
         "overhead remain.",
         "Davies et al., Proc. IEEE 109(5), 2021"),
    Fact("core_concepts.synaptic_plasticity",
         "Synaptic plasticity mechanisms such as spike-timing-dependent plasticity (STDP) "
         "strengthen or weaken synapses depending on the relative timing of pre- and "
         "postsynaptic spikes.",
         "Bi & Poo, J. Neurosci. 18(24), 1998"),
    Fact("hardware.intel_loihi",
         "Intel Loihi (2018) is an asynchronous digital neuromorphic research chip in a 14 nm "
         "process with 128 neuromorphic cores, about 130,000 neurons and 130 million synapses. "
         "It supports on-chip learning with programmable synaptic plasticity rules.",
         "Davies et al., IEEE Micro 38(1), 2018"),
    Fact("hardware.intel_loihi_2",
         "Loihi 2 (Intel, 2021) supports up to one million neurons per chip, graded spikes and "
         "programmable neuron models, and is programmed through the open-source Lava framework.",
         "Orchard et al., IEEE SiPS, 2021"),
    Fact("hardware.ibm_truenorth",
         "IBM TrueNorth (2014) is a digital neuromorphic chip with one million neurons and "
         "256 million synapses in 4096 neurosynaptic cores, built for very low power inference. "
         "It has no on-chip learning: networks are trained offline and then deployed.",
         "Merolla et al., Science 345(6197), 2014"),
    Fact("hardware.brainscales",
         "BrainScaleS (Heidelberg University) is a mixed-signal analog neuromorphic system that "
         "runs in accelerated time, roughly a thousand to ten thousand times faster than "
         "biological real time.",
         "Schemmel et al., IEEE ISCAS, 2010; Pehle et al., Front. Neurosci. 16, 2022"),
    Fact("hardware.spinnaker",
         "SpiNNaker (University of Manchester) is a digital, massively parallel neuromorphic "
         "platform built from ARM processor cores, designed for real-time simulation of large "
         "spiking neural networks.",
         "Furber et al., Proc. IEEE 102(5), 2014"),
    Fact("hardware.memristors",
         "Memristors are two-terminal devices whose resistance depends on the history of the "
         "current through them. Most retain their state without power (non-volatile), although "
         "volatile types exist. They are studied as compact analog synapses, "
         "with device variability and limited endurance as open challenges.",
         "Chua, IEEE Trans. Circuit Theory 18(5), 1971; Strukov et al., Nature 453, 2008"),
    Fact("hardware.online_learning_varies",
         "Support for on-chip (online) learning differs between neuromorphic platforms: "
         "Loihi supports it, TrueNorth does not.",
         "Davies et al., IEEE Micro 38(1), 2018; Merolla et al., Science 345(6197), 2014"),
    Fact("sensors.event_camera",
         "Dynamic vision sensors (event cameras) do not output frames. Each pixel "
         "asynchronously reports brightness changes with microsecond-scale timing.",
         "Lichtsteiner, Posch & Delbruck, IEEE JSSC 43(2), 2008; Gallego et al., IEEE TPAMI 44(1), 2022"),
    Fact("algorithms_and_training.surrogate_gradient",
         "The spike function is a non-differentiable step whose derivative is zero almost "
         "everywhere, so deep SNNs are trained with surrogate gradients (for example SuperSpike "
         "or SLAYER) that substitute a smooth derivative during backpropagation.",
         "Neftci, Mostafa & Zenke, IEEE Signal Process. Mag. 36(6), 2019"),
    Fact("algorithms_and_training.conversion",
         "ANN-to-SNN conversion maps a trained rate-based network onto spiking neurons. It can "
         "reach competitive accuracy but usually needs long simulation windows and careful "
         "threshold or weight scaling; short windows cost accuracy.",
         "Rueckauer et al., Front. Neurosci. 11, 2017"),
    Fact("algorithms_and_training.direct_training",
         "Direct training of SNNs with surrogate gradients can exploit temporal dynamics better "
         "than conversion, but is often more computationally expensive than training an ANN "
         "because gradients are propagated through time.",
         "Neftci, Mostafa & Zenke, IEEE Signal Process. Mag. 36(6), 2019"),
    Fact("algorithms_and_training.stdp",
         "STDP is a local, unsupervised learning rule: a presynaptic spike shortly before a "
         "postsynaptic spike strengthens the synapse, and the reverse order weakens it. "
         "Variants include triplet STDP and reward-modulated STDP.",
         "Bi & Poo, J. Neurosci. 18(24), 1998"),
    Fact("neurons.lif",
         "The leaky integrate-and-fire (LIF) neuron is a simplified point model: it integrates "
         "input into a leaky membrane potential and emits a spike when a threshold is crossed. "
         "It does not model ion channels.",
         "Gerstner & Kistler, Spiking Neuron Models, 2002"),
    Fact("neurons.hodgkin_huxley",
         "The Hodgkin-Huxley model describes action potentials with voltage-dependent sodium "
         "and potassium conductances. It is far more detailed and computationally expensive "
         "than the leaky integrate-and-fire model.",
         "Hodgkin & Huxley, J. Physiol. 117(4), 1952"),
    Fact("coding_schemes.rate_coding",
         "In rate coding, information is encoded in the average number of spikes over a time window.",
         "Gerstner & Kistler, Spiking Neuron Models, 2002"),
    Fact("coding_schemes.temporal_coding",
         "In temporal coding, information is carried by the precise timing of spikes, as in "
         "latency, rank-order or phase coding. It is a different scheme from rate coding.",
         "Gerstner & Kistler, Spiking Neuron Models, 2002"),
    Fact("coding_schemes.population_coding",
         "In population coding, information is distributed across the activity of many neurons.",
         "Gerstner & Kistler, Spiking Neuron Models, 2002"),
    Fact("performance.accuracy",
         "SNNs are not generally more accurate than ANNs. On static image benchmarks they "
         "typically match or trail ANNs unless carefully trained.",
         "Deng et al., Neural Networks 121, 2020; Roy, Jaiswal & Panda, Nature 575, 2019"),
    Fact("performance.energy",
         "Neuromorphic systems can be orders of magnitude more energy efficient than CPUs and "
         "GPUs for certain sparse, event-driven workloads. The advantage depends on the "
         "workload and is not guaranteed for dense workloads.",
         "Davies et al., Proc. IEEE 109(5), 2021"),
    Fact("performance.gpu",
         "Neuromorphic hardware complements GPUs rather than replacing them: GPUs remain better "
         "suited to dense matrix workloads and large-scale training.",
         "Davies et al., Proc. IEEE 109(5), 2021; Roy, Jaiswal & Panda, Nature 575, 2019"),
    Fact("performance.idle_energy",
         "Idle neuromorphic hardware still consumes energy: static leakage and the supporting "
         "digital or analog infrastructure draw power even when no spikes occur.",
         "Davies et al., Proc. IEEE 109(5), 2021"),
    Fact("performance.biological_realism",
         "Most practical SNNs use highly simplified neuron models such as LIF, not biophysically "
         "detailed models, so they are not biologically realistic simulations of real neurons.",
         "Gerstner & Kistler, Spiking Neuron Models, 2002"),
    Fact("key_papers_and_facts.mead",
         "Carver Mead introduced the term neuromorphic in the late 1980s for analog circuits "
         "that mimic neural systems.",
         "Mead, Proc. IEEE 78(10), 1990"),
    Fact("key_papers_and_facts.maass_1997",
         "Wolfgang Maass characterised networks of spiking neurons as the third generation of "
         "neural network models in 1997.",
         "Maass, Neural Networks 10(9), 1997"),
    Fact("key_papers_and_facts.davies_2018",
         "Mike Davies and colleagues at Intel introduced the Loihi chip in IEEE Micro in 2018.",
         "Davies et al., IEEE Micro 38(1), 2018"),
]

MYTHS: list[Myth] = [
    Myth("myth.accuracy", "Spiking neural networks are more accurate than conventional artificial neural networks.",
         "SNNs often lag ANNs in accuracy on static benchmarks unless carefully trained.", "performance.accuracy"),
    Myth("myth.replace_gpu", "Neuromorphic chips replace GPUs for deep learning.",
         "They excel at sparse, event-driven, low-power workloads, not dense matrix multiplies.", "performance.gpu"),
    Myth("myth.coding", "Rate coding and temporal coding are the same thing.",
         "Rate coding uses the average spike rate; temporal coding uses precise spike times or order.",
         "coding_schemes.temporal_coding"),
    Myth("myth.online_learning", "All neuromorphic hardware supports on-chip learning.",
         "TrueNorth does not support on-chip learning; Loihi does.", "hardware.online_learning_varies"),
    Myth("myth.realism", "Practical spiking neural networks are biologically realistic models of real neurons.",
         "Most practical SNNs use highly simplified neuron models, not Hodgkin-Huxley dynamics.",
         "performance.biological_realism"),
    Myth("myth.idle_energy", "Idle neuromorphic hardware consumes no energy.",
         "Static leakage and the supporting digital or analog fabric still draw power.", "performance.idle_energy"),
    Myth("myth.energy", "Neuromorphic hardware is more energy efficient than GPUs for every workload.",
         "The energy advantage holds for sparse, event-driven workloads and is not guaranteed otherwise.",
         "performance.energy"),
    Myth("myth.conversion", "Converting an ANN to an SNN costs no accuracy even with very few timesteps.",
         "Conversion usually needs long simulation windows; short windows cost accuracy.",
         "algorithms_and_training.conversion"),
]

_FACT_BY_ID = {f.id: f for f in FACTS}


def get_fact(fact_id: str) -> Fact:
    return _FACT_BY_ID[fact_id]


def get_all_mistakes() -> list[str]:
    """Kept for API compatibility: human-readable myth descriptions."""
    return [f"{m.claim} - {m.correction}" for m in MYTHS]


# --- machine-checkable constraints -------------------------------------------------

_ONCHIP_LEARNING = (
    r"\b(?:on[- ]chip|on[- ]device|online|in[- ]situ)\s+(?:\w+\s+)?(?:learning|training|plasticity|adaptation)\b"
    r"|\b(?:synaptic\s+)?plasticity\b"
    r"|\b(?:learn\w*|train\w*|adapt\w*)\b[^.;]*?\b(?:on|into|in)\s+the\s+(?:chip|device)\b"
    r"|\badapts?\s+its\s+(?:synaptic\s+)?weights\b"
    r"|\blearning\s+on\s+(?:the\s+)?(?:chip|device)\b"
)
_SNN = r"\bsnns?\b|\bspiking\b"
_NEUROMORPHIC_CTX = r"neuromorphic|spiking|\bsnns?\b"


@dataclass(frozen=True)
class AttributeConstraint:
    """`entity` has (holds=True) or does not have (holds=False) the property `attribute`.

    A claim violates it when a clause mentions the entity and asserts the opposite.
    `context`, if set, must match somewhere in the sentence. `generic` constraints are
    statements about a whole class ("SNNs") and are skipped for hedged or first-person
    empirical claims ("our SNN outperforms ...", "can outperform").
    """
    id: str
    entity: str
    attribute: str
    holds: bool
    fact_id: str
    message: str
    context: str | None = None
    generic: bool = False


ATTRIBUTE_CONSTRAINTS: list[AttributeConstraint] = [
    AttributeConstraint("truenorth.no_onchip_learning", r"\btruenorth\b", _ONCHIP_LEARNING, False,
                        "hardware.ibm_truenorth",
                        "TrueNorth has no on-chip learning; its networks are trained offline and deployed."),
    AttributeConstraint("loihi.onchip_learning", r"\bloihi\b", _ONCHIP_LEARNING, True,
                        "hardware.intel_loihi",
                        "Loihi does support on-chip learning with programmable plasticity rules."),
    AttributeConstraint("spinnaker.not_analog", r"\bspinnaker\b", r"\banalog(?:ue)?\b|\bmixed[- ]signal\b", False,
                        "hardware.spinnaker", "SpiNNaker is a digital platform built from ARM cores, not an analog system."),
    AttributeConstraint("spinnaker.digital", r"\bspinnaker\b", r"\bdigital\b", True,
                        "hardware.spinnaker", "SpiNNaker is a digital platform."),
    AttributeConstraint("brainscales.not_digital_realtime", r"\bbrainscales\b",
                        r"\b(?:fully|purely|entirely)\s+digital\b|\bslower\b"
                        r"|\b(?:runs?|operat\w+|emulat\w+)\s+(?:\w+\s+)?in\s+(?:biological\s+)?real[- ]time\b",
                        False, "hardware.brainscales",
                        "BrainScaleS is a mixed-signal analog system that runs faster than biological real time."),
    AttributeConstraint("brainscales.analog_accelerated", r"\bbrainscales\b",
                        r"\banalog(?:ue)?\b|\bmixed[- ]signal\b|\baccelerated\b", True,
                        "hardware.brainscales", "BrainScaleS is a mixed-signal, accelerated analog system."),
    AttributeConstraint("spike_fn.not_differentiable",
                        r"\b(?:spike|spiking|heaviside|step|threshold)\s+(?:step\s+)?(?:function|non-?linearity|activation)\b",
                        r"(?<!non-)(?<!non)\bdifferentiable\b", False,
                        "algorithms_and_training.surrogate_gradient",
                        "The spike function is a non-differentiable step; that is why surrogate gradients are needed."),
    AttributeConstraint("stdp.not_supervised", r"\bstdp\b|\bspike[- ]timing[- ]dependent\s+plasticity\b",
                        r"\bis\s+an?\s+(?:\w+\s+){0,2}?supervised\b|\bglobal\s+(?:error|supervis\w+|learning)\b|\bbackpropagat\w+",
                        False, "algorithms_and_training.stdp",
                        "STDP is a local, unsupervised rule; it does not use a global error signal or backpropagation."),
    AttributeConstraint("stdp.local", r"\bstdp\b|\bspike[- ]timing[- ]dependent\s+plasticity\b",
                        r"\blocal\b|\bunsupervised\b", True,
                        "algorithms_and_training.stdp", "STDP is a local, unsupervised learning rule."),
    AttributeConstraint("stdp.direction", r"\bstdp\b|\bspike[- ]timing[- ]dependent\s+plasticity\b",
                        r"\bpre\w*(?:-synaptic)?\b[^.;]*?\bbefore\b[^.;]*?\bpost\w*[^.;]*?\b(?:weaken\w*|depress\w*|decreas\w*|reduc\w*)"
                        r"|\bpost\w*(?:-synaptic)?\b[^.;]*?\bbefore\b[^.;]*?\bpre\w*[^.;]*?\b(?:strengthen\w*|potentiat\w*|increas\w*)",
                        False, "algorithms_and_training.stdp",
                        "In classical STDP, pre-before-post strengthens the synapse and post-before-pre weakens it."),
    AttributeConstraint("lif.no_ion_channels", r"\bleaky\s+integrate[- ]and[- ]fire\b|\blif\b",
                        r"\bion[- ]channels?\b|\bsodium\b|\bpotassium\b|\bbiophysic\w+"
                        r"|\bbiologically\s+(?:realistic|accurate|faithful|detailed|complete)\b",
                        False, "neurons.lif",
                        "LIF is a simplified point-neuron model; it does not model ion channels (that is Hodgkin-Huxley)."),
    AttributeConstraint("hh.not_simple", r"\bhodgkin[-–\s]+huxley\b",
                        r"\b(?:simpler|cheaper|faster)\b|\bless\s+(?:expensive|costly|complex|detailed)\b", False,
                        "neurons.hodgkin_huxley",
                        "Hodgkin-Huxley is more detailed and more expensive to simulate than LIF, not simpler."),
    AttributeConstraint("dvs.no_frames",
                        r"\bevent(?:[- ]based)?\s+(?:vision\s+)?(?:cameras?|sensors?)\b|\bdynamic\s+vision\s+sensors?\b|\bdvs\b",
                        r"\b(?:full|complete|whole|entire)\s+(?:image\s+)?frames?\b|\bfixed\s+(?:frame\s+)?rate\b"
                        r"|\bsynchronous(?:ly)?\b|\bframes\s+per\s+second\b|\bfps\b",
                        False, "sensors.event_camera",
                        "Event cameras emit asynchronous per-pixel events; they do not output frames at a fixed rate."),
    AttributeConstraint("rate_coding.not_timing", r"\brate[- ]cod(?:ing|es?)\b",
                        r"\b(?:precise|exact)\s+(?:spike\s+)?tim\w+|\btiming\s+of\s+(?:individual\s+)?spikes\b", False,
                        "coding_schemes.rate_coding",
                        "Rate coding uses the average firing rate; precise spike timing is temporal coding."),
    AttributeConstraint("temporal_coding.not_rate", r"\b(?:temporal|latency)[- ]cod(?:ing|es?)\b",
                        r"\b(?:mean|average)\s+(?:firing\s+|spike\s+)?rates?\b", False,
                        "coding_schemes.temporal_coding",
                        "Temporal coding uses precise spike timing; the mean firing rate is rate coding."),
    AttributeConstraint("coding.not_same", r"\brate[- ]cod(?:ing|es?)\b",
                        r"\bsame\b|\bidentical\b|\bequivalent\b|\binterchangeable\b|\bno\s+(?:\w+\s+)?difference\b", False,
                        "coding_schemes.temporal_coding",
                        "Rate coding and temporal coding are different encoding schemes.",
                        context=r"\btemporal[- ]cod(?:ing|es?)\b"),
    # Deliberately says nothing about volatility: volatile (e.g. Mott) memristors exist.
    AttributeConstraint("memristor.two_terminal", r"\bmemrist\w+", r"\bthree[- ]terminal\b", False,
                        "hardware.memristors", "Memristors are two-terminal devices."),
    AttributeConstraint("gpu.not_replaced", r"\bgpus?\b|\bgraphics\s+process\w+",
                        r"\breplac\w+|\bobsolete\b|\bsupersed\w+|\bno\s+longer\s+(?:any\s+)?(?:need|reason)\b", False,
                        "performance.gpu",
                        "Neuromorphic hardware complements GPUs for sparse, event-driven tasks; it does not replace them.",
                        context=_NEUROMORPHIC_CTX),
    AttributeConstraint("snn.not_more_accurate", _SNN,
                        r"\bmore\s+accurate\b|\b(?:higher|better|superior)\s+accuracy\b|\boutperform\w*|\bbeat\w*", False,
                        "performance.accuracy",
                        "SNNs are not generally more accurate than ANNs; they often trade accuracy for efficiency.",
                        context=r"\banns?\b|\bartificial\b|\bconventional\b|\bstandard\b|\bdnns?\b|\bcnns?\b|\bdeep\s+(?:neural\s+)?networks?\b|\baccura\w+",
                        generic=True),
    AttributeConstraint("snn.not_biologically_realistic", _SNN,
                        r"\bbiologically\s+(?:realistic|accurate|faithful|complete|exact)\b"
                        r"|\b(?:complete|full|exact|faithful\w*)\s+(?:\w+\s+){0,2}?(?:biolog\w+|biophysic\w+)", False,
                        "performance.biological_realism",
                        "Practical SNNs use highly simplified neuron models, not biologically complete ones.",
                        generic=True),
    AttributeConstraint("dense.no_energy_advantage", r"\bdense\b",
                        r"\benergy\s+savings?\b|\bmore\s+(?:energy[- ])?efficient\b|\b(?:less|lower)\s+(?:energy|power)\b", False,
                        "performance.energy",
                        "The neuromorphic energy advantage applies to sparse, event-driven workloads, not dense ones.",
                        context=_NEUROMORPHIC_CTX),
]


@dataclass(frozen=True)
class NumericConstraint:
    """Counts of `noun` reported for `entity` must be within 2x of an accepted value."""
    id: str
    entity: str
    noun: str
    accepted: tuple[float, ...]
    fact_id: str
    exclude: str | None = None      # skip when this also matches (e.g. 'loihi 2' for Loihi 1 numbers)


NUMERIC_CONSTRAINTS: list[NumericConstraint] = [
    NumericConstraint("truenorth.neurons", r"\btruenorth\b", "neurons", (1e6,), "hardware.ibm_truenorth"),
    NumericConstraint("truenorth.synapses", r"\btruenorth\b", "synapses", (256e6,), "hardware.ibm_truenorth"),
    NumericConstraint("truenorth.cores", r"\btruenorth\b", "cores", (4096,), "hardware.ibm_truenorth"),
    # Plain "Loihi" may mean either generation, so both generations' figures are accepted.
    NumericConstraint("loihi.neurons", r"\bloihi\b", "neurons", (130e3, 1e6), "hardware.intel_loihi"),
    NumericConstraint("loihi.synapses", r"\bloihi\b", "synapses", (130e6, 120e6), "hardware.intel_loihi"),
    NumericConstraint("loihi.cores", r"\bloihi\b", "cores", (128,), "hardware.intel_loihi", exclude=r"\bloihi\s*2\b"),
]


@dataclass(frozen=True)
class AttributionConstraint:
    """`entity` was created by / belongs to someone matching `accepted`."""
    id: str
    entity: str
    accepted: str
    fact_id: str
    owner: str


ATTRIBUTION_CONSTRAINTS: list[AttributionConstraint] = [
    AttributionConstraint("loihi.intel", r"Loihi", r"intel|davies|orchard", "hardware.intel_loihi", "Intel"),
    AttributionConstraint("truenorth.ibm", r"TrueNorth", r"ibm|merolla|modha", "hardware.ibm_truenorth", "IBM"),
    AttributionConstraint("spinnaker.manchester", r"SpiNNaker", r"manchester|furber", "hardware.spinnaker",
                          "the University of Manchester"),
    AttributionConstraint("brainscales.heidelberg", r"BrainScaleS", r"heidelberg|schemmel|meier", "hardware.brainscales",
                          "Heidelberg University"),
    AttributionConstraint("neuromorphic_term.mead", r"term\s+[\"'“]?neuromorphic(?:\s+\w+)?", r"mead",
                          "key_papers_and_facts.mead", "Carver Mead"),
]

# Organisations recognised in "<Org> <chip>" / "<Org>'s <chip>" prefixes.
KNOWN_ORGS = r"Intel|IBM|Manchester|Heidelberg|Stanford|Google|Qualcomm|Samsung|BrainChip|MIT|Nvidia|NVIDIA"

# --- retrieval ---------------------------------------------------------------------

class BM25Index:
    """Okapi BM25 (Robertson & Zaragoza, 2009) over the FACTS. ~30 facts, so a dense
    scan per query is fine. ponytail: O(|KB|) per query; switch to an inverted index
    if the KB grows past a few thousand records."""

    def __init__(self, facts: list[Fact], k1: float = 1.5, b: float = 0.75):
        self.facts = facts
        self.k1, self.b = k1, b
        # The id is indexed too: 'hardware.ibm_truenorth' contributes 'hardware', 'ibm', 'truenorth'.
        self.docs = [Counter(content_tokens(f"{f.id.replace('.', ' ').replace('_', ' ')} {f.text}")) for f in facts]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = sum(self.lengths) / len(self.lengths)
        n = len(facts)
        df = Counter(term for d in self.docs for term in d)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        terms = set(content_tokens(query))
        scored = []
        for fact, doc, length in zip(self.facts, self.docs, self.lengths):
            matched = [t for t in terms if t in doc]
            if not matched:
                continue
            norm = self.k1 * (1 - self.b + self.b * length / self.avg_len)
            score = sum(self.idf[t] * doc[t] * (self.k1 + 1) / (doc[t] + norm) for t in matched)
            scored.append((score, len(matched), fact))
        scored.sort(key=lambda x: -x[0])
        return [
            {
                "id": f.id, "source": f.id, "content": f.text, "citation": f.citation,
                "score": round(score, 3), "matched_terms": n_matched,
                # Same meaning as the original field: 'high' when two or more query terms match.
                "relevance": "high" if n_matched >= 2 else "medium",
            }
            for score, n_matched, f in scored[:top_k]
        ]


_INDEX = BM25Index(FACTS)


def search_knowledge(query: str, top_k: int = 3) -> list[dict]:
    return _INDEX.search(query, top_k)
