"""
Stage 1-2 pipeline tests. Use real spaCy and WordNet when installed, with a
stand-in encoder instead of BERT so no model download is needed.
"""

import pytest

spacy = pytest.importorskip("spacy")
pytest.importorskip("nltk")

from backend.pipeline import PunPipeline, format_evidence  # noqa: E402
from backend.punbot import EVIDENCE_DIVIDER, PunBot, looks_like_candidate  # noqa: E402
from tests.fake_encoder import FakeEncoder  # noqa: E402


@pytest.fixture(scope="module")
def pipeline():
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip("en_core_web_sm not installed")
    try:
        from backend.pipeline.lexicon import load_wordnet
        wn = load_wordnet()
    except Exception:
        pytest.skip("WordNet data not available")
    return PunPipeline(nlp=nlp, encoder=FakeEncoder(), wordnet=wn, threshold=0.3)


def test_finds_pivot_candidate_with_pos_filter(pipeline):
    doc = pipeline.nlp("I used to be a banker, but I lost interest.")
    found = {c.text: (c, synsets) for c, synsets in pipeline.find_candidates(doc)}
    cand, synsets = found["interest"]
    assert cand.pos == "NOUN" and cand.dep == "dobj" and cand.head == "lost"
    assert all(s.pos() == "n" for s in synsets)          # Stage 2: noun senses only
    assert cand.senses_total > cand.senses_kept           # verb senses were filtered out
    assert cand.filtered_out and "(verb)" in cand.filtered_out[0]


def test_light_verbs_are_not_candidates(pipeline):
    doc = pipeline.nlp("I went to the bank.")
    texts = [c.text for c, _ in pipeline.find_candidates(doc)]
    assert "went" not in texts and "bank" in texts


def test_particle_verb_adds_multiword_senses(pipeline):
    doc = pipeline.nlp("The gardener got promoted because his work really branched out.")
    cand, synsets = next((c, s) for c, s in pipeline.find_candidates(doc) if c.text.startswith("branched"))
    assert cand.text == "branched out"
    assert "diversify.v.03" in {s.name() for s in synsets}   # branch_out sense joins the branch senses


def test_two_senses_need_different_triggers(pipeline):
    result = pipeline.analyze("I went to the bank.")
    bank = next(c for c in result.candidates if c.text == "bank")
    assert bank.double_score == 0.0                       # only one context word ("went")
    assert result.call == "Non-pun"


def test_analyze_returns_serializable_result(pipeline):
    import json
    result = pipeline.analyze("I used to be a banker, but I lost interest.")
    assert result.call in ("Pun", "Non-pun")
    data = result.to_dict()
    json.dumps(data)
    best = data["best"]
    assert best and best["sense_a"]["trigger"] != best["sense_b"]["trigger"]
    text = format_evidence(result)
    assert text.startswith("Pipeline evidence") and "Sense A" in text and "Sense B" in text


def test_sentence_without_candidates(pipeline):
    result = pipeline.analyze("It is what it is.")
    assert result.call == "Non-pun" and result.best is None
    assert "No candidate word" in format_evidence(result) or "not scorable" in format_evidence(result)


@pytest.mark.parametrize("text,expected", [
    ("I used to be a banker, but I lost interest.", True),
    ("Why is this funny?", False),
    ("What are the two meanings of interest here", False),
    ("Explain this joke like I am ten years old.", False),
    ("ok", False),
])
def test_looks_like_candidate(text, expected):
    assert looks_like_candidate(text, []) is expected


class RecordingLLM:
    def __init__(self):
        self.messages = []

    def generate(self, system_prompt, history, user_message, temperature=None):
        self.messages.append(user_message)
        return "**Classification**: Pun"


def test_punbot_sends_evidence_only_for_candidate_sentences(pipeline):
    llm = RecordingLLM()
    bot = PunBot(llm, pipeline)
    out = bot.respond("I used to be a banker, but I lost interest.", [])
    assert out["pipeline"] and EVIDENCE_DIVIDER in out["sent_message"]
    assert "Pipeline evidence" in llm.messages[-1]
    out = bot.respond("Why is it funny?", [])
    assert out["pipeline"] is None and llm.messages[-1] == "Why is it funny?"


def test_punbot_llm_only_mode():
    llm = RecordingLLM()
    bot = PunBot(llm, None)
    out = bot.respond("I used to be a banker, but I lost interest.", [])
    assert bot.mode == "llm_only" and out["pipeline"] is None
    assert llm.messages[-1] == "I used to be a banker, but I lost interest."
