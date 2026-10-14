"""
stages.py: Stage 1 (BERT + WordNet sense activation) and Stage 2 (spaCy
structural check) of the pun pipeline.

For each candidate pivot word in a sentence:

  Stage 2a  spaCy gives the word's part of speech, lemma and dependency role.
            Only WordNet senses with the same part of speech are kept
            ("POS integrity"); the rest are counted as filtered out.
  Stage 1   BERT embeds the word in the sentence and each kept sense's WordNet
            definition ("interest: a fixed charge for borrowing money...").
              activation  centered cosine between the word in context and the
                          definition: how strongly the sentence calls up that sense
              trigger     the other word in the sentence closest to the
                          definition (reported to explain the sense, not scored)
            Sense A is the most activated sense. Sense B is the most activated
            sense from a *different* WordNet category (noun.possession vs
            noun.cognition, say), so near-duplicate senses never count as a
            second meaning. double-sense score = Sense B's activation: how
            strongly an unrelated second meaning is still present.
  Stage 2b  spaCy reports how each trigger word relates to the pivot in the
            dependency parse.

The sentence is called a pun (pipeline-only call) when the best candidate's
double-sense score reaches the threshold. Claude (Stage 3) gets all of this as
evidence and makes the final call.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from backend.pipeline.encoder import cosine
from backend.pipeline.lexicon import SPACY_TO_WN, load_wordnet

CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}
# Very general verbs carry dozens of WordNet senses and are almost never the pivot;
# they still count as context words.
LIGHT_VERBS = {"be", "have", "do", "go", "get", "make", "take", "come", "give", "use", "say", "let", "put", "keep", "seem"}
POS_NAMES = {"n": "noun", "v": "verb", "a": "adjective", "s": "adjective", "r": "adverb"}


@dataclass
class SenseScore:
    synset: str
    definition: str
    category: str
    activation: float          # centered cosine, word in context vs. sense headword
    activation_raw: float      # same without the anisotropy correction
    support: float = 0.0       # centered cosine, best other word vs. gloss
    support_raw: float = 0.0
    trigger: Optional[str] = None
    trigger_relation: Optional[str] = None

    @property
    def score(self) -> float:
        return self.activation


@dataclass
class Candidate:
    text: str
    lemma: str
    pos: str
    dep: str
    head: str
    start: int
    end: int
    senses_total: int = 0
    senses_kept: int = 0
    filtered_out: List[str] = field(default_factory=list)
    sense_a: Optional[SenseScore] = None
    sense_b: Optional[SenseScore] = None
    double_score: float = 0.0
    note: str = ""


@dataclass
class PipelineResult:
    sentence: str
    call: str
    threshold: float
    best: Optional[Candidate]
    candidates: List[Candidate]
    models: Dict[str, str]
    seconds: float

    def to_dict(self) -> dict:
        d = asdict(self)
        for c in [d["best"], *d["candidates"]]:
            if c:
                for k in ("sense_a", "sense_b"):
                    if c[k]:
                        c[k]["score"] = c[k]["activation"]
        return d


class PunPipeline:
    def __init__(self, spacy_model: str = "en_core_web_sm", bert_model: str = "bert-base-uncased",
                 threshold: float = 0.30, max_senses: int = 8, nlp=None, encoder=None, wordnet=None):
        self.spacy_model, self.bert_model = spacy_model, bert_model
        self.threshold, self.max_senses = threshold, max_senses
        self._nlp, self._encoder, self._wn = nlp, encoder, wordnet
        self._gloss_cache: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}

    # -- lazy loading (models are large; load once, on first use) ---------
    @property
    def nlp(self):
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load(self.spacy_model)
        return self._nlp

    @property
    def encoder(self):
        if self._encoder is None:
            from backend.pipeline.encoder import load_encoder
            self._encoder = load_encoder(self.bert_model)
        return self._encoder

    @property
    def wn(self):
        if self._wn is None:
            self._wn = load_wordnet()
        return self._wn

    def warm_up(self) -> None:
        self.nlp, self.encoder, self.wn  # noqa: B018 (trigger loading)
        self.encoder.mean_vector

    # -- Stage 2a: candidates and POS integrity ----------------------------
    def find_candidates(self, doc) -> List[Tuple[Candidate, list]]:
        wn = self.wn
        found = []
        for tok in doc:
            if tok.pos_ not in CONTENT_POS or tok.is_stop or not tok.is_alpha or len(tok) < 3:
                continue
            if tok.pos_ == "VERB" and tok.lemma_.lower() in LIGHT_VERBS:
                continue
            wn_pos = SPACY_TO_WN[tok.pos_]
            lemmas = [tok.lemma_.lower()]
            if tok.text.lower() != lemmas[0]:
                lemmas.append(tok.text.lower())
            start, end = tok.idx, tok.idx + len(tok)
            # particle verbs ("branched out"): add the multiword's senses too
            prt = [c for c in tok.children if c.dep_ == "prt"]
            if prt and wn.synsets(f"{lemmas[0]}_{prt[0].text.lower()}", pos="v"):
                lemmas.append(f"{lemmas[0]}_{prt[0].text.lower()}")
                start, end = min(start, prt[0].idx), max(end, prt[0].idx + len(prt[0]))

            all_synsets = []
            for lemma in lemmas:
                for s in wn.synsets(lemma):
                    if s not in all_synsets:
                        all_synsets.append(s)
            kept = [s for s in all_synsets if s.pos() in wn_pos]
            removed = [s for s in all_synsets if s.pos() not in wn_pos]

            cand = Candidate(
                text=doc.text[start:end], lemma=lemmas[-1] if "_" in lemmas[-1] else lemmas[0],
                pos=tok.pos_, dep=tok.dep_, head=tok.head.text, start=start, end=end,
                senses_total=len(all_synsets), senses_kept=len(kept),
                filtered_out=[f"{s.name()} ({POS_NAMES.get(s.pos(), s.pos())}): {s.definition()}" for s in removed[:3]],
            )
            if len(kept) < 2:
                cand.note = "fewer than two WordNet senses with this part of speech"
            found.append((cand, kept[: self.max_senses]))
        return found

    # -- Stage 1: sense activation and support -----------------------------
    def _gloss_vectors(self, synsets, headword: str):
        todo = [s for s in synsets if (s.name(), headword) not in self._gloss_cache]
        if todo:
            texts, heads = [], []
            for s in todo:
                examples = "; ".join(s.examples()[:2])
                texts.append(f"{headword}: {s.definition()}" + (f"; {examples}" if examples else ""))
                heads.append(len(headword))
            means, head_vecs = self.encoder.text_vectors(texts, heads)
            for s, m, h in zip(todo, means, head_vecs):
                self._gloss_cache[(s.name(), headword)] = (m, h)
        return [self._gloss_cache[(s.name(), headword)] for s in synsets]

    def score_candidate(self, cand: Candidate, synsets, target_vec, context: List[Tuple[object, np.ndarray]]) -> None:
        enc = self.encoder
        headword = cand.lemma.replace("_", " ")
        vectors = self._gloss_vectors(synsets, headword)

        def best_trigger(gloss_mean, exclude: Optional[str] = None):
            best = (0.0, 0.0, None)
            for tok, vec in context:
                if exclude and tok.text == exclude:
                    continue
                c = enc.centered_cosine(vec, gloss_mean)
                if best[2] is None or c > best[0]:
                    best = (c, cosine(vec, gloss_mean), tok)
            return best

        scored = []
        for s, (gmean, _ghead) in zip(synsets, vectors):
            sup, sup_raw, trig = best_trigger(gmean)
            scored.append((s, gmean, SenseScore(
                synset=s.name(), definition=s.definition(), category=s.lexname(),
                activation=round(enc.centered_cosine(target_vec, gmean), 3),
                activation_raw=round(cosine(target_vec, gmean), 3),
                support=round(sup, 3), support_raw=round(sup_raw, 3),
                trigger=trig.text if trig is not None else None,
                trigger_relation=self.relation(trig, cand) if trig is not None else None,
            )))
        scored.sort(key=lambda x: x[2].activation, reverse=True)
        s_a, _, a = scored[0]
        cand.sense_a = a

        # Sense B: the strongest sense from a different WordNet category (e.g. noun.possession
        # vs noun.cognition), so near-duplicate senses don't count as a second meaning.
        b_entry = next(((s, g, b) for s, g, b in scored[1:] if s.lexname() != s_a.lexname()), None)
        if b_entry is None:
            cand.note = cand.note or "all its senses are in the same WordNet category, so no clearly different second meaning"
            return
        _, gmean_b, b = b_entry
        if b.trigger == a.trigger and len(context) > 1:  # prefer a different word as its trigger
            sup, sup_raw, trig = best_trigger(gmean_b, exclude=a.trigger)
            b.support, b.support_raw = round(sup, 3), round(sup_raw, 3)
            b.trigger = trig.text if trig is not None else None
            b.trigger_relation = self.relation(trig, cand) if trig is not None else None
        cand.sense_b = b
        cand.double_score = round(b.activation, 3)  # how strongly the unrelated second sense is still active

    # -- Stage 2b: dependency relation between trigger and pivot -----------
    @staticmethod
    def relation(trigger, cand: Candidate) -> str:
        pivot = next((t for t in trigger.doc if t.idx == cand.start or (cand.start <= t.idx < cand.end and t.pos_ in CONTENT_POS)), None)
        if pivot is None:
            return "same sentence"
        if pivot.head == trigger:
            return f"'{trigger.text}' governs '{pivot.text}' ({pivot.dep_})"
        if trigger.head == pivot:
            return f"'{trigger.text}' depends on '{pivot.text}' ({trigger.dep_})"
        ancestors_p = [pivot, *pivot.ancestors]
        ancestors_t = [trigger, *trigger.ancestors]
        common = next((a for a in ancestors_p if a in ancestors_t), None)
        if common is None:
            return "same sentence"
        hops = ancestors_p.index(common) + ancestors_t.index(common)
        clause = "same clause" if hops <= 3 else "different clause"
        return f"{clause}, {hops} dependency hops apart via '{common.text}'"

    # -- full run ----------------------------------------------------------
    def analyze(self, sentence: str) -> PipelineResult:
        t0 = time.time()
        doc = self.nlp(sentence)
        found = self.find_candidates(doc)
        scorable = [(c, s) for c, s in found if len(s) >= 2]

        if scorable:
            context_toks = [t for t in doc if t.pos_ in CONTENT_POS | {"NUM"} and t.is_alpha and not t.is_stop]
            spans = [(c.start, c.end) for c, _ in scorable] + [(t.idx, t.idx + len(t)) for t in context_toks]
            vecs = self.encoder.span_vectors(sentence, spans)
            cand_vecs, ctx_vecs = vecs[: len(scorable)], vecs[len(scorable):]
            for (cand, synsets), tvec in zip(scorable, cand_vecs):
                context = [(t, v) for t, v in zip(context_toks, ctx_vecs) if not (cand.start <= t.idx < cand.end)]
                self.score_candidate(cand, synsets, tvec, context)

        candidates = sorted((c for c, _ in found), key=lambda c: c.double_score, reverse=True)
        best = candidates[0] if candidates and candidates[0].sense_b is not None else None
        call = "Pun" if best and best.double_score >= self.threshold else "Non-pun"
        return PipelineResult(
            sentence=sentence, call=call, threshold=self.threshold, best=best, candidates=candidates,
            models={"parser": self.spacy_model, "encoder": self.bert_model, "lexicon": "WordNet 3.0"},
            seconds=round(time.time() - t0, 2),
        )


def format_evidence(result: PipelineResult, max_candidates: int = 3) -> str:
    """Plain-text summary of Stages 1-2 for Claude (Stage 3)."""
    lines = [
        "Pipeline evidence (Stages 1-2: spaCy parse, BERT embeddings, WordNet senses). "
        "This is automatic and can be wrong; it only knows WordNet senses of words as spelled.",
        f"Stage 1-2 call: {result.call} (best double-sense score "
        f"{result.best.double_score if result.best else 0:.2f}, threshold {result.threshold:.2f})",
    ]
    shown = [c for c in result.candidates if c.sense_a][:max_candidates]
    if not shown:
        lines.append("No candidate word has two or more WordNet senses with a matching part of speech.")
    for c in shown:
        lines.append(f"- Candidate '{c.text}' ({c.pos.lower()}, {c.dep} of '{c.head}'), "
                     f"double-sense score {c.double_score:.2f}; {c.senses_kept} of {c.senses_total} WordNet senses fit its part of speech")
        for label, s in (("Sense A", c.sense_a), ("Sense B", c.sense_b)):
            if s:
                trig = f"triggered by '{s.trigger}' ({s.trigger_relation})" if s.trigger else "no separate trigger word"
                lines.append(f"    {label}: {s.synset} \"{s.definition}\"; {trig}; "
                             f"activation {s.activation:.2f} (raw cosine {s.activation_raw:.2f}), support {s.support:.2f}")
        if c.note:
            lines.append(f"    Note: {c.note}")
    unscored = [c.text for c in result.candidates if not c.sense_a]
    if unscored:
        lines.append(f"Words checked but not scorable (fewer than two senses for their part of speech): {', '.join(unscored)}")
    return "\n".join(lines)
