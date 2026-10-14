"""
punbot.py: the full three-stage system behind the chat window and the evaluation.

  Stage 1-2  PunPipeline (spaCy + BERT + WordNet) analyzes the sentence
  Stage 3    Claude receives the sentence plus the pipeline evidence, makes the
             final call and handles the conversation

Questions and follow-ups go straight to Claude; only sentences that look like a
candidate pun are run through Stages 1-2.
"""

from typing import Dict, List, Optional

from backend.prompts import SYSTEM_PROMPT

EVIDENCE_DIVIDER = "\n\n---\n"
QUESTION_STARTS = ("what", "why", "how", "who", "when", "where", "which", "is ", "are ", "can ", "could ",
                   "does ", "do ", "would ", "should ", "explain", "tell me", "give me", "write", "make ")


def looks_like_candidate(message: str, history: List[Dict[str, str]]) -> bool:
    """Heuristic: a short statement (not a question or request) is a sentence to analyze."""
    text = message.strip()
    lowered = text.lower()
    if not text or text.endswith("?") or "\n" in text:
        return False
    if lowered.startswith(QUESTION_STARTS):
        return False
    return 3 <= len(text.split()) <= 40


class PunBot:
    def __init__(self, llm, pipeline=None):
        self.llm = llm
        self.pipeline = pipeline

    @property
    def mode(self) -> str:
        return "full" if self.pipeline is not None else "llm_only"

    def respond(self, message: str, history: List[Dict[str, str]], analyze: Optional[bool] = None,
                temperature: Optional[float] = None) -> dict:
        result = None
        sent_message = message
        if self.pipeline is not None and (analyze if analyze is not None else looks_like_candidate(message, history)):
            from backend.pipeline import format_evidence
            result = self.pipeline.analyze(message)
            sent_message = message + EVIDENCE_DIVIDER + format_evidence(result)

        answer = self.llm.generate(system_prompt=SYSTEM_PROMPT, history=history, user_message=sent_message, temperature=temperature)
        return {
            "answer": answer,
            "sent_message": sent_message,
            "pipeline": result.to_dict() if result is not None else None,
            "mode": self.mode,
        }


def build_pipeline(cfg, force: bool = False) -> "Optional[object]":
    """Create and warm up the Stage 1-2 pipeline, or return None if PIPELINE_MODE turns it off."""
    if cfg.PIPELINE_MODE != "full" and not force:
        return None
    from backend.pipeline import PunPipeline
    pipeline = PunPipeline(spacy_model=cfg.SPACY_MODEL, bert_model=cfg.BERT_MODEL,
                           threshold=cfg.PUN_THRESHOLD, max_senses=cfg.MAX_SENSES)
    pipeline.warm_up()
    return pipeline
