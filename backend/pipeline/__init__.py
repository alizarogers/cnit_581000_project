"""
The Stage 1-2 pun pipeline that runs in front of Claude.

Stage 1 (BERT + WordNet): find candidate pivot words and score which of their
WordNet senses the sentence context activates.
Stage 2 (spaCy): keep only senses that fit the word's part of speech and
record the word's grammatical role and the words that trigger each sense.
Stage 3 (Claude) receives this as evidence; see backend/main.py.
"""

from backend.pipeline.stages import PunPipeline, PipelineResult, format_evidence

__all__ = ["PunPipeline", "PipelineResult", "format_evidence"]
