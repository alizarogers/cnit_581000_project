"""A deterministic stand-in for BERT so pipeline tests run without downloading the model."""
import numpy as np, hashlib
from backend.pipeline.encoder import cosine
class FakeEncoder:
    def _v(self, s):
        r=np.random.default_rng(int(hashlib.md5(s.encode()).hexdigest()[:8],16)); return r.normal(size=16)+3
    def span_vectors(self, text, spans): return [self._v(text[a:b]) for a,b in spans]
    def text_vectors(self, texts, heads): return [self._v(t) for t in texts],[self._v(t[:h]) for t,h in zip(texts,heads)]
    mean_vector=np.full(16,3.0)
    def centered_cosine(self,a,b): return cosine(a-self.mean_vector,b-self.mean_vector)
