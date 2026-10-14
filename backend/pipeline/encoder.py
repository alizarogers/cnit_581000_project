"""
encoder.py: contextual word vectors from BERT, with an anisotropy correction.

Raw BERT vectors all point in roughly the same direction (anisotropy), so the
cosine between almost any two words comes out high. To make similarities
meaningful we subtract a background mean vector, estimated from a fixed set of
everyday sentences, before computing cosines. Both the raw and the corrected
cosine are reported so the difference can be shown.
"""

from functools import lru_cache
from typing import List, Sequence, Tuple

import numpy as np

# Neutral sentences used only to estimate the mean ("common direction") vector.
BACKGROUND_SENTENCES = [
    "The weather was cold and rainy all week.",
    "She opened the window to let in some fresh air.",
    "Our team finished the report before the deadline.",
    "He drove to the store to buy bread and milk.",
    "The children played in the park after school.",
    "The museum is closed on Mondays.",
    "They watched a movie together on Friday night.",
    "The doctor asked him to come back next month.",
    "A small dog barked at the delivery truck.",
    "The meeting was moved to the large conference room.",
    "She studied chemistry at the state university.",
    "The river flooded the fields after the storm.",
    "Please turn off the lights when you leave.",
    "The price of coffee went up again this year.",
    "He fixed the broken chair with some glue.",
    "The train arrived twenty minutes late.",
    "My grandmother grows tomatoes in her garden.",
    "The company hired three new engineers in March.",
    "We hiked to the top of the hill before sunset.",
    "The teacher wrote the answers on the board.",
    "The band played music until midnight.",
    "Fresh snow covered the roads this morning.",
    "The software update fixed several bugs.",
    "They painted the kitchen a light shade of blue.",
    "The library has thousands of old newspapers.",
    "A strong wind knocked over the fence.",
    "She sent a letter to her cousin in Ohio.",
    "The restaurant serves breakfast all day.",
    "He lost his wallet on the bus yesterday.",
    "The city built a new bridge across the bay.",
]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denom) if denom else 0.0


class BertEncoder:
    """Wraps a Hugging Face BERT model. Vectors = mean of the last N hidden layers."""

    def __init__(self, model_name: str = "bert-base-uncased", last_layers: int = 4):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        try:
            self.model = AutoModel.from_pretrained(model_name, dtype=torch.float32)
        except TypeError:  # older transformers
            self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        self.model.eval()
        self.last_layers = last_layers
        self._mean = None

    # -- core -------------------------------------------------------------
    def _hidden(self, texts: Sequence[str]):
        torch = self._torch
        enc = self.tokenizer(list(texts), return_tensors="pt", padding=True, truncation=True,
                             max_length=128, return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")
        with torch.no_grad():
            out = self.model(**enc, output_hidden_states=True)
        layers = torch.stack(out.hidden_states[-self.last_layers:]).mean(0)  # (batch, seq, dim)
        return layers.numpy(), offsets.numpy(), enc["attention_mask"].numpy()

    def span_vectors(self, text: str, spans: Sequence[Tuple[int, int]]) -> List[np.ndarray]:
        """Vector for each character span: mean of the word pieces that overlap it."""
        hidden, offsets, mask = self._hidden([text])
        h, off, m = hidden[0], offsets[0], mask[0]
        vectors = []
        for start, end in spans:
            idx = [i for i, (s, e) in enumerate(off) if m[i] and e > s and s < end and e > start]
            vectors.append(h[idx].mean(0) if idx else h[m.astype(bool)].mean(0))
        return vectors

    def text_vectors(self, texts: Sequence[str], head_chars: Sequence[int]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """For each text return (mean of all word pieces, vector of the headword at [0, head_chars))."""
        hidden, offsets, mask = self._hidden(texts)
        means, heads = [], []
        for h, off, m, hc in zip(hidden, offsets, mask, head_chars):
            real = [i for i, (s, e) in enumerate(off) if m[i] and e > s]
            head = [i for i in real if off[i][0] < hc]
            means.append(h[real].mean(0))
            heads.append(h[head].mean(0) if head else h[real].mean(0))
        return means, heads

    @property
    def mean_vector(self) -> np.ndarray:
        if self._mean is None:
            hidden, offsets, mask = self._hidden(BACKGROUND_SENTENCES)
            rows = [h[i] for h, off, m in zip(hidden, offsets, mask)
                    for i, (s, e) in enumerate(off) if m[i] and e > s]
            self._mean = np.mean(rows, axis=0)
        return self._mean

    def centered_cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        mu = self.mean_vector
        return cosine(a - mu, b - mu)


@lru_cache(maxsize=2)
def load_encoder(model_name: str) -> BertEncoder:
    return BertEncoder(model_name)
