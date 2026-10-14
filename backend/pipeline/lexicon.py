"""
lexicon.py: WordNet access (via NLTK), downloading the data on first use.
"""

SPACY_TO_WN = {"NOUN": "n", "PROPN": "n", "VERB": "v", "ADJ": "as", "ADV": "r"}


def load_wordnet():
    import nltk
    from nltk.corpus import wordnet as wn

    try:
        wn.ensure_loaded()
    except LookupError:
        nltk.download("wordnet", quiet=True)
        from nltk.corpus import wordnet as wn  # noqa: F811 (reload after download)
        wn.ensure_loaded()
    return wn
