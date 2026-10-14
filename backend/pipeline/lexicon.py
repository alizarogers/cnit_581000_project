"""
lexicon.py: WordNet access (via NLTK), downloading the data on first use.
"""

SPACY_TO_WN = {"NOUN": "n", "PROPN": "n", "VERB": "v", "ADJ": "as", "ADV": "r"}


def _download_wordnet() -> None:
    import nltk

    if nltk.download("wordnet", quiet=True):
        return
    # Python from python.org on macOS often lacks root certificates, so the plain
    # download fails with CERTIFICATE_VERIFY_FAILED. Retry using certifi's bundle.
    import ssl

    try:
        import certifi
    except ImportError:
        certifi = None
    original = ssl._create_default_https_context
    try:
        if certifi is not None:
            ssl._create_default_https_context = lambda *a, **k: ssl.create_default_context(cafile=certifi.where())
        if not nltk.download("wordnet", quiet=True):
            raise LookupError(
                "Could not download WordNet. On a Mac, run the 'Install Certificates.command' file in your "
                "Python folder (Applications > Python 3.x), then try again.")
    finally:
        ssl._create_default_https_context = original


def load_wordnet():
    from nltk.corpus import wordnet as wn

    try:
        wn.ensure_loaded()
    except LookupError:
        _download_wordnet()
        from nltk.corpus import wordnet as wn  # noqa: F811 (reload after download)
        wn.ensure_loaded()
    return wn
