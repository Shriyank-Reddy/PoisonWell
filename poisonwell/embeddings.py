"""Turn text into vectors. Two interchangeable backends behind one small interface.

Every embedder returns L2-normalised vectors, so cosine similarity is just a dot product.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class Embedder:
    """Interface: fit() learns from the corpus (if needed), embed() returns a 2-D array."""

    def fit(self, texts: list[str]) -> "Embedder":
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class TfidfEmbedder(Embedder):
    """Bag-of-words vectors weighted by how rare each word is. Fast, offline, no download."""

    def __init__(self):
        # sublinear_tf dampens repeated words (log scaling); the vectors are L2-normalised.
        self.vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)

    def fit(self, texts):
        self.vectorizer.fit(texts)
        return self

    def embed(self, texts):
        return self.vectorizer.transform(texts).toarray()


class MiniLMEmbedder(Embedder):
    """Neural sentence embeddings (needs `pip install sentence-transformers` + a model download)."""

    def __init__(self):
        from sentence_transformers import SentenceTransformer  # imported lazily on purpose
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def embed(self, texts):
        return np.asarray(self.model.encode(texts, normalize_embeddings=True))


def get_embedder(name: str = "tfidf") -> Embedder:
    """Pick a backend by name. Falls back to TF-IDF if MiniLM cannot be loaded."""
    if name == "minilm":
        try:
            return MiniLMEmbedder()
        except Exception as err:  # no package, no network, ... -> keep going with TF-IDF
            print(f"[embeddings] MiniLM unavailable ({type(err).__name__}); using TF-IDF.")
    return TfidfEmbedder()
