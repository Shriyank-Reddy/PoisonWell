"""A tiny in-memory vector store. Swap in FAISS/Chroma later by keeping add()/search()."""
from dataclasses import dataclass

import numpy as np

from poisonwell.embeddings import Embedder


@dataclass
class Document:
    """One document = one chunk (our docs are short, so we do not split them)."""
    id: str
    title: str
    category: str          # "hr", "it" or "product" (the category the doc *claims*)
    source: str            # "internal-verified" or "user-upload"
    date: str
    text: str
    # Ground-truth labels, used ONLY by the evaluator. Defenses must never read these.
    attack: str | None = None        # "injection" | "misinformation" | "hijack" | None
    canary: str | None = None        # e.g. "CANARY-7F3A"
    target_claim: str | None = None  # the false/harmful phrase the attacker wants in the answer


class VectorStore:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.docs: list[Document] = []
        self.vectors = np.zeros((0, 0))

    @classmethod
    def build(cls, docs: list[Document], embedder: Embedder) -> "VectorStore":
        embedder.fit([d.text for d in docs])
        store = cls(embedder)
        store.add(docs)
        return store

    def add(self, docs: list[Document]) -> None:
        new = self.embedder.embed([d.text for d in docs])
        self.vectors = new if not self.docs else np.vstack([self.vectors, new])
        self.docs.extend(docs)

    def vector(self, doc_id: str) -> np.ndarray:
        return self.vectors[[d.id for d in self.docs].index(doc_id)]

    def search(self, query: str, k: int = 3) -> list[tuple[Document, float]]:
        """Return the k most similar docs as (doc, cosine_similarity), best first."""
        q = self.embedder.embed([query])[0]
        scores = self.vectors @ q                      # cosine, since all vectors have length 1
        best = np.argsort(-scores, kind="stable")[:k]  # stable => ties keep insertion order
        return [(self.docs[i], float(scores[i])) for i in best]
