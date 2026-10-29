"""The RAG pipeline: retrieve chunks -> (optionally) filter with the defense -> answer."""
import json
from dataclasses import dataclass, field

from poisonwell import config
from poisonwell.embeddings import get_embedder
from poisonwell.llm import LLM, MockLLM
from poisonwell.store import Document, VectorStore


def load_docs(folder: str) -> list[Document]:
    """Load every *.json file in data/<folder>/ into Document objects."""
    docs = []
    for path in sorted((config.DATA_DIR / folder).glob("*.json")):
        docs += [Document(**row) for row in json.loads(path.read_text())]
    return docs


@dataclass
class RetrievedChunk:
    doc: Document
    similarity: float
    risk: float | None = None      # filled in when the defense is ON
    blocked: bool = False
    reason: str = ""


@dataclass
class RAGResult:
    question: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    answer: str = ""

    @property
    def used(self) -> list[RetrievedChunk]:
        return [c for c in self.chunks if not c.blocked]


class RAGPipeline:
    def __init__(self, docs: list[Document], llm: LLM | None = None,
                 embedder_name: str | None = None, top_k: int = config.TOP_K):
        self.llm = llm or MockLLM()
        self.top_k = top_k
        self.store = VectorStore.build(docs, get_embedder(embedder_name or config.EMBEDDER))
        self._defense = None  # built on first use (see the `defense` property)

    @property
    def defense(self):
        if self._defense is None:
            from poisonwell.defenses import DefenseLayer  # imported here to avoid a circular import
            self._defense = DefenseLayer(self.store)
        return self._defense

    def ask(self, question: str, defense_on: bool = False) -> RAGResult:
        hits = self.store.search(question, self.top_k)
        chunks = [RetrievedChunk(doc, sim) for doc, sim in hits]
        if defense_on:
            for chunk, verdict in zip(chunks, self.defense.assess([c.doc for c in chunks])):
                chunk.risk, chunk.blocked, chunk.reason = verdict.risk, verdict.blocked, verdict.reason
        answer = self.llm.generate(question, [c.doc for c in chunks if not c.blocked])
        return RAGResult(question, chunks, answer)
