"""Smoke tests for the clean RAG pipeline (TF-IDF + MockLLM, fully offline)."""
from poisonwell.rag import RAGPipeline, load_docs


def make_pipeline():
    return RAGPipeline(load_docs("clean_docs"), embedder_name="tfidf")


def test_clean_docs_loaded():
    docs = load_docs("clean_docs")
    assert 40 <= len(docs) <= 45
    assert all(100 <= len(d.text.split()) <= 250 for d in docs)
    assert {d.source for d in docs} == {"internal-verified", "user-upload"}


def test_retrieval_finds_the_right_doc():
    result = make_pipeline().ask("How many sick days do I get?")
    assert result.chunks[0].doc.id == "hr-02"


def test_clean_answer_is_correct():
    assert "20 days" in make_pipeline().ask("How many vacation days do I get per year?").answer
