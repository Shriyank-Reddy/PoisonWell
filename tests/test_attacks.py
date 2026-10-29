from poisonwell.attacks import ATTACKS, attack_succeeded, load_attack_docs, load_questions
from poisonwell.rag import RAGPipeline, load_docs


def test_every_poison_doc_is_tagged_and_unique():
    docs = load_attack_docs()
    assert {d.attack for d in docs} == set(ATTACKS)
    assert all(d.source == "user-upload" for d in docs)
    canaries = [d.canary for d in docs]
    assert len(set(canaries)) == len(canaries)


def test_claims_never_appear_in_clean_docs():
    clean = " ".join(d.text.lower() for d in load_docs("clean_docs"))
    for d in load_attack_docs():
        assert d.canary.lower() not in clean and d.target_claim.lower() not in clean


def test_question_set_shape():
    qs = load_questions()
    assert len(qs) == 30
    assert sum(q["attack"] is not None for q in qs) == 20


def test_each_attack_works_without_defense():
    for attack in ATTACKS:
        docs = load_attack_docs(attack)
        pipe = RAGPipeline(load_docs("clean_docs") + docs, embedder_name="tfidf")
        qs = [q for q in load_questions() if q["attack"] == attack]
        wins = sum(attack_succeeded(pipe.ask(q["question"]).answer, docs) for q in qs)
        assert wins >= len(qs) // 2, (attack, wins, len(qs))
