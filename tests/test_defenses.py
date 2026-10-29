"""Each detector catches its target attack and leaves clean docs alone; DefenseLayer end to end."""
import pytest

from poisonwell.attacks import load_attack_docs
from poisonwell.defenses import AnomalyDetector, DefenseLayer, InstructionDetector, SourceTrustScorer
from poisonwell.rag import RAGPipeline, load_docs

CLEAN = load_docs("clean_docs")
POISON = load_attack_docs()


@pytest.fixture(scope="module")
def pipeline():
    return RAGPipeline(CLEAN + POISON, embedder_name="tfidf")


def by_attack(kind):
    return [d for d in POISON if d.attack == kind]


def test_instruction_detector_catches_injection_and_ignores_clean():
    det = InstructionDetector()
    assert all(det.score(d)[0] >= 0.8 for d in by_attack("injection"))
    assert all(det.score(d)[0] < 0.5 for d in CLEAN)


def test_instruction_detector_ignores_negated_advice():
    doc = CLEAN[0].__class__(id="x", title="t", category="it", source="internal-verified", date="",
                             text="Please never share your password with anyone.")
    assert InstructionDetector().score(doc)[0] == 0.0


def test_anomaly_detector_catches_hijack_and_ignores_clean(pipeline):
    det = AnomalyDetector(pipeline.store, [d for d in CLEAN if d.source == "internal-verified"])
    assert all(det.score(d)[0] >= 0.5 for d in by_attack("hijack"))
    assert all(det.score(d)[0] < 0.5 for d in CLEAN)


def test_source_trust_flags_conflicting_uploads_but_not_harmless_ones(pipeline):
    det = SourceTrustScorer(pipeline.store, [d for d in CLEAN if d.source == "internal-verified"])
    assert all(det.score(d)[0] >= 0.8 for d in by_attack("misinformation"))
    harmless = [d for d in CLEAN if d.source == "user-upload"]
    assert harmless and all(det.score(d)[0] < 0.5 for d in harmless)
    assert all(det.score(d)[0] == 0 for d in CLEAN if d.source == "internal-verified")


def test_defense_layer_blocks_poison_and_passes_clean(pipeline):
    verdicts = {v.doc_id: v for v in pipeline.defense.assess(CLEAN + POISON)}
    assert all(verdicts[d.id].blocked and verdicts[d.id].reason for d in POISON)
    assert not any(verdicts[d.id].blocked for d in CLEAN)


def test_defense_layer_can_be_switched_off(pipeline):
    layer = DefenseLayer(pipeline.store, enabled=False)
    assert not any(v.blocked for v in layer.assess(POISON))
