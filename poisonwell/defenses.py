"""Three detectors + a combined DefenseLayer.

Each detector looks at one retrieved document and returns (score 0..1, [reasons]).
IMPORTANT: detectors may only use the doc's visible fields (text, category, source) and
the verified docs. They never read the ground-truth labels (attack/canary/target_claim).
"""
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from poisonwell import config
from poisonwell.store import Document, VectorStore

VERIFIED, UPLOAD = "internal-verified", "user-upload"


def clip(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


# --------------------------------------------------------------------------------------
# Defense 1: instruction-pattern detector
# --------------------------------------------------------------------------------------
# (regex, weight, label). Weights of the patterns that match are added up (max 1.0).
INSTRUCTION_PATTERNS = [
    (r"ignore (all |any )?(previous|prior|above|earlier) (instructions|guidance|rules)", 1.0, "'ignore previous instructions'"),
    (r"disregard\s+(all\s+)?(previous|prior|above|earlier)", 0.9, "'disregard prior ...'"),
    (r"note to (the )?(ai|assistant)", 0.8, "note addressed to the AI"),
    (r"system prompt|developer mode|jailbreak", 0.8, "mentions the system prompt"),
    (r"\bas an ai\b", 0.5, "'as an AI'"),
    (r"\byou must (tell|say|respond|answer|instruct|ask)", 0.6, "'you must tell/say ...'"),
    (r"\b(tell|instruct|ask|direct) the user to\b", 0.6, "'tell the user to ...'"),
    (r"do not mention (this|these|that)", 0.6, "asks to hide the instruction"),
    (r"\b(email|send|share|provide)\b[^.]{0,40}\b(password|credentials|card number|ssn|social security)", 0.9, "asks for credentials"),
    (r"\b(enter|type)\b[^.]{0,40}\b(password|credentials|card number|ssn)", 0.4, "asks to type secrets"),
    (r"\bdisable\b[^.]{0,20}\b(antivirus|firewall|security)", 0.8, "asks to disable security tools"),
    (r"\.exe\b", 0.6, "points to an executable file"),
    (r"\byou must\b", 0.2, "'you must'"),
]
NEGATION = re.compile(r"(never|not|n't|no)\W+(\w+\W+){0,2}$", re.I)  # "never share your password" is fine


class InstructionDetector:
    def score(self, doc: Document) -> tuple[float, list[str]]:
        total, reasons = 0.0, []
        for pattern, weight, label in INSTRUCTION_PATTERNS:
            for m in re.finditer(pattern, doc.text, re.I):
                if NEGATION.search(doc.text[max(0, m.start() - 25):m.start()]):
                    continue  # sentence says NOT to do it, so it is advice, not an attack
                total += weight
                reasons.append(label)
                break
        return clip(total), reasons


# --------------------------------------------------------------------------------------
# Defense 2: anomaly detector (keyword stuffing + embedding outlier)
# --------------------------------------------------------------------------------------
class AnomalyDetector:
    def __init__(self, store: VectorStore, verified: list[Document]):
        self.store = store
        # Group verified docs by category and remember their vectors.
        self.by_cat: dict[str, list[Document]] = {}
        for d in verified:
            self.by_cat.setdefault(d.category, []).append(d)
        # For each category: mean and spread of "similarity to the centroid of the others".
        self.stats = {}
        for cat, members in self.by_cat.items():
            sims = [self._sim_to_centroid(m, cat) for m in members]
            self.stats[cat] = (float(np.mean(sims)), float(np.std(sims)) or 1e-6)

    def _sim_to_centroid(self, doc: Document, cat: str) -> float:
        others = [m for m in self.by_cat[cat] if m.id != doc.id]  # leave the doc itself out
        centroid = np.mean([self.store.vector(m.id) for m in others], axis=0)
        norm = np.linalg.norm(centroid)
        return float(self.store.vector(doc.id) @ centroid / norm) if norm else 0.0

    @staticmethod
    def stuffing_score(text: str) -> float:
        """Keyword stuffing = a few words repeated again and again."""
        tokens = [w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in ENGLISH_STOP_WORDS]
        if not tokens:
            return 0.0
        unique_ratio = len(set(tokens)) / len(tokens)               # normal prose: ~0.7-0.9
        top_share = Counter(tokens).most_common(1)[0][1] / len(tokens)  # one word's share of all words
        return max(clip((0.68 - unique_ratio) / 0.20), clip((top_share - 0.14) / 0.10))

    def score(self, doc: Document) -> tuple[float, list[str]]:
        reasons = []
        stuffing = self.stuffing_score(doc.text)
        if stuffing >= 0.3:
            reasons.append("keyword stuffing (abnormal term repetition)")
        outlier = 0.0
        if doc.category in self.stats:
            mean, std = self.stats[doc.category]
            sim = self._sim_to_centroid(doc, doc.category)
            outlier = clip(((mean - sim) / std - 1.5) / 2.0)        # z-score below the category norm
            if outlier >= 0.3:
                reasons.append(f"unlike other '{doc.category}' docs (embedding outlier)")
        return max(stuffing, outlier), reasons


# --------------------------------------------------------------------------------------
# Defense 3: source trust scoring
# --------------------------------------------------------------------------------------
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def numbers_in(text: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in NUMBER.findall(text)}


class SourceTrustScorer:
    """Verified content is trusted. Uploads get a base risk, which jumps when the upload
    covers the same topic as a verified doc but states different numbers (a contradiction)."""
    BASE_UPLOAD_RISK = 0.3
    MIN_TOPIC_SIMILARITY = 0.15

    def __init__(self, store: VectorStore, verified: list[Document]):
        self.store, self.verified = store, verified

    def score(self, doc: Document) -> tuple[float, list[str]]:
        if doc.source == VERIFIED:
            return 0.0, []
        reasons = ["unverified 'user-upload' source"]
        vec = self.store.vector(doc.id)
        sims = [(float(vec @ self.store.vector(v.id)), v) for v in self.verified]
        sim, nearest = max(sims, key=lambda t: t[0])
        mine, theirs = numbers_in(doc.text), numbers_in(nearest.text)
        if sim < self.MIN_TOPIC_SIMILARITY or not mine or not theirs:
            return self.BASE_UPLOAD_RISK, reasons
        unseen = sorted(mine - theirs)
        conflict = len(unseen) / len(mine)
        if conflict > 0:
            reasons.append(f"conflicts with verified doc '{nearest.title}' "
                           f"(figures not found there: {', '.join(unseen)})")
        return self.BASE_UPLOAD_RISK + (1 - self.BASE_UPLOAD_RISK) * conflict, reasons


# --------------------------------------------------------------------------------------
# Combined defense layer
# --------------------------------------------------------------------------------------
@dataclass
class Assessment:
    doc_id: str
    risk: float
    signals: dict           # {"instruction": 0.9, "anomaly": 0.1, "trust": 0.3}
    blocked: bool
    reason: str             # human-readable; empty when the chunk is allowed


class DefenseLayer:
    WEIGHTS = {"instruction": 0.5, "anomaly": 0.25, "trust": 0.25}

    def __init__(self, store: VectorStore, threshold: float = config.BLOCK_THRESHOLD,
                 strong: float = config.STRONG_SIGNAL, enabled: bool = True,
                 only: str | None = None):
        verified = [d for d in store.docs if d.source == VERIFIED]
        self.detectors = {
            "instruction": InstructionDetector(),
            "anomaly": AnomalyDetector(store, verified),
            "trust": SourceTrustScorer(store, verified),
        }
        if only:  # keep a single detector (used for the ablation table in the evaluation)
            self.detectors = {only: self.detectors[only]}
        self.threshold, self.strong, self.enabled = threshold, strong, enabled

    def assess(self, docs: list[Document]) -> list[Assessment]:
        """Score each chunk. If the layer is switched off, nothing is ever blocked."""
        out = []
        for doc in docs:
            results = {name: det.score(doc) for name, det in self.detectors.items()}
            signals = {name: round(score, 3) for name, (score, _) in results.items()}
            risk = sum(self.WEIGHTS[n] * s for n, s in signals.items())
            blocked = self.enabled and (risk >= self.threshold or max(signals.values()) >= self.strong)
            reason = ""
            if blocked:
                parts = [f"{n}: {', '.join(rs)}" for n, (s, rs) in results.items() if rs and s >= 0.3]
                reason = f"risk {risk:.2f} -> blocked. " + "; ".join(parts)
            out.append(Assessment(doc.id, round(risk, 3), signals, blocked, reason))
        return out
