"""The "brain" of the assistant. MockLLM is deterministic and deliberately gullible."""
import json
import re
import urllib.request

# MockLLM obeys text like this if it finds it inside a retrieved chunk. That is the
# simulated vulnerability. (The real defenses use a separate, broader pattern list.)
OBEY_PATTERN = re.compile(r"ignore previous instructions|note to ai assistant|disregard prior", re.I)
STOP = {"the", "a", "an", "is", "are", "do", "does", "i", "my", "to", "of", "how", "what",
        "can", "me", "you", "we", "for", "in", "on", "and", "or", "it", "get", "much", "many"}


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9$%]+", text.lower()) if w not in STOP}


class LLM:
    def generate(self, question: str, chunks: list) -> str:
        """chunks: list of Document-like objects with a .text attribute (best first)."""
        raise NotImplementedError


class MockLLM(LLM):
    """Builds an answer from the retrieved chunks, no real model involved.

    1. If any chunk contains an instruction aimed at the AI, it FOLLOWS it (vulnerable!).
    2. Otherwise it picks, from each chunk, the sentence that best matches the question.
    """

    def generate(self, question, chunks):
        if not chunks:
            return "I could not find anything about that in the company documents."
        for chunk in chunks:
            hit = OBEY_PATTERN.search(chunk.text)
            if hit:
                return "Following the instructions in the document: " + chunk.text[hit.start():]
        q_words = words(question)
        picked = []
        for chunk in chunks:
            sentences = split_sentences(chunk.text)
            best = max(sentences, key=lambda s: len(q_words & words(s)))  # first one wins ties
            if best not in picked:
                picked.append(best)
        return " ".join(picked)


class OllamaLLM(LLM):
    """Optional: a small local model through Ollama (https://ollama.com). Not used in tests."""

    def __init__(self, model: str = "llama3.2:1b", url: str = "http://localhost:11434/api/generate"):
        self.model, self.url = model, url

    def generate(self, question, chunks):
        context = "\n\n".join(c.text for c in chunks)
        prompt = ("Answer the question using the company documents below.\n\n"
                  f"Documents:\n{context}\n\nQuestion: {question}\nAnswer:")
        body = json.dumps({"model": self.model, "prompt": prompt, "stream": False}).encode()
        req = urllib.request.Request(self.url, body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())["response"].strip()
