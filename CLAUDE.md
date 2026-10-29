# PoisonWell: notes for Claude / contributors

**Goal:** a small, readable RAG poisoning attack + defense lab (portfolio project for a CS student).
Clarity beats cleverness. Keep the code base small (~2k lines excluding data).

**Stack:** Python 3.11+, NumPy, scikit-learn (TF-IDF), matplotlib, Streamlit, pytest.
Optional: sentence-transformers (MiniLM), Ollama.

## Commands
- Run app: `streamlit run app/streamlit_app.py`
- Run tests: `python -m pytest -q`
- Run eval: `python -m poisonwell.evaluate` (rewrites `results/`; update README numbers afterwards)
- Regenerate data: `python scripts/make_data.py && python scripts/make_attacks.py && python scripts/make_heldout.py`

## File map
- `poisonwell/config.py` settings (embedder, top_k, thresholds)
- `embeddings.py` TF-IDF / MiniLM, `store.py` Document + VectorStore, `llm.py` MockLLM / OllamaLLM
- `rag.py` RAGPipeline, `attacks.py` attack metadata + success check, `defenses.py` 3 detectors + DefenseLayer
- `evaluate.py` eval runner + chart, `app/streamlit_app.py` UI, `tests/` pytest, `data/` synthetic docs/questions

## Rules
1. Keep it simple: no Docker, DB server, auth, CI, or extra frameworks.
2. MockLLM + TF-IDF are the defaults; tests must run offline and fast.
3. Synthetic data only. No real people, companies, emails, or domains (use `*.example`).
4. Detectors must never read the ground-truth labels (`attack`, `canary`, `target_claim`).
5. README numbers must come from a real eval run.
6. Held-out set (`data/heldout_*`) is evaluated separately and must never be used to tune detectors. If you change a detector, say so, and write a fresh held-out set.
