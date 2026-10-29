"""PoisonWell UI.  Run:  streamlit run app/streamlit_app.py
Uses TF-IDF + MockLLM, so it needs no API keys, no downloads, and runs on Streamlit Cloud."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `import poisonwell` works

import streamlit as st

from poisonwell import config
from poisonwell.attacks import ATTACKS, attack_succeeded, load_attack_docs, load_questions
from poisonwell.rag import RAGPipeline, load_docs

st.set_page_config(page_title="PoisonWell", page_icon="☠️", layout="wide")


@st.cache_resource
def get_pipeline(attack: str) -> RAGPipeline:
    """One pipeline per attack choice (building it is quick, but there is no need to repeat it)."""
    poison = [] if attack == "none" else load_attack_docs(None if attack == "all" else attack)
    return RAGPipeline(load_docs("clean_docs") + poison, embedder_name="tfidf")


def playground():
    st.header("Playground")
    st.write("Pick a question and an attack, then see what the assistant retrieves and says. "
             "Turn the defense on to see which chunks get blocked, and why.")
    questions = load_questions()
    col1, col2 = st.columns(2)
    with col1:
        choice = st.selectbox("Question", [q["question"] for q in questions] + ["(type my own)"])
        if choice == "(type my own)":
            choice = st.text_input("Your question", "How many vacation days do I get per year?")
    with col2:
        attack = st.selectbox("Planted attack", ["none"] + list(ATTACKS) + ["all"])
        defense_on = st.toggle("Defense layer ON", value=False)
    if attack in ATTACKS:
        st.caption(ATTACKS[attack])

    pipe = get_pipeline(attack)
    result = pipe.ask(choice, defense_on=defense_on)

    st.subheader("Retrieved chunks")
    for i, c in enumerate(result.chunks, 1):
        poison = bool(c.doc.attack)
        tag = "🟥 POISON" if poison else "🟩 clean"
        status = "⛔ BLOCKED" if c.blocked else ("✅ used" if defense_on else "used")
        with st.expander(f"#{i}  {c.doc.title}  —  {tag}  |  {c.doc.source}  |  "
                         f"similarity {c.similarity:.2f}  |  {status}", expanded=c.blocked or poison):
            if c.risk is not None:
                st.write(f"**Risk score:** {c.risk:.2f}")
            if c.blocked:
                st.error(c.reason)
            st.write(c.doc.text)

    st.subheader("Answer")
    st.info(result.answer)
    poison_docs = [d for d in pipe.store.docs if d.attack]
    if poison_docs:
        if attack_succeeded(result.answer, poison_docs):
            st.error("Attack SUCCEEDED: the answer contains the attacker's canary or claim.")
        else:
            st.success("Attack failed: the answer does not contain the attacker's content.")
    with st.expander("How this works"):
        st.write("The question is turned into a vector, the 3 most similar docs are retrieved, "
                 "and (if the defense is on) risky ones are removed before the assistant answers. "
                 "The mock assistant is deliberately naive and obeys instructions it finds in documents.")


def results_page():
    st.header("Results")
    path = config.RESULTS_DIR / "results.json"
    if not path.exists():
        st.warning("No results yet. Run `python -m poisonwell.evaluate` first.")
        return
    r = json.loads(path.read_text())
    st.caption(f"Embedder: {r['embedder']} | LLM: {r['llm']} | top_k: {r['top_k']} | "
               f"{r['n_attack_questions']} attack questions, {r['n_normal_questions']} normal questions")
    st.image(str(config.RESULTS_DIR / "asr_chart.png"))
    rows = [{"Attack": k, "Questions": v["n"], "ASR (defense off)": f"{v['asr_off']:.0%}",
             "ASR (defense on)": f"{v['asr_on']:.0%}"} for k, v in r["per_attack"].items()]
    rows.append({"Attack": "OVERALL", "Questions": r["n_attack_questions"],
                 "ASR (defense off)": f"{r['overall']['asr_off']:.0%}",
                 "ASR (defense on)": f"{r['overall']['asr_on']:.0%}"})
    st.table(rows)
    c1, c2, c3 = st.columns(3)
    c1.metric("False positive rate", f"{r['false_positive_rate']:.1%}",
              f"{r['clean_chunks_blocked']}/{r['clean_chunks_retrieved']} clean chunks blocked", delta_color="off")
    c2.metric("Clean accuracy (defense off)", f"{r['clean_accuracy_off']:.0%}")
    c3.metric("Clean accuracy (defense on)", f"{r['clean_accuracy_on']:.0%}")
    if "ablation_asr_on" in r:
        st.subheader("What does each detector contribute?")
        st.write("Overall ASR when only one detector is active:")
        st.table([{"Detector alone": k, "ASR": f"{v:.0%}"} for k, v in r["ablation_asr_on"].items()])
    held_path = config.RESULTS_DIR / "heldout_results.json"
    if held_path.exists():
        h = json.loads(held_path.read_text())
        st.subheader("Held-out attacks (unseen)")
        st.write("Written after the detectors were frozen, with new phrasing and formats. "
                 "Compare with the tuned numbers above.")
        st.image(str(config.RESULTS_DIR / "asr_chart_heldout.png"))
        d = h["diagnostic_no_canaries"]
        st.table([{"Set": "Known attacks (tuned)", "ASR off": f"{r['overall']['asr_off']:.0%}",
                   "ASR on": f"{r['overall']['asr_on']:.0%}", "False positive rate": f"{r['false_positive_rate']:.1%}",
                   "ASR on, canaries removed": f"{r['diagnostic_no_canaries']['overall']['asr_on']:.0%}"},
                  {"Set": "Held-out attacks (unseen)", "ASR off": f"{h['overall']['asr_off']:.0%}",
                   "ASR on": f"{h['overall']['asr_on']:.0%}", "False positive rate": f"{h['false_positive_rate']:.1%}",
                   "ASR on, canaries removed": f"{d['overall']['asr_on']:.0%}"}])
        st.caption("'Canaries removed' is a diagnostic: the trust scorer's number-conflict check was "
                   "counting digits inside canary tokens, which flatters the headline held-out result.")
    with st.expander("Per-question details"):
        st.dataframe([{"id": x["id"], "question": x["question"], "attack": x["attack"],
                       "success_off": x.get("success_off"), "success_on": x.get("success_on"),
                       "blocked": ", ".join(b["doc"] for b in x["blocked"])} for x in r["records"]])


def about_page():
    st.header("About PoisonWell")
    st.markdown("""
**RAG** (retrieval-augmented generation) lets an AI assistant answer from a company's documents: it
finds the most relevant docs for a question and writes an answer from them.

**RAG poisoning** means an attacker gets a malicious document into that collection (a shared wiki,
an upload form, a scraped web page). The assistant then trusts it. This matters because the model
cannot easily tell *documents to read* from *instructions to obey*.

**What PoisonWell does** (all data is synthetic; Northwind Labs is fictional):
1. A small RAG assistant answers questions from ~40 internal docs.
2. Three attacks plant poisoned docs, each with a unique canary token so success is objective:
   *hidden instruction injection*, *misinformation*, and *retrieval hijacking*.
3. Three defenses score every retrieved chunk: an *instruction-pattern detector*, an *anomaly
   detector* (keyword stuffing, embedding outliers), and *source trust scoring*. A combined
   *DefenseLayer* blocks risky chunks and says why.
4. The evaluation runs 30 questions with the defense off and on, and reports attack success rate,
   false positive rate, and clean-answer accuracy.

The assistant is a deterministic *mock* that deliberately follows instructions found in documents,
which keeps results reproducible. See `docs/HOW_IT_WORKS.md` for the full walkthrough.
""")


PAGES = {"Playground": playground, "Results": results_page, "About": about_page}
st.sidebar.title("☠️ PoisonWell")
PAGES[st.sidebar.radio("Page", list(PAGES))]()
