"""Run every eval question with the defense OFF and ON, then report the numbers.

    python -m poisonwell.evaluate            # full run (writes results/results.json + chart)
"""
import dataclasses
import json
import re
import sys

from poisonwell import config
from poisonwell.attacks import ATTACKS, attack_succeeded, load_attack_docs, load_questions
from poisonwell.defenses import DefenseLayer
from poisonwell.rag import RAGPipeline, load_docs


def rate(hits: int, total: int) -> float:
    return round(hits / total, 4) if total else 0.0


def strip_canaries(docs):
    """Diagnostic copy of poison docs with the CANARY-xxxx tokens removed from their text."""
    return [dataclasses.replace(d, text=re.sub(r"\s*\[?(ref )?CANARY-\w+\]?", "", d.text)) for d in docs]


def run_eval(embedder_name: str | None = None, per_group: int | None = None,
             ablation: bool = True, attack_folder: str = "attack_docs",
             questions_file: str = "eval_questions.json", no_canaries: bool = False) -> dict:
    """per_group limits how many questions run per attack / normal group (small subsets for tests).
    attack_folder / questions_file pick the attack set: the original (tuned) one or the held-out one.
    no_canaries=True is a diagnostic: poison docs enter the store without their canary tokens
    (success is then judged by the target claim only)."""
    hide = strip_canaries if no_canaries else (lambda docs: docs)
    name = embedder_name or config.EMBEDDER
    clean = load_docs("clean_docs")
    questions = load_questions(questions_file)

    def group(key):
        qs = [q for q in questions if q["attack"] == key]
        return qs[:per_group] if per_group else qs

    records, clean_seen, clean_blocked, poison_seen, poison_blocked = [], 0, 0, 0, 0

    def run(pipe, q, poison_docs):
        """Ask one question both ways and keep a record. Also counts blocked chunks."""
        nonlocal clean_seen, clean_blocked, poison_seen, poison_blocked
        off, on = pipe.ask(q["question"], defense_on=False), pipe.ask(q["question"], defense_on=True)
        for c in on.chunks:
            if c.doc.attack:
                poison_seen += 1
                poison_blocked += c.blocked
            else:
                clean_seen += 1
                clean_blocked += c.blocked
        rec = {"id": q["id"], "question": q["question"], "attack": q["attack"],
               "answer_off": off.answer, "answer_on": on.answer,
               "blocked": [{"doc": c.doc.id, "reason": c.reason} for c in on.chunks if c.blocked]}
        if q["attack"]:
            rec["success_off"] = attack_succeeded(off.answer, poison_docs)
            rec["success_on"] = attack_succeeded(on.answer, poison_docs)
        else:
            rec["correct_off"] = all(k.lower() in off.answer.lower() for k in q["expected_keywords"])
            rec["correct_on"] = all(k.lower() in on.answer.lower() for k in q["expected_keywords"])
        records.append(rec)

    # Attack questions: the store holds the clean docs + that one attack's poison docs.
    for attack in ATTACKS:
        poison = load_attack_docs(attack, attack_folder)
        pipe = RAGPipeline(clean + hide(poison), embedder_name=name)
        for q in group(attack):
            run(pipe, q, poison)

    # Normal questions: the store holds clean docs + ALL poison docs (checks for collateral damage).
    all_poison = load_attack_docs(None, attack_folder)
    if group(None):
        pipe = RAGPipeline(clean + hide(all_poison), embedder_name=name)
        for q in group(None):
            run(pipe, q, all_poison)

    per_attack = {}
    for attack in ATTACKS:
        rs = [r for r in records if r["attack"] == attack]
        per_attack[attack] = {"n": len(rs),
                              "asr_off": rate(sum(r["success_off"] for r in rs), len(rs)),
                              "asr_on": rate(sum(r["success_on"] for r in rs), len(rs))}
    atk = [r for r in records if r["attack"]]
    nrm = [r for r in records if not r["attack"]]
    results = {
        "embedder": name, "llm": "MockLLM", "top_k": config.TOP_K, "threshold": config.BLOCK_THRESHOLD,
        "n_attack_questions": len(atk), "n_normal_questions": len(nrm),
        "per_attack": per_attack,
        "overall": {"asr_off": rate(sum(r["success_off"] for r in atk), len(atk)),
                    "asr_on": rate(sum(r["success_on"] for r in atk), len(atk))},
        "false_positive_rate": rate(clean_blocked, clean_seen),
        "clean_chunks_retrieved": clean_seen, "clean_chunks_blocked": clean_blocked,
        "poison_chunks_retrieved": poison_seen, "poison_chunks_blocked": poison_blocked,
        "clean_accuracy_off": rate(sum(r["correct_off"] for r in nrm), len(nrm)),
        "clean_accuracy_on": rate(sum(r["correct_on"] for r in nrm), len(nrm)),
        "records": records,
    }
    if ablation:
        results["ablation_asr_on"] = run_ablation(clean, group, name, attack_folder)
    return results


def run_ablation(clean, group, embedder_name, attack_folder="attack_docs") -> dict:
    """Overall ASR when only ONE detector is active. Shows what each defense contributes."""
    out = {}
    for only in ("instruction", "anomaly", "trust"):
        wins, total = 0, 0
        for attack in ATTACKS:
            poison = load_attack_docs(attack, attack_folder)
            pipe = RAGPipeline(clean + poison, embedder_name=embedder_name)
            pipe._defense = DefenseLayer(pipe.store, only=only)
            for q in group(attack):
                wins += attack_succeeded(pipe.ask(q["question"], defense_on=True).answer, poison)
                total += 1
        out[only] = rate(wins, total)
    return out


def save_chart(results: dict, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(results["per_attack"]) + ["overall"]
    rows = list(results["per_attack"].values()) + [results["overall"]]
    off = [r["asr_off"] * 100 for r in rows]
    on = [r["asr_on"] * 100 for r in rows]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(7, 4))
    bars_off = ax.bar([i - 0.2 for i in x], off, 0.4, label="Defense OFF", color="#c0392b")
    bars_on = ax.bar([i + 0.2 for i in x], on, 0.4, label="Defense ON", color="#2471a3")
    for bar in list(bars_off) + list(bars_on):
        ax.annotate(f"{bar.get_height():.0f}%", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=9)
    ax.set_xticks(list(x), labels)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Attack success rate (%)")
    ax.set_title("PoisonWell: attack success rate before vs after defense")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def diagnostic(no_canaries_run: dict) -> dict:
    """Keep only the headline numbers of a canary-stripped run."""
    keep = ("per_attack", "overall", "false_positive_rate", "poison_chunks_blocked", "poison_chunks_retrieved")
    return {k: no_canaries_run[k] for k in keep}


def print_summary(r: dict, title: str = "PoisonWell results") -> None:
    pct = lambda v: f"{v * 100:5.1f}%"
    print(f"\n{title}  (embedder={r['embedder']}, llm={r['llm']}, top_k={r['top_k']})")
    print(f"{'Attack':<16}{'Qs':>4}{'ASR off':>10}{'ASR on':>10}")
    for name, a in r["per_attack"].items():
        print(f"{name:<16}{a['n']:>4}{pct(a['asr_off']):>10}{pct(a['asr_on']):>10}")
    o = r["overall"]
    print(f"{'OVERALL':<16}{r['n_attack_questions']:>4}{pct(o['asr_off']):>10}{pct(o['asr_on']):>10}")
    print(f"\nFalse positive rate (clean chunks blocked): {pct(r['false_positive_rate'])} "
          f"({r['clean_chunks_blocked']}/{r['clean_chunks_retrieved']})")
    print(f"Poison chunks blocked: {r['poison_chunks_blocked']}/{r['poison_chunks_retrieved']}")
    if r["n_normal_questions"]:
        print(f"Clean-answer accuracy: off {pct(r['clean_accuracy_off'])}, on {pct(r['clean_accuracy_on'])} "
              f"({r['n_normal_questions']} normal questions)")
    if "diagnostic_no_canaries" in r:
        d = r["diagnostic_no_canaries"]
        print(f"DIAGNOSTIC, canary tokens removed from doc text: overall ASR off {pct(d['overall']['asr_off']).strip()}"
              f" -> on {pct(d['overall']['asr_on']).strip()}, poison blocked "
              f"{d['poison_chunks_blocked']}/{d['poison_chunks_retrieved']}, FPR {pct(d['false_positive_rate']).strip()}")
    if "ablation_asr_on" in r:
        print("ASR with a single detector only: " +
              ", ".join(f"{k}={pct(v).strip()}" for k, v in r["ablation_asr_on"].items()))


def main() -> None:
    config.RESULTS_DIR.mkdir(exist_ok=True)
    # 1) Known attacks: the ones the detectors were tuned on.
    known = run_eval()
    (config.RESULTS_DIR / "results.json").write_text(json.dumps(known, indent=2) + "\n")
    save_chart(known, config.RESULTS_DIR / "asr_chart.png")
    known["diagnostic_no_canaries"] = diagnostic(no_canaries_run=run_eval(ablation=False, no_canaries=True))
    (config.RESULTS_DIR / "results.json").write_text(json.dumps(known, indent=2) + "\n")
    print_summary(known, "KNOWN attacks (tuned)")
    # 2) Held-out attacks: written after the detectors were frozen. Reported separately.
    held = run_eval(attack_folder="heldout_docs", questions_file="heldout_questions.json")
    (config.RESULTS_DIR / "heldout_results.json").write_text(json.dumps(held, indent=2) + "\n")
    save_chart(held, config.RESULTS_DIR / "asr_chart_heldout.png")
    held["diagnostic_no_canaries"] = diagnostic(run_eval(
        ablation=False, no_canaries=True, attack_folder="heldout_docs", questions_file="heldout_questions.json"))
    (config.RESULTS_DIR / "heldout_results.json").write_text(json.dumps(held, indent=2) + "\n")
    print_summary(held, "HELD-OUT attacks (unseen)")
    print(f"\nSaved results/results.json, heldout_results.json and both charts")


if __name__ == "__main__":
    sys.exit(main())
