from poisonwell.evaluate import run_eval, save_chart


def test_eval_runs_end_to_end_on_a_small_subset(tmp_path):
    r = run_eval(embedder_name="tfidf", per_group=2, ablation=False)
    assert r["n_attack_questions"] == 6 and r["n_normal_questions"] == 2
    for key in ("injection", "misinformation", "hijack"):
        assert 0 <= r["per_attack"][key]["asr_on"] <= r["per_attack"][key]["asr_off"] <= 1
    assert r["overall"]["asr_on"] < r["overall"]["asr_off"]
    assert 0 <= r["false_positive_rate"] <= 1
    save_chart(r, tmp_path / "chart.png")
    assert (tmp_path / "chart.png").stat().st_size > 0


def test_heldout_set_runs_and_is_separate():
    from poisonwell.attacks import load_attack_docs, load_questions
    docs = load_attack_docs(None, "heldout_docs")
    assert len(docs) == 12 and all(d.source == "user-upload" for d in docs)
    original = {d.canary for d in load_attack_docs()}
    assert not original & {d.canary for d in docs}          # held-out canaries are all new
    assert len(load_questions("heldout_questions.json")) == 12
    r = run_eval(embedder_name="tfidf", per_group=1, ablation=False,
                 attack_folder="heldout_docs", questions_file="heldout_questions.json")
    assert r["n_attack_questions"] == 3 and r["n_normal_questions"] == 0
