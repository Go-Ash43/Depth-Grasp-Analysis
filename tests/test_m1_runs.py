"""scripts/m1_runs.py: read-only run integrity, venue groups, the D7-1 rule (backup optional, logical id
separate from the directory), shared provenance, metric-record comparison (P-E1 rev 4)."""
import copy
import json
from pathlib import Path

import pytest
import torch

import training.runcfg as rc
from scripts import m1_runs as mr
from training.checkpoint import append_log, atomic_write_bytes, atomic_write_json, ckpt_path, csv_path, save_checkpoint

E, SPE = 2, 3


@pytest.fixture
def cfg_path(tmp_path):
    c = dict(json.loads(rc.DEFAULT_CONFIG.read_text()), status="frozen", epochs=E)
    p = tmp_path / "train_frozen.json"
    p.write_text(json.dumps(c))
    return p


def make_run(d, model, seed, cfg_path, env="pod", epochs=E, complete=True, run_id=None, **over):
    d = Path(d)
    d.mkdir(parents=True)
    run_id = run_id or f"{model}{seed}{env}"
    shared = {"kind": "production", "epochs": epochs, "batch_size": 32, "steps_per_epoch": SPE, "horizon": epochs * SPE,
              "n_train": 10, "train_slots_sha256": "t", "limit_train": 0, "holes": "train", "flag_setting": "deployed",
              "val_split": "val", "val_conditions": ["mid", "none"], "n_val": 5,
              "config_sha256": rc.file_sha256(cfg_path), "labels_sha256": rc.file_sha256(mr.CURRENT["labels_sha256"]),
              "train_data_sha256": rc.file_sha256(mr.CURRENT["train_data_sha256"]),
              "frozen_sha256": rc.file_sha256(mr.CURRENT["frozen_sha256"]), "manifest_sha256": "m",
              "git_commit": "2b85aa6d0a148da11a8b7a0e79e5fbf81654c3e6"}
    shared.update(over)
    rcfg = {"run_id": run_id, "model": model, "seed": seed, **shared, "git_dirty": False, "budget_valid": True,
            "env_key": {"gpus": [env]}}
    atomic_write_json(d / "run_config.json", rcfg)
    n_done = epochs if complete else epochs - 1
    val = None
    for e in range(n_done):
        csv_sha = atomic_write_bytes(csv_path(d, e), f"csv {model}{seed} {e}\n".encode())
        meta = {"run_id": run_id, "epoch": e, "global_step": (e + 1) * SPE, "csv_sha256": csv_sha, "model": model,
                "seed": seed, "holes": "train", "horizon": shared["horizon"], "config_sha256": shared["config_sha256"]}
        ck = save_checkpoint(ckpt_path(d, e), {"meta": meta, "model": {"w": torch.zeros(1)}})
        val = {"none": {"n": 5}}
        append_log(d, {"run_id": run_id, "epoch": e, "global_step": (e + 1) * SPE, "ckpt_sha256": ck,
                       "csv_sha256": csv_sha, "val": val, "budget_valid": True, "continuation_exact": True})
    if complete:
        atomic_write_json(d / "final.json", {"run_id": run_id, "completed_epochs": epochs, "final_val": val,
                                             "final_checkpoint": str(ckpt_path(d, epochs - 1))})
    return d


def make_set(root, cfg_path, m3_pod=True, m3_laptop=True, laptop_complete=True):
    for s in (1, 2):
        for m in mr.MODELS:
            make_run(root / f"{m}_s{s}", m, s, cfg_path, "pod")
    for m in ("A0", "R"):
        make_run(root / f"{m}_s3", m, 3, cfg_path, "laptop")
    pod = make_run(root / "M_s3_pod", "M", 3, cfg_path, "pod") if m3_pod else None
    lap = make_run(root / "M_s3", "M", 3, cfg_path, "laptop", complete=laptop_complete) if m3_laptop else None
    return mr.default_runs(root), (str(pod) if pod else None), (str(lap) if lap else None)


def test_pod_chosen_with_interrupted_backup(tmp_path, cfg_path):
    runs, pod, lap = make_set(tmp_path, cfg_path, laptop_complete=False)
    sel = mr.select(runs, pod, lap, "2b85aa6", cfg_path)
    assert sel["m3"]["chosen"] == "pod" and sel["runs"]["M:3"]["path"].endswith("M_s3_pod")
    assert sel["m3"]["laptop_candidate"]["status"].startswith("incomplete or rejected")
    assert sorted(sel["runs"]) == sorted(f"{m}:{s}" for m in mr.MODELS for s in (1, 2, 3))


def test_backup_absent_entirely(tmp_path, cfg_path):
    runs, pod, _ = make_set(tmp_path, cfg_path, m3_laptop=False)
    assert mr.select(runs, pod, None, "2b85aa6", cfg_path)["m3"]["chosen"] == "pod"


def test_laptop_used_when_pod_incomplete(tmp_path, cfg_path):
    runs, _, lap = make_set(tmp_path, cfg_path, m3_pod=False)
    make_run(tmp_path / "M_s3_pod", "M", 3, cfg_path, "pod", complete=False)
    sel = mr.select(runs, str(tmp_path / "M_s3_pod"), lap, "2b85aa6", cfg_path)
    assert sel["m3"]["chosen"] == "laptop" and sel["runs"]["M:3"]["path"].endswith("/M_s3")


def test_pod_with_foreign_env_rejected(tmp_path, cfg_path):
    runs, _, lap = make_set(tmp_path, cfg_path, m3_pod=False)
    make_run(tmp_path / "M_s3_pod", "M", 3, cfg_path, "elsewhere")
    sel = mr.select(runs, str(tmp_path / "M_s3_pod"), lap, "2b85aa6", cfg_path)
    assert sel["m3"]["chosen"] == "laptop" and "env_key" in sel["m3"]["pod_candidate"]["status"]


@pytest.mark.parametrize("over", [{"train_slots_sha256": "other"}, {"git_commit": "f" * 40},
                                  {"config_sha256": "0" * 64}])
def test_pod_with_incompatible_provenance_falls_back_to_backup(tmp_path, cfg_path, over):
    """Review fix 3: an incompatible pod M_s3 is a rejected candidate, not an abort."""
    runs, _, lap = make_set(tmp_path, cfg_path, m3_pod=False)
    make_run(tmp_path / "M_s3_pod", "M", 3, cfg_path, "pod", **over)
    sel = mr.select(runs, str(tmp_path / "M_s3_pod"), lap, "2b85aa6", cfg_path)
    assert sel["m3"]["chosen"] == "laptop"
    assert "provenance differs" in sel["m3"]["pod_candidate"]["status"] and list(over)[0] in sel["m3"]["pod_candidate"]["status"]


def test_no_usable_m3(tmp_path, cfg_path):
    runs, _, _ = make_set(tmp_path, cfg_path, m3_pod=False, m3_laptop=False)
    with pytest.raises(mr.RunError, match="no usable M_s3"):
        mr.select(runs, None, None, "2b85aa6", cfg_path)


def test_tampered_checkpoint_and_wrong_identity_refused(tmp_path, cfg_path):
    runs, pod, lap = make_set(tmp_path, cfg_path)
    p = ckpt_path(runs[("A0", 1)], 0)
    p.write_bytes(p.read_bytes() + b"x")
    with pytest.raises(mr.RunError, match="checkpoint 0 sha256"):
        mr.select(runs, pod, lap, "2b85aa6", cfg_path)
    with pytest.raises(mr.RunError, match="expected R/1"):
        mr.check_run(runs[("M", 1)], "R", 1, E)


def test_shared_field_and_commit_and_venue_group(tmp_path, cfg_path):
    runs, pod, lap = make_set(tmp_path, cfg_path)
    with pytest.raises(mr.RunError, match="expected 1234567"):
        mr.select(runs, pod, lap, "1234567", cfg_path)
    d = tmp_path / "other"
    make_set(d, cfg_path)
    import shutil
    shutil.rmtree(d / "R_s2")
    make_run(d / "R_s2", "R", 2, cfg_path, "pod", train_slots_sha256="different")
    r2, p2, l2 = mr.default_runs(d), str(d / "M_s3_pod"), str(d / "M_s3")
    with pytest.raises(mr.RunError, match="train_slots_sha256"):
        mr.select(r2, p2, l2, "2b85aa6", cfg_path)
    shutil.rmtree(d / "R_s2")
    make_run(d / "R_s2", "R", 2, cfg_path, "laptop")
    with pytest.raises(mr.RunError, match="pod group"):
        mr.select(r2, p2, l2, "2b85aa6", cfg_path)


def test_draft_config_refused(tmp_path, cfg_path):
    runs, pod, lap = make_set(tmp_path, cfg_path)
    c = json.loads(cfg_path.read_text())
    c["status"] = "draft"
    p = tmp_path / "draft.json"
    p.write_text(json.dumps(c))
    with pytest.raises(mr.RunError, match="frozen"):
        mr.select(runs, pod, lap, "2b85aa6", p)


def _decision(run_dir):
    return {"decision_version": 1, "policy": "P", "metric": "primary", "promoted": False, "threshold_pct": 95,
            "comparison": ">", "clean": {"n": 5, "matches_primary": 4},
            "run": {"run_dir": str(run_dir), "run_id": "A01pod", "kind": "production", "model": "A0", "seed": 1,
                    "epochs": E, "git_commit": "2b85", "config_sha256": "c", "config_path": f"{run_dir}/cfg.json",
                    "continuation_exact": True},
            "final_checkpoint": {"path": f"{run_dir}/ckpt/epoch_001.pt", "sha256": "K"},
            "final_csv": {"path": f"{run_dir}/val/epoch_001.csv", "sha256": "C"}}


def test_relocated_pod_metric_record_compared_by_portable_fields(tmp_path, monkeypatch):
    local = _decision(tmp_path / "A0_s1")
    pod = _decision("/workspace/runs/A0_s1")             # absent here: must never be opened
    (tmp_path / "local.json").write_text(json.dumps(local))
    (tmp_path / "pod.json").write_text(json.dumps(pod))
    calls = []
    monkeypatch.setattr(mr.mp, "load_decision", lambda p, recheck=True: calls.append(str(p)) or copy.deepcopy(local))
    a0 = {"run_id": "A01pod", "final_ckpt": {"sha256": "K"}}
    rec = mr.check_metric(tmp_path / "local.json", tmp_path / "pod.json", a0)
    assert rec["metric"] == "primary" and calls == [str(tmp_path / "local.json")]
    pod["metric"] = "strict"
    (tmp_path / "pod.json").write_text(json.dumps(pod))
    with pytest.raises(mr.RunError, match="metric"):
        mr.check_metric(tmp_path / "local.json", tmp_path / "pod.json", a0)
    pod = _decision("/workspace/runs/A0_s1")
    pod["final_checkpoint"]["sha256"] = "OTHER"
    (tmp_path / "pod.json").write_text(json.dumps(pod))
    with pytest.raises(mr.RunError, match="final_checkpoint_sha256"):
        mr.check_metric(tmp_path / "local.json", tmp_path / "pod.json", a0)
    (tmp_path / "pod.json").write_text(json.dumps(_decision("/workspace/runs/A0_s1")))
    with pytest.raises(mr.RunError, match="not from the selected"):
        mr.check_metric(tmp_path / "local.json", tmp_path / "pod.json", {**a0, "run_id": "other"})


def test_main_writes_once(tmp_path, cfg_path, monkeypatch):
    make_set(tmp_path / "R", cfg_path)
    argv = ["--runs-root", str(tmp_path / "R"), "--commit", "2b85aa6", "--m3-pod", str(tmp_path / "R" / "M_s3_pod"),
            "--m3-laptop", str(tmp_path / "R" / "M_s3"), "--config", str(cfg_path), "--out", str(tmp_path / "sel")]
    assert mr.main(argv + ["--no-metric"]) == 0 and not (tmp_path / "sel").exists()
    with pytest.raises(SystemExit, match="metric-local"):
        mr.main(argv)
