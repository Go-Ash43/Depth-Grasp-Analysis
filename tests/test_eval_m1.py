"""scripts/eval_m1.py + scripts/h1_harness.load_eval on the tiny synthetic cache (CPU): CSV = run_val rows,
corruption sidecars, training-CSV comparison, one-look ledger (concurrent launch, crash retry, retry
exhaustion, identity change, crash between completion and publication), equal-area masks with different
geometry refused (P-E1 rev 4)."""
import csv
import json
import os
from pathlib import Path

import numpy as np
import pytest

import training.runcfg as rc
from models.grconv_lite import build_model
from scripts import eval_m1 as em
from scripts import h1_harness as hh
from scripts.m1_runs import CURRENT, sha256
from tests.test_dataset import SPLITS, cache  # noqa: F401  (fixture)
from tests.test_recovery import fake_env
from tests.test_runcfg import restore_precision, write_manifest  # noqa: F401  (fixture)
from training.checkpoint import save_checkpoint

ALLOWED = ("train",)


@pytest.fixture
def setup(cache, tmp_path, monkeypatch, restore_precision):  # noqa: F811
    monkeypatch.delenv("DEPTH_LIES_PATHS", raising=False)
    monkeypatch.setattr(rc, "require_clean_tree", lambda allow: {"git_commit": "abc1234", "git_dirty": False})
    monkeypatch.setattr(rc, "env_record", lambda: fake_env("A"))
    man = write_manifest(cache, tmp_path / "manifest.json")
    cfg = dict(json.loads(rc.DEFAULT_CONFIG.read_text()), status="frozen", epochs=14)
    cfg_p = tmp_path / "cfg.json"
    cfg_p.write_text(json.dumps(cfg))
    ek = {**rc.env_key(fake_env("A")), "execution": rc.device_record("cpu")}
    runs = {}
    for m in ("R", "A0", "M"):
        for s in (1, 2, 3):
            d = tmp_path / "runs" / f"{m}_s{s}"
            ck = save_checkpoint(d / "ckpt" / "epoch_013.pt",
                                 {"meta": {"model": m, "seed": s}, "model": build_model(m, s).state_dict()})
            runs[f"{m}:{s}"] = {"model": m, "seed": s, "path": str(d), "run_id": f"{m}{s}", "env_key": ek,
                                "final_ckpt": {"path": str(d / "ckpt" / "epoch_013.pt"), "sha256": ck},
                                "final_csv": {"path": str(d / "val" / "epoch_013.csv"), "sha256": None}}
    shared = {"config_sha256": rc.file_sha256(cfg_p), "manifest_sha256": rc.file_sha256(man),
              "val_conditions": ["mid", "none"], **{k: rc.file_sha256(p) for k, p in CURRENT.items()}}
    met = tmp_path / "metric_local.json"
    met.write_text(json.dumps({"metric": "primary"}))
    sel = {"runs": runs, "shared": shared, "m3": {"chosen": "pod"},
           "metric_decision": {"metric": "primary", "local_sha256": sha256(met)}}
    sel_p = tmp_path / "selected_runs.json"
    sel_p.write_text(json.dumps(sel))
    return {"cache": cache, "man": man, "cfg": cfg_p, "sel": sel_p, "tmp": tmp_path, "ledger": tmp_path / "ledger",
            "metric": met}


def _argv(S, out, *extra):
    return ["--selected", str(S["sel"]), "--split", "train", "--manifest", str(S["man"]), "--config", str(S["cfg"]),
            "--cache-dir", str(S["cache"]), "--out", str(out), "--device", "cpu", "--num-workers", "0",
            "--hash-workers", "2", "--expect-manifest-sha256", rc.file_sha256(S["man"]), *extra]


def _test_run(S, out, *extra):                         # "train" plays the ledger-guarded test split
    S["ledger"].mkdir(exist_ok=True)
    return em.main(_argv(S, out, *extra), splits=SPLITS, allowed=ALLOWED, ledger_splits=("train",),
                   _ledger_dir=S["ledger"])


def _val_run(S, out):                                   # "train" plays val (training-CSV comparison on)
    return em.main(_argv(S, out), splits=SPLITS, allowed=ALLOWED, ledger_splits=())


def _ledger(S):
    return json.loads((S["ledger"] / em.LEDGER_NAME).read_text())


def test_test_run_once_then_publish_only(setup):
    S = setup
    out = S["tmp"] / "test"
    assert _test_run(S, out) == 0
    L = _ledger(S)
    assert L["status"] == "completed" and L["retries"] == 0
    res = out / "results"
    for rel, h in L["outputs"].items():
        assert sha256(res / rel) == h
    side = list(csv.DictReader((res / "M_s1" / "mid.sidecar.csv").read_text().splitlines()))
    rows = list(csv.DictReader((res / "M_s1" / "mid.csv").read_text().splitlines()))
    assert [r["slot"] for r in side] == sorted((r["slot"] for r in rows), key=int)
    assert all(int(r["n_synth_cells"]) >= 0 for r in side)
    with pytest.raises(SystemExit, match="one look"):
        _test_run(S, out)
    with pytest.raises(SystemExit, match="one look"):
        _test_run(S, S["tmp"] / "another_out")       # a new --out does not give a second look
    assert _test_run(S, out, "--publish-only") == 0
    assert json.loads((out / "summary.json").read_text())["summary"]


def test_concurrent_launch_refused(setup):
    S = setup
    S["ledger"].mkdir()
    fd = em.acquire(str(S["ledger"] / em.LEDGER_NAME) + ".lock")
    try:
        with pytest.raises(SystemExit, match="in progress"):
            _test_run(S, S["tmp"] / "test")
    finally:
        os.close(fd)
    assert not (S["ledger"] / em.LEDGER_NAME).exists()


def _crash(monkeypatch, times):
    real, n = em.eval_one, {"k": 0}

    def flaky(*a, **k):
        n["k"] += 1
        if n["k"] <= times:
            raise RuntimeError("simulated crash")
        return real(*a, **k)
    monkeypatch.setattr(em, "eval_one", flaky)


def test_crash_retry_once_then_exhausted(setup, monkeypatch):
    S = setup
    out = S["tmp"] / "test"
    _crash(monkeypatch, 1)
    with pytest.raises(RuntimeError, match="simulated"):
        _test_run(S, out)
    assert _ledger(S)["status"] == "started" and not (out / "summary.json").exists()
    assert _test_run(S, out) == 0                     # the one allowed retry
    L = _ledger(S)
    assert L["status"] == "completed" and L["retries"] == 1


def test_retry_exhaustion_and_identity_change(setup, monkeypatch):
    S = setup
    out = S["tmp"] / "test"
    _crash(monkeypatch, 2)
    with pytest.raises(RuntimeError):
        _test_run(S, out)
    with pytest.raises(RuntimeError):
        _test_run(S, out)                             # retry 1 crashes too
    with pytest.raises(SystemExit, match="retry limit"):
        _test_run(S, out)
    S2 = dict(S, ledger=S["tmp"] / "ledger2")
    _crash(monkeypatch, 1)
    with pytest.raises(RuntimeError):
        _test_run(S2, out)
    monkeypatch.setattr(rc, "env_record", lambda: fake_env("B"))
    with pytest.raises(SystemExit, match="identical identities"):
        _test_run(S2, out)


def test_crash_between_completion_and_publication(setup, monkeypatch):
    S = setup
    out = S["tmp"] / "test"
    real = em.summary_from_csv
    monkeypatch.setattr(em, "summary_from_csv", lambda b: (_ for _ in ()).throw(RuntimeError("publish crash")))
    with pytest.raises(RuntimeError, match="publish crash"):
        _test_run(S, out)
    assert _ledger(S)["status"] == "completed" and not (out / "summary.json").exists()
    monkeypatch.setattr(em, "summary_from_csv", real)
    with pytest.raises(SystemExit, match="one look"):
        _test_run(S, out)                             # no second inference
    assert _test_run(S, out, "--publish-only") == 0 and (out / "summary.json").exists()


def _write_training_csvs(S, res):
    for k in ("R:1", "A0:1", "M:1", "R:2", "A0:2", "M:2", "R:3", "A0:3", "M:3"):
        m, s = k.split(":")
        p = Path(S["tmp"] / "runs" / f"{m}_s{s}" / "val" / "epoch_013.csv")
        p.parent.mkdir(parents=True, exist_ok=True)
        a = (res / f"{m}_s{s}" / "none.csv").read_text().splitlines()
        b = (res / f"{m}_s{s}" / "mid.csv").read_text().splitlines()
        p.write_text("\n".join(a + b[1:]) + "\n")
    return p


def test_val_mode_compares_with_training_csv(setup):
    S = setup
    assert _test_run(S, S["tmp"] / "first") == 0
    last = _write_training_csvs(S, S["tmp"] / "first" / "results")
    assert _val_run(S, S["tmp"] / "val") == 0
    rec = json.loads((S["tmp"] / "val" / "results" / "eval_record.json").read_text())
    c = rec["training_comparison"]["M:3"]
    assert c["exact_required"] and c["mid"]["rows_identical"] and c["none"]["primary"]["acc_change_pp"] == 0
    rows = list(csv.DictReader(last.read_text().splitlines()))
    rows[0]["match_primary"] = "False" if rows[0]["match_primary"] == "True" else "True"
    with open(last, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(SystemExit, match="rows differ"):
        _val_run(S, S["tmp"] / "val2")


def test_harness_load_eval_refuses_equal_area_different_geometry(setup):
    S = setup
    assert _test_run(S, S["tmp"] / "t") == 0
    res = S["tmp"] / "t" / "results"
    files = json.loads((res / "eval_record.json").read_text())["files"]
    from data.cache_layout import open_cache
    slots = sorted(int(r["slot"]) for r in open_cache(S["cache"], "train", splits=SPLITS)["index"] if r["status"] == "ok")
    rows, side = hh.load_eval(res, files, slots)
    assert set(rows) == {f"{m}_s{s}" for m in ("R", "A0", "M") for s in (1, 2, 3)}
    # same n_synth_cells, different geometry -> different hash -> refused
    a = np.zeros((224, 224), bool)
    a[10:14, 10:14] = True
    b = np.zeros((224, 224), bool)
    b[50:52, 50:58] = True
    assert a.sum() == b.sum() and em._h(a) != em._h(b)
    p = res / "R_s2" / "mid.sidecar.csv"
    sd = list(csv.DictReader(p.read_text().splitlines()))
    sd[0]["m_synth_sha256"] = em._h(b)
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=em.SIDE_FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(sd)
    with pytest.raises(hh.HarnessError, match="corruption identity of R_s2"):
        hh.load_eval(res, files, slots)


def test_crash_after_rename_before_ledger_completion_recovers(setup, monkeypatch):
    """Review fix 2: started ledger + fully written results -> completed WITHOUT inference or a retry."""
    S = setup
    out = S["tmp"] / "test"

    def boom(stage):
        if stage == "after_rename":
            raise RuntimeError("crash after rename")
    monkeypatch.setattr(em, "_crash_hook", boom)
    with pytest.raises(RuntimeError, match="after rename"):
        _test_run(S, out)
    assert _ledger(S)["status"] == "started" and (out / "results").exists()
    monkeypatch.setattr(em, "_crash_hook", lambda stage: None)
    calls = []
    real = em.eval_one
    monkeypatch.setattr(em, "eval_one", lambda *a, **k: calls.append(1) or real(*a, **k))
    assert _test_run(S, out) == 0
    L = _ledger(S)
    assert L["status"] == "completed" and L["retries"] == 0 and "recovered" in L and calls == []
    assert (out / "summary.json").exists()


def test_crash_window_publish_only_and_tampered_results_refused(setup, monkeypatch):
    S = setup
    out = S["tmp"] / "test"
    monkeypatch.setattr(em, "_crash_hook", lambda st: (_ for _ in ()).throw(RuntimeError("x")) if st == "after_rename" else None)
    with pytest.raises(RuntimeError):
        _test_run(S, out)
    monkeypatch.setattr(em, "_crash_hook", lambda st: None)
    p = out / "results" / "A0_s1" / "mid.csv"
    p.write_bytes(p.read_bytes() + b"\n")
    with pytest.raises(SystemExit, match="sha256 differs"):
        _test_run(S, out, "--publish-only")
    assert _ledger(S)["status"] == "started"


def _harness_argv(S, eval_out, out, split="train"):
    return ["--selected", str(S["sel"]), "--split", split, "--eval-dir", str(eval_out),
            "--metric-local", str(S["metric"]), "--config", str(S["cfg"]), "--cache-dir", str(S["cache"]),
            "--out", str(out), "--resamples", "50"]


def test_harness_on_the_test_split_end_to_end(setup, monkeypatch):
    """Review fix 1: the harness analyses the ledger-guarded split; G1 is reported, not gated, off validation."""
    S = setup
    monkeypatch.setattr(hh.mp, "load_decision", lambda p, recheck=True: {"metric": "primary"})
    assert _test_run(S, S["tmp"] / "t") == 0
    kw = dict(allowed=("train",), ledger_splits=("train",), g1_splits=(), _ledger_dir=S["ledger"])
    assert hh.main(_harness_argv(S, S["tmp"] / "t", S["tmp"] / "h1"), **kw) == 0
    res = json.loads((S["tmp"] / "h1" / "h1.json").read_text())
    assert res["split"] == "train" and res["G1_clean_A0"]["gated"] is False
    assert res["primary"]["condition"] == "mid" and "reading" in res["primary"]
    breakdown = res["misses_by_target_opening_no_verdict"]
    assert len(breakdown) == 36
    assert all(set(v) == {"primary", "strict"} for v in breakdown.values())
    assert res["G1_clean_A0"]["passed"] is None
    gated =json.loads((S["tmp"] / "h1" / "h1.json").read_text())
    assert gated["inputs"]["index_sha256"] == sha256(Path(S["cache"]) / "train" / "index.csv")
    kw_val = dict(allowed=("train",), ledger_splits=(), g1_splits=("train",))
    original_hits = hh.hits

    def failed_clean(rows, model, cond, slots, metric):   # clean A0 forced to 0 % -> G1 fails
        if model == "A0" and cond == "none":
            return np.zeros((3, len(slots)))
        return original_hits(rows, model, cond, slots, metric)
    monkeypatch.setattr(hh, "hits", failed_clean)
    assert hh.main(_harness_argv(S, S["tmp"] / "t", S["tmp"] / "h1b"), **kw_val) == 1
    g1 = json.loads((S["tmp"] / "h1b" / "h1.json").read_text())["G1_clean_A0"]
    assert g1["gated"] is True and g1["passed"] is False


@pytest.mark.parametrize("state", ["dirty", "changed"])
def test_harness_requires_frozen_test_analysis(setup, monkeypatch, state):
    """Review round 2, finding 1: the test split is analysed only at the commit the test run recorded."""
    S = setup
    assert _test_run(S, S["tmp"] / "t") == 0

    def guard(allow):
        if state == "dirty":
            raise SystemExit("working tree is dirty")
        return {"git_commit": "different_commit", "git_dirty": False}
    monkeypatch.setattr(rc, "require_clean_tree", guard)
    kw = dict(allowed=("train",), ledger_splits=("train",), g1_splits=(), _ledger_dir=S["ledger"])
    with pytest.raises(SystemExit, match="dirty|frozen analysis commit"):
        hh.main(_harness_argv(S, S["tmp"] / "t", S["tmp"] / "h1"), **kw)
    assert not (S["tmp"] / "h1").exists()


def test_harness_refuses_uncompleted_test_ledger(setup, monkeypatch):
    S = setup
    monkeypatch.setattr(hh.mp, "load_decision", lambda p, recheck=True: {"metric": "primary"})
    monkeypatch.setattr(em, "_crash_hook", lambda st: (_ for _ in ()).throw(RuntimeError("x")) if st == "after_rename" else None)
    with pytest.raises(RuntimeError):
        _test_run(S, S["tmp"] / "t")
    kw = dict(allowed=("train",), ledger_splits=("train",), g1_splits=(), _ledger_dir=S["ledger"])
    with pytest.raises(SystemExit, match="not completed"):
        hh.main(_harness_argv(S, S["tmp"] / "t", S["tmp"] / "h1"), **kw)


def test_harness_refuses_index_not_in_verified_manifest(setup, monkeypatch):
    """Review fix 4: a changed object mapping is refused, never silently used."""
    S = setup
    monkeypatch.setattr(hh.mp, "load_decision", lambda p, recheck=True: {"metric": "primary"})
    assert _test_run(S, S["tmp"] / "t") == 0
    idx = Path(S["cache"]) / "train" / "index.csv"
    rows = list(csv.DictReader(idx.read_text().splitlines()))
    rows[0]["object_id"] = rows[0]["object_id"] + "_moved"
    with open(idx, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    kw = dict(allowed=("train",), ledger_splits=("train",), g1_splits=(), _ledger_dir=S["ledger"])
    with pytest.raises(SystemExit, match="verified manifest entry"):
        hh.main(_harness_argv(S, S["tmp"] / "t", S["tmp"] / "h1"), **kw)


def test_incomplete_evaluation_refused(setup):
    S = setup
    assert _test_run(S, S["tmp"] / "t") == 0
    res = S["tmp"] / "t" / "results"
    files = json.loads((res / "eval_record.json").read_text())["files"]
    files.pop("M_s3/high.csv")
    with pytest.raises(hh.HarnessError, match="missing"):
        hh.load_eval(res, files, [0])
