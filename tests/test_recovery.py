"""training/checkpoint.py and scripts/train.py: atomic writes, weights_only checkpoints, log parsing,
crash recovery, and uninterrupted == interrupted + resumed (CPU, tiny synthetic cache)."""
import json
from pathlib import Path

import pytest
import torch

import training.runcfg as rc
from scripts import train as tr
from tests.test_dataset import SPLITS, cache  # noqa: F401  (fixture)
from tests.test_runcfg import restore_precision, write_manifest  # noqa: F401  (fixture)
from training.checkpoint import (acquire_run_lock, append_log, atomic_write_bytes, capture_state, ckpt_path,
                                 csv_path, load_checkpoint, read_log, recover, release_run_lock, restore_state,
                                 save_checkpoint)
from models.grconv_lite import build_model


# ------------------------------------------------------------------ units
def test_atomic_write_leaves_no_tmp(tmp_path):
    p = tmp_path / "d" / "f.bin"
    atomic_write_bytes(p, b"abc")
    assert p.read_bytes() == b"abc" and not list(p.parent.glob("*.tmp"))


def _objects():
    m = build_model("R", 0)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
    m(torch.randn(2, 3, 224, 224))["q"].mean().backward()
    opt.step()
    sch.step()
    gens = {"train_loader": torch.Generator().manual_seed(1), "val_loader": torch.Generator().manual_seed(2)}
    return m, opt, sch, gens


def test_checkpoint_round_trip_weights_only(tmp_path):
    m, opt, sch, gens = _objects()
    sha = save_checkpoint(tmp_path / "c.pt", capture_state(m, opt, sch, gens, {"epoch": 0, "model": "R"}))
    st = load_checkpoint(tmp_path / "c.pt", sha)               # torch.load(weights_only=True) inside
    m2, opt2, sch2, gens2 = _objects()
    restore_state(st, m2, opt2, sch2, gens2)
    assert all(torch.equal(a, b) for a, b in zip(m.state_dict().values(), m2.state_dict().values()))
    assert sch2.state_dict() == sch.state_dict()
    assert torch.equal(gens2["val_loader"].get_state(), gens["val_loader"].get_state())
    with pytest.raises(RuntimeError, match="sha256"):
        load_checkpoint(tmp_path / "c.pt", "0" * 64)


def test_read_log_torn_and_malformed(tmp_path):
    (tmp_path / "log.jsonl").write_text('{"epoch": 0}\n{"epoch": 1')
    recs, torn = read_log(tmp_path)
    assert recs == [{"epoch": 0}] and torn == '{"epoch": 1'
    (tmp_path / "log.jsonl").write_text('{"epoch": 0}\n{"epo\n')
    recs, torn = read_log(tmp_path)
    assert recs == [{"epoch": 0}] and torn == '{"epo'
    (tmp_path / "log.jsonl").write_text('{"epoch": 0}\n{bad\n{"epoch": 2}\n')
    with pytest.raises(RuntimeError, match="malformed"):
        read_log(tmp_path)


def _fake_run(tmp_path, n_committed, orphan=True):
    m, opt, sch, gens = _objects()
    for e in range(n_committed + (1 if orphan else 0)):
        csha = atomic_write_bytes(csv_path(tmp_path, e), f"epoch {e}\n".encode())
        meta = {"run_id": "r", "epoch": e, "global_step": e + 1, "csv_sha256": csha, "model": "R"}
        ksha = save_checkpoint(ckpt_path(tmp_path, e), capture_state(m, opt, sch, gens, meta))
        if e < n_committed:
            append_log(tmp_path, {"run_id": "r", "epoch": e, "global_step": e + 1, "ckpt_sha256": ksha,
                                  "csv_sha256": csha})
    (tmp_path / "ckpt" / "epoch_009.pt.tmp").write_bytes(b"partial")


def test_recover_archives_finished_orphans_and_tmp(tmp_path):
    _fake_run(tmp_path, 2)
    with open(tmp_path / "log.jsonl", "a") as fh:
        fh.write('{"run_id": "r", "ep')                       # torn final record
    k, rep = recover(tmp_path, "r", {"model": "R"})
    assert k == 1
    assert {"ckpt/epoch_002.pt", "val/epoch_002.csv", "ckpt/epoch_009.pt.tmp"} <= set(rep["archived"])
    assert not ckpt_path(tmp_path, 2).exists() and ckpt_path(tmp_path, 1).exists()
    assert (tmp_path / rep["archive_dir"] / "ckpt" / "epoch_002.pt").exists()
    assert len(read_log(tmp_path)[0]) == 2 and read_log(tmp_path)[1] is None


def test_recover_refuses_tampering(tmp_path):
    _fake_run(tmp_path, 2, orphan=False)
    with pytest.raises(RuntimeError, match="meta model"):
        recover(tmp_path, "r", {"model": "A0"})
    csv_path(tmp_path, 1).write_bytes(b"changed\n")
    with pytest.raises(RuntimeError, match="val CSV"):
        recover(tmp_path, "r", {"model": "R"})


# ------------------------------------------------------------------ end to end
class Crash(RuntimeError):
    pass


def fake_env(tag="A", ok=True):
    """A fixed environment record: the suite must not depend on this machine's packages."""
    return {"versions": {"n_pins": 0, "ok": ok, "python": "3.11.9", "python_ok": True,
                         "mismatches": {} if ok else {"numpy": {"locked": "2.4.6", "installed": "0"}}},
            "hardware": {"gpus": [{"name": f"gpu-{tag}"}], "torch": "t", "torch_cuda": "c", "cudnn": 1,
                         "driver": ["d"], "cv2": "5", "numpy": "2", "cpus": {"usable": 4}, "platform": "p"},
            "torch_threads": 4}


@pytest.fixture
def env(cache, tmp_path, monkeypatch, restore_precision):  # noqa: F811
    monkeypatch.delenv("DEPTH_LIES_PATHS", raising=False)
    monkeypatch.setattr(rc, "check_w_norm", lambda *a, **k: {"stub": "synthetic cache"})
    monkeypatch.setattr(rc, "env_record", lambda: fake_env("A"))
    man = write_manifest(cache, tmp_path / "manifest.json")
    # an explicit DRAFT copy: no test depends on the live config's status / epochs / seeds
    draft = dict(json.loads(rc.DEFAULT_CONFIG.read_text()), status="draft", epochs=None)
    (tmp_path / "draft.json").write_text(json.dumps(draft))
    return cache, man, tmp_path


def _argv(cache_dir, man, out, *extra, epochs=2):
    return ["--kind", "smoke", "--model", "A0", "--seed", "1", "--epochs", str(epochs), "--allow-dirty",
            "--config", str(Path(man).parent / "draft.json"),
            "--device", "cpu", "--num-workers", "0", "--val-split", "train", "--cache-dir", str(cache_dir),
            "--manifest", str(man), "--hash-workers", "2", "--out", str(out), *extra]


def _states_equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_states_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_states_equal(x, y) for x, y in zip(a, b))
    return a == b


def _final(out):
    recs, _ = read_log(out)
    return load_checkpoint(ckpt_path(out, 1), recs[1]["ckpt_sha256"])


def _crash_at(monkeypatch, stage, epoch):
    def boom(s, e):
        if (s, e) == (stage, epoch):
            raise Crash(f"{s} {e}")
    monkeypatch.setattr(tr, "_crash_hook", boom)


def _no_crash(monkeypatch):
    monkeypatch.setattr(tr, "_crash_hook", lambda s, e: None)


def _a_vs_b(env, monkeypatch, stage, epoch, extra=()):
    cache_dir, man, tmp = env
    assert tr.main(_argv(cache_dir, man, tmp / "a", *extra), splits=SPLITS) == 0
    _crash_at(monkeypatch, stage, epoch)
    with pytest.raises(Crash):
        tr.main(_argv(cache_dir, man, tmp / "b", *extra), splits=SPLITS)
    _no_crash(monkeypatch)
    assert tr.main(_argv(cache_dir, man, tmp / "b", *extra, "--resume"), splits=SPLITS) == 0
    a, b = _final(tmp / "a"), _final(tmp / "b")
    for k in ("model", "optimizer", "scheduler"):
        assert _states_equal(a[k], b[k]), k
    assert a["meta"]["global_step"] == b["meta"]["global_step"]
    return tmp


@pytest.mark.parametrize("stage,epoch", [("after_ckpt", 0), ("after_log", 0), ("after_csv", 1),
                                         ("after_train", 1)])
def test_interrupted_and_resumed_equals_uninterrupted(env, monkeypatch, stage, epoch):
    cache_dir, man, tmp = env
    assert tr.main(_argv(cache_dir, man, tmp / "a"), splits=SPLITS) == 0
    _crash_at(monkeypatch, stage, epoch)
    with pytest.raises(Crash):
        tr.main(_argv(cache_dir, man, tmp / "b"), splits=SPLITS)
    _no_crash(monkeypatch)
    assert tr.main(_argv(cache_dir, man, tmp / "b", "--resume"), splits=SPLITS) == 0
    a, b = _final(tmp / "a"), _final(tmp / "b")
    for k in ("model", "optimizer", "scheduler"):
        assert _states_equal(a[k], b[k]), k
    assert a["meta"]["global_step"] == b["meta"]["global_step"]
    rec = json.loads((tmp / "b" / "recovery.json").read_text())[-1]
    if stage == "after_ckpt":                          # finished checkpoint, no log line: an orphan
        assert rec["last_committed_epoch"] == -1 and "ckpt/epoch_000.pt" in rec["archived"]
    if stage == "after_csv":
        assert rec["last_committed_epoch"] == 0 and "val/epoch_001.csv" in rec["archived"]


def test_resume_bound_to_model_and_seed(env):
    cache_dir, man, tmp = env
    assert tr.main(_argv(cache_dir, man, tmp / "a"), splits=SPLITS) == 0
    argv = _argv(cache_dir, man, tmp / "a", "--resume")
    argv[argv.index("--seed") + 1] = "2"
    with pytest.raises(SystemExit, match="immutable"):
        tr.main(argv, splits=SPLITS)
    with pytest.raises(SystemExit, match="exist"):
        tr.main(_argv(cache_dir, man, tmp / "a"), splits=SPLITS)


@pytest.mark.parametrize("extra,msg", [
    (["--kind", "production", "--val-split", "val"], "frozen"),
    (["--kind", "pilot", "--val-split", "val"], "smoke runs only"),
    (["--kind", "pilot"], "validates on 'val' only"),
    (["--kind", "production"], "validates on 'val' only"),
    (["--val-split", "test_module"], "test shards"),
    (["--val-split", "test_sealed"], "test shards"),
    (["--epochs", "0"], "positive"),
    (["--epochs", "-1"], "positive"),
])
def test_kind_rules(env, extra, msg):
    cache_dir, man, tmp = env
    with pytest.raises(SystemExit, match=msg):
        tr.main(_argv(cache_dir, man, tmp / "x") + extra, splits=SPLITS)
    assert not (tmp / "x").exists()


def test_production_with_frozen_config_refuses_overrides(env):
    cache_dir, man, tmp = env
    frozen = tmp / "frozen.json"
    frozen.write_text(json.dumps(dict(json.loads((tmp / "draft.json").read_text()), status="frozen", epochs=2)))
    argv = _argv(cache_dir, man, tmp / "x", "--kind", "production", "--val-split", "val", "--config", str(frozen))
    with pytest.raises(SystemExit, match="takes no --epochs"):
        tr.main(argv, splits=SPLITS)


def test_device_change_breaks_exact_continuation(env, monkeypatch):
    cache_dir, man, tmp = env
    out = tmp / "r"
    _crash_at(monkeypatch, "after_log", 0)
    with pytest.raises(Crash):
        tr.main(_argv(cache_dir, man, out, epochs=3), splits=SPLITS)
    _no_crash(monkeypatch)
    monkeypatch.setattr(rc, "device_record", lambda d: {"type": "cuda", "index": 0, "name": "Fake GPU"})
    with pytest.raises(SystemExit, match="environment differs"):
        tr.main(_argv(cache_dir, man, out, "--resume", epochs=3), splits=SPLITS)
    assert tr.main(_argv(cache_dir, man, out, "--resume", "--allow-env-change", epochs=3), splits=SPLITS) == 0
    assert [r["continuation_exact"] for r in read_log(out)[0]] == [True, False, False]


def test_run_lock_excludes_a_second_process_and_is_released(env):
    cache_dir, man, tmp = env
    out = tmp / "a"
    assert tr.main(_argv(cache_dir, man, out), splits=SPLITS) == 0      # released after success
    fd = acquire_run_lock(out)                                            # a "second process" holds it
    try:
        with pytest.raises(SystemExit, match="in use by another process"):
            tr.main(_argv(cache_dir, man, out, "--resume"), splits=SPLITS)
    finally:
        release_run_lock(fd)
    assert tr.main(_argv(cache_dir, man, out, "--resume"), splits=SPLITS) == 0   # free again
    fd = acquire_run_lock(out)                                            # and released after that resume
    release_run_lock(fd)


def test_lock_released_after_an_exception(env, monkeypatch):
    cache_dir, man, tmp = env
    out = tmp / "a"
    _crash_at(monkeypatch, "after_train", 0)
    with pytest.raises(Crash):
        tr.main(_argv(cache_dir, man, out), splits=SPLITS)
    release_run_lock(acquire_run_lock(out))                               # would raise if still held


def test_epoch_timing_covers_outputs_and_log_commit(env):
    cache_dir, man, tmp = env
    assert tr.main(_argv(cache_dir, man, tmp / "a", epochs=3), splits=SPLITS) == 0
    recs = read_log(tmp / "a")[0]
    for i, r in enumerate(recs):
        t = r["time"]
        assert t["epoch_wall_s"] >= t["train_s"] + t["val_s"] + t["csv_s"] + t["ckpt_s"]
        assert r["projection_includes_log_commit"] == (i > 0)
        if i > 0:
            assert t["prev_log_commit_s"] > 0
            assert r["projected_run_s"] == pytest.approx((t["epoch_wall_s"] + t["prev_log_commit_s"]) * 3)
    assert json.loads((tmp / "a" / "final.json").read_text())["last_log_commit_s"] > 0


def test_version_mismatch_refused(env, monkeypatch):
    cache_dir, man, tmp = env
    monkeypatch.setattr(rc, "env_record", lambda: fake_env("A", ok=False))
    with pytest.raises(SystemExit, match="requirements-lock"):
        tr.main(_argv(cache_dir, man, tmp / "x"), splits=SPLITS)


def test_resuming_a_completed_run_keeps_final_metrics(env):
    cache_dir, man, tmp = env
    assert tr.main(_argv(cache_dir, man, tmp / "a"), splits=SPLITS) == 0
    before = json.loads((tmp / "a" / "final.json").read_text())
    assert tr.main(_argv(cache_dir, man, tmp / "a", "--resume"), splits=SPLITS) == 0
    after = json.loads((tmp / "a" / "final.json").read_text())
    assert after["final_val"] is not None and after["final_val"] == before["final_val"]
    assert len(read_log(tmp / "a")[0]) == 2


def test_resuming_a_completed_run_keeps_last_log_commit(env):
    cache_dir, man, tmp = env
    assert tr.main(_argv(cache_dir, man, tmp / "a"), splits=SPLITS) == 0
    before = json.loads((tmp / "a" / "final.json").read_text())["last_log_commit_s"]
    assert before is not None and before > 0
    assert tr.main(_argv(cache_dir, man, tmp / "a", "--resume"), splits=SPLITS) == 0
    assert json.loads((tmp / "a" / "final.json").read_text())["last_log_commit_s"] == before


def test_continuation_exact_false_is_inherited(env, monkeypatch):
    """A -> B (--allow-env-change) -> A: every epoch after the change stays non-exact."""
    cache_dir, man, tmp = env
    out = tmp / "r"
    _crash_at(monkeypatch, "after_log", 0)
    with pytest.raises(Crash):
        tr.main(_argv(cache_dir, man, out, epochs=3), splits=SPLITS)
    monkeypatch.setattr(rc, "env_record", lambda: fake_env("B"))
    with pytest.raises(SystemExit, match="environment differs"):
        tr.main(_argv(cache_dir, man, out, "--resume", epochs=3), splits=SPLITS)
    _crash_at(monkeypatch, "after_log", 1)
    with pytest.raises(Crash):
        tr.main(_argv(cache_dir, man, out, "--resume", "--allow-env-change", epochs=3), splits=SPLITS)
    monkeypatch.setattr(rc, "env_record", lambda: fake_env("A"))   # back to the original environment
    _no_crash(monkeypatch)
    assert tr.main(_argv(cache_dir, man, out, "--resume", epochs=3), splits=SPLITS) == 0
    assert [r["continuation_exact"] for r in read_log(out)[0]] == [True, False, False]
    assert [h["continuation_exact"] for h in json.loads((out / "recovery.json").read_text())] == [False, False]


def test_two_recoveries_in_one_second_never_collide(tmp_path, monkeypatch):
    _fake_run(tmp_path, 2)
    monkeypatch.setattr("training.checkpoint.time.strftime", lambda fmt: "20000101-000000")
    with open(tmp_path / "log.jsonl", "a") as fh:
        fh.write('{"torn": 1')
    _, r1 = recover(tmp_path, "r", {"model": "R"})
    with open(tmp_path / "log.jsonl", "a") as fh:      # second recovery: ONLY a torn log tail
        fh.write('{"torn": 2')
    _, r2 = recover(tmp_path, "r", {"model": "R"})
    assert r1["archive_dir"] != r2["archive_dir"]
    assert (tmp_path / r1["archive_dir"] / "log.torn_tail.txt").read_text() == '{"torn": 1'
    assert (tmp_path / r2["archive_dir"] / "log.torn_tail.txt").read_text() == '{"torn": 2'
    assert r2["archived"] == ["log.jsonl (torn final record)"]


def test_resume_equality_with_persistent_workers(env, monkeypatch):
    _a_vs_b(env, monkeypatch, "after_log", 0, extra=("--num-workers", "2", "--no-pin-memory"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device (run on the pod)")
def test_cuda_resume_equality_with_persistent_workers(env, monkeypatch):
    for stage, epoch in (("after_log", 0), ("after_ckpt", 1)):
        tmp = env[2]
        for d in ("a", "b"):
            if (tmp / d).exists():
                import shutil
                shutil.rmtree(tmp / d)
        _a_vs_b(env, monkeypatch, stage, epoch, extra=("--device", "cuda", "--num-workers", "2"))
