"""Enforced metric promotion (proj_v6 7.5, item 7): the registered policy text, the exact > 95 % boundary,
refusals for anything that is not production A0 seed 1's complete final checkpoint under the frozen config,
and tamper checks on the recorded decision. Synthetic run directories only (no GPU, no data)."""
import json
from pathlib import Path

import pytest

import training.runcfg as rc
from eval import metric_policy as mp
from eval.val_epoch import CSV_FIELDS, rows_to_csv_bytes
from scripts import metric_decision as md
from training.checkpoint import append_log, atomic_write_bytes, atomic_write_json, ckpt_path, csv_path

REPO = Path(__file__).resolve().parents[1]
N = 10_000
EPOCHS = 2


def frozen_cfg(tmp_path, **edit):
    cfg = json.loads((REPO / "configs" / "train_v1.json").read_text())
    cfg.update({"status": "frozen", "epochs": EPOCHS, "val_conditions": ["none", "mid"]}, **edit)
    p = tmp_path / "train_frozen.json"
    p.write_text(json.dumps(cfg, indent=1))
    return p


def _rows(matches, strict, n=N):
    out = []
    for cond, mp_, ms_ in (("none", matches, strict), ("mid", 0, 0)):     # mid rows never count
        for s in range(n):
            r = {k: "" for k in CSV_FIELDS}
            r.update(condition=cond, slot=s, match_primary=s < mp_, match_strict=s < ms_)
            out.append(r)
    return out


def _val(matches, strict, n=N):
    return {"none": {"condition": "none", "n": n, "matches_primary": matches, "top1_primary": matches / n,
                     "matches_strict": strict, "top1_strict": strict / n},
            "mid": {"condition": "mid", "n": n, "matches_primary": 0, "top1_primary": 0.0,
                    "matches_strict": 0, "top1_strict": 0.0}}


def make_run(tmp_path, cfg_path, matches=9501, strict=8000, csv_matches=None, name="run", log_epochs=EPOCHS,
             final=True, **rc_edit):
    """A complete synthetic production A0 seed-1 run; keyword edits override run_config fields."""
    d = tmp_path / name
    cfg = json.loads(Path(cfg_path).read_text())
    rcfg = {"run_id": "r1", "kind": "production", "model": "A0", "seed": 1, "epochs": EPOCHS, "git_dirty": False,
            "git_commit": "c0ffee", "config_sha256": rc.file_sha256(cfg_path), "config": cfg,
            "val_conditions": ["mid", "none"], "n_val": N}
    rcfg.update(rc_edit)
    atomic_write_json(d / "run_config.json", rcfg)
    cm = matches if csv_matches is None else csv_matches
    for e in range(log_epochs):
        csv_sha = atomic_write_bytes(csv_path(d, e), rows_to_csv_bytes(_rows(cm, strict)))
        ck_sha = atomic_write_bytes(ckpt_path(d, e), f"weights {e}".encode())
        append_log(d, {"run_id": "r1", "epoch": e, "ckpt_sha256": ck_sha, "csv_sha256": csv_sha,
                       "val": _val(matches, strict), "continuation_exact": True})
    if final:
        atomic_write_json(d / "final.json", {"run_id": "r1", "completed_epochs": EPOCHS, "selection": "final_epoch",
                                             "final_checkpoint": str(ckpt_path(d, EPOCHS - 1)),
                                             "final_val": _val(matches, strict)})
    return d


@pytest.fixture
def cfgp(tmp_path):
    return frozen_cfg(tmp_path)


# ---------------------------------------------------------------- policy text and boundary
def test_policy_text_is_the_config_text_and_variants_are_refused():
    cfg, _ = rc.load_train_cfg()
    assert mp.parse_policy(cfg["metric_promotion"])["promote_to"] == "strict"
    for bad in (mp.POLICY_TEXT + " ", mp.POLICY_TEXT.replace("> 0.95", ">= 0.95"), None, ""):
        with pytest.raises(mp.PolicyError):
            mp.parse_policy(bad)


def test_boundary_is_strictly_greater_and_exact():
    assert mp.promote(9501, 10_000) and not mp.promote(9500, 10_000)
    assert mp.promote(20, 21) and not mp.promote(19, 20)          # 95.24 % vs exactly 95 %
    for bad in ((-1, 10), (11, 10), (1, 0), (9.5, 10), (True, 10)):
        with pytest.raises(mp.PolicyError):
            mp.promote(*bad)


def test_9501_promotes_to_strict(tmp_path, cfgp):
    rec = mp.decide(make_run(tmp_path, cfgp, matches=9501), cfgp)
    assert rec["metric"] == "strict" and rec["promoted"] is True
    assert rec["clean"]["n"] == N and rec["clean"]["matches_primary"] == 9501      # mid rows not counted


def test_exactly_95_stays_primary(tmp_path, cfgp):
    rec = mp.decide(make_run(tmp_path, cfgp, matches=9500), cfgp)
    assert rec["metric"] == "primary" and rec["promoted"] is False


# ---------------------------------------------------------------- refusals
@pytest.mark.parametrize("edit, msg", [
    ({"kind": "pilot"}, "kind"), ({"model": "M"}, "model"), ({"seed": 2}, "seed"),
    ({"git_dirty": True}, "git_dirty"), ({"git_dirty": None}, "git_dirty"),
    ({"config_sha256": "0" * 64}, "config_sha256"), ({"epochs": 3}, "epochs"),
    ({"val_conditions": ["mid"]}, "clean"), ({"n_val": N - 1}, "clean n"),
])
def test_wrong_run_is_refused(tmp_path, cfgp, edit, msg):
    with pytest.raises(mp.PolicyError, match=msg):
        mp.decide(make_run(tmp_path, cfgp, **edit), cfgp)


def test_draft_config_is_refused(tmp_path):
    p = frozen_cfg(tmp_path, status="draft")
    with pytest.raises(mp.PolicyError, match="frozen"):
        mp.decide(make_run(tmp_path, p), p)


def test_config_edited_after_the_run_is_refused(tmp_path, cfgp):
    d = make_run(tmp_path, cfgp)
    cfg = json.loads(cfgp.read_text())
    cfg["batch_size"] = 16
    cfgp.write_text(json.dumps(cfg))
    with pytest.raises(mp.PolicyError, match="config"):
        mp.decide(d, cfgp)


def test_incomplete_run_is_refused(tmp_path, cfgp):
    with pytest.raises(mp.PolicyError, match="final.json"):
        mp.decide(make_run(tmp_path, cfgp, name="nofinal", final=False), cfgp)
    with pytest.raises(mp.PolicyError, match="epochs"):
        mp.decide(make_run(tmp_path, cfgp, name="short", log_epochs=1), cfgp)


def test_torn_log_tail_is_refused(tmp_path, cfgp):
    d = make_run(tmp_path, cfgp)
    with open(d / "log.jsonl", "a") as fh:
        fh.write('{"run_id": "r1", "ep')
    with pytest.raises(mp.PolicyError, match="torn"):
        mp.decide(d, cfgp)


def test_final_val_must_equal_last_log_record(tmp_path, cfgp):
    d = make_run(tmp_path, cfgp)
    fin = json.loads((d / "final.json").read_text())
    fin["final_val"]["none"]["matches_primary"] = 9600
    atomic_write_json(d / "final.json", fin)
    with pytest.raises(mp.PolicyError, match="final_val"):
        mp.decide(d, cfgp)


def test_csv_recount_must_agree_with_the_log(tmp_path, cfgp):
    d = make_run(tmp_path, cfgp, matches=9501, csv_matches=9500)     # log says 9501, CSV (sha-consistent) 9500
    with pytest.raises(mp.PolicyError, match="CSV 9500"):
        mp.decide(d, cfgp)


def test_changed_checkpoint_or_csv_is_refused(tmp_path, cfgp):
    d = make_run(tmp_path, cfgp)
    ckpt_path(d, EPOCHS - 1).write_bytes(b"other weights")
    with pytest.raises(mp.PolicyError, match="sha256"):
        mp.decide(d, cfgp)


# ---------------------------------------------------------------- recorded decision
def _write(tmp_path, d, cfgp, out="dec"):
    assert md.main(["--a0-run", str(d), "--config", str(cfgp), "--out", str(tmp_path / out)]) == 0
    return tmp_path / out / md.OUT_NAME


def test_script_round_trip_and_idempotent_rerun(tmp_path, cfgp):
    d = make_run(tmp_path, cfgp, matches=9501)
    p = _write(tmp_path, d, cfgp)
    assert mp.load_decision(p)["metric"] == "strict"
    first = p.read_bytes()
    _write(tmp_path, d, cfgp)                                     # identical: no-op
    assert p.read_bytes() == first


def test_script_refuses_a_different_decision(tmp_path, cfgp):
    _write(tmp_path, make_run(tmp_path, cfgp, matches=9501, name="a"), cfgp)
    b = make_run(tmp_path, cfgp, matches=9500, name="b")
    with pytest.raises(SystemExit, match="different decision"):
        md.main(["--a0-run", str(b), "--config", str(cfgp), "--out", str(tmp_path / "dec")])


def test_script_refuses_a_bad_run(tmp_path, cfgp):
    d = make_run(tmp_path, cfgp, model="R")
    with pytest.raises(SystemExit, match="refused"):
        md.main(["--a0-run", str(d), "--config", str(cfgp), "--out", str(tmp_path / "dec")])
    assert not (tmp_path / "dec").exists()


@pytest.mark.parametrize("field, value", [("metric", "primary"), ("promoted", False), ("policy", "x"),
                                          ("decision_version", 2)])
def test_tampered_decision_is_refused(tmp_path, cfgp, field, value):
    p = _write(tmp_path, make_run(tmp_path, cfgp, matches=9501), cfgp)
    rec = json.loads(p.read_text())
    rec[field] = value
    p.write_text(json.dumps(rec))
    with pytest.raises(mp.PolicyError):
        mp.load_decision(p)


def test_tampered_counts_fail_the_recheck(tmp_path, cfgp):
    p = _write(tmp_path, make_run(tmp_path, cfgp, matches=9501), cfgp)
    rec = json.loads(p.read_text())
    rec["clean"]["matches_primary"] = 9502                        # still "strict", but not what the run says
    p.write_text(json.dumps(rec))
    mp.load_decision(p, recheck=False)                            # internally consistent ...
    with pytest.raises(mp.PolicyError, match="re-derived"):
        mp.load_decision(p)                                       # ... but the re-derivation catches it


def test_files_changed_after_the_decision_are_refused(tmp_path, cfgp):
    d = make_run(tmp_path, cfgp, matches=9501)
    p = _write(tmp_path, d, cfgp)
    with open(csv_path(d, EPOCHS - 1), "ab") as fh:
        fh.write(b"\n")
    with pytest.raises(mp.PolicyError, match="changed since the decision"):
        mp.load_decision(p, recheck=False)
