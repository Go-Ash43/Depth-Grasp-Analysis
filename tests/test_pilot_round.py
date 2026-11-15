"""Pilot round (proj_v4 section 5.13): the two arm configs, q_reduction in the real training loop, the new
validation columns, edit-location checks for flag-blind models, and the pre-registered pilot analysis."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import training.runcfg as rc
from data.dataset import GraspDataset
from data.flag_edits import FLAG_EDITS
from data.labels import load_labels_cfg, targets_from_cfg
from eval.invariants import protocol_a_check
from eval.val_epoch import classify_sample
from models.grconv_lite import build_model
from models.losses import TERMS, grasp_loss
import hashlib
import sys

import data.flag_edits as fe
from eval.invariants import edit_mask
from scripts import train as tr
from scripts.pilot_compare import (check_csv_sha, check_runs, decide_budget, decide_loss, miss_table,
                                   narrow_share, object_bootstrap)
from tests.test_dataset import DCFG, SPLITS, cache  # noqa: F401  (fixture)
from tests.test_recovery import _argv, env  # noqa: F401  (fixture)
from tests.test_runcfg import restore_precision  # noqa: F401  (fixture)
from training.loop import train_epoch

REPO = Path(__file__).resolve().parents[1]
W = {"q": 1.0, "cos2": 1.0, "sin2": 1.0, "width": 1.0}
ALL, SPLIT = (next(t for t, v in rc.Q_REDUCTION_TEXT.items() if v == x) for x in ("all", "split"))


# ---------------------------------------------------------------- configs
PILOT_E = 7                                                          # proj_v6 6.7 (commit 2)
SQ_ARM_SHA256 = "8d4467c240af7df41e4702ab792229fe18c69132ea13d652cfb238bbdf47a64c"   # train_v1_sq.json at the pilot


def test_sq_arm_config_is_the_registered_pilot_arm():
    """The SQ pilot arm stays byte-identical to what the pilot runs recorded (proj_v4 5.13, proj_v6 6.7)."""
    p = REPO / "configs" / "train_v1_sq.json"
    assert rc.file_sha256(p) == SQ_ARM_SHA256
    b = json.loads(p.read_text())
    assert (b["status"], b["epochs"], b["val_conditions"], b["loss"]["q_reduction"]) == ("draft", None, ["none"], SPLIT)


def test_frozen_config_is_a_pilot_arm_plus_permitted_changes():
    """proj_v6 7.6: frozen = the winning arm (config = SQ arm with ALL) + status, epochs, val_conditions."""
    f, _ = rc.load_train_cfg()
    assert f["status"] == "frozen" and f["epochs"] in (PILOT_E, 2 * PILOT_E)
    assert f["val_conditions"] == ["none", "mid"]                                    # item 7 (a)
    assert f["loss"]["q_reduction"] in (ALL, SPLIT)
    winner = json.loads((REPO / "configs" / "train_v1_sq.json").read_text())
    winner["loss"]["q_reduction"] = f["loss"]["q_reduction"]
    g = copy.deepcopy(f)
    g.update(status="draft", epochs=None, val_conditions=["none"])
    assert g == winner


def test_unknown_q_reduction_text_is_refused():
    c, _ = rc.load_train_cfg()
    c = copy.deepcopy(c)
    c["loss"]["q_reduction"] = "split"                                               # code word, not the text
    with pytest.raises(ValueError, match="q_reduction"):
        rc.validate_train_cfg(c)


# ---------------------------------------------------------------- training loop
class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.c = torch.nn.Conv2d(1, 4, 1)

    def forward(self, x):
        y = self.c(x)
        return {k: y[:, i:i + 1] for i, k in enumerate(TERMS)}


def batches(n=3, b=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        bt = {k: torch.rand(b, 1, 224, 224, generator=g) for k in ("x",) + TERMS}
        bt["void"] = torch.rand(b, 1, 224, 224, generator=g) < 0.9
        bt.update({"severity": ["clean"] * b, "hole_frac": torch.zeros(b), "covered_lie_frac": torch.zeros(b)})
        out.append(bt)
    return out


def _opt(model):
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)


def via_loop(text, progress=0, log=None, bts=None):
    m = Tiny()
    opt, sch = _opt(m)
    cfg = {"loss": {"weights": W, "beta": 1.0, "q_reduction": text}}
    train_epoch(m, bts if bts is not None else batches(), opt, sch, "cpu", cfg, progress_every=progress,
                log=log or (lambda s: None))
    return m


def manual(**kw):
    m = Tiny()
    opt, sch = _opt(m)
    m.train()
    for bt in batches():
        loss, _ = grasp_loss(m(bt["x"]), {k: bt[k] for k in TERMS + ("void",)}, W, 1.0, **kw)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sch.step()
    return m


def same(a, b):
    return all(torch.equal(p, q) for p, q in zip(a.state_dict().values(), b.state_dict().values()))


def test_config_loss_training_is_bit_identical_to_the_previous_call():
    assert same(via_loop(ALL), manual())                     # previous call: no q_reduction argument


def test_split_reaches_the_loss_in_the_real_loop():
    s = via_loop(SPLIT)
    assert same(s, manual(q_reduction="split"))
    assert not same(s, via_loop(ALL))


def test_progress_lines_do_not_change_training():
    lines = []
    m = via_loop(ALL, progress=1, log=lines.append)
    assert same(m, via_loop(ALL)) and len(lines) == 3 and "samples/s" in lines[0]
    assert "monitor loss (Q pooled over all pixels)" in lines[0]


def test_window_max_shows_an_injected_spike():
    bts = batches()
    bts[1]["q"] = bts[1]["q"] + 50.0                         # one spike step (step 2 of 3)
    lines = []
    via_loop(ALL, progress=3, log=lines.append, bts=bts)
    assert len(lines) == 1 and "Q window steps 1-3" in lines[0] and "@ step 2" in lines[0]
    lines = []
    via_loop(ALL, progress=1, log=lines.append, bts=batches())  # windows reset every line
    assert [("Q window steps %d-%d" % (i, i)) in ln for i, ln in enumerate(lines, 1)] == [True] * 3


# ---------------------------------------------------------------- console.log captures failures
def test_crash_traceback_reaches_console_log(env, monkeypatch):  # noqa: F811
    cache_dir, man, tmp = env

    def boom(s, e):
        if (s, e) == ("after_train", 0):
            raise RuntimeError("injected crash xyz")
    monkeypatch.setattr(tr, "_crash_hook", boom)
    with pytest.raises(RuntimeError):
        tr.main(_argv(cache_dir, man, tmp / "a", epochs=1), splits=SPLITS)
    log = (tmp / "a" / "console.log").read_text()
    assert "Traceback" in log and "injected crash xyz" in log
    assert not isinstance(sys.stdout, tr._Tee) and not isinstance(sys.stderr, tr._Tee)   # restored


def test_systemexit_message_reaches_console_log(env, monkeypatch):  # noqa: F811
    cache_dir, man, tmp = env

    def refuse(s, e):
        if (s, e) == ("after_train", 0):
            raise SystemExit("injected refusal abc")
    monkeypatch.setattr(tr, "_crash_hook", refuse)
    with pytest.raises(SystemExit):
        tr.main(_argv(cache_dir, man, tmp / "b", epochs=1), splits=SPLITS)
    assert "SystemExit: injected refusal abc" in (tmp / "b" / "console.log").read_text()


# ---------------------------------------------------------------- validation columns
LCFG = load_labels_cfg()
GT = np.array([[512.0, 512.0, 30.0, 80.0, 40.0]])


def maps():
    t = targets_from_cfg(GT, LCFG)
    tgt = {k: np.asarray(t[k], np.float32) for k in TERMS}
    tgt["void"] = np.asarray(t["void"]).astype(np.uint8)
    return {k: tgt[k].copy() for k in TERMS}, tgt


def test_new_columns_perfect_prediction():
    pred, tgt = maps()
    r = classify_sample(pred, tgt, GT, LCFG)
    assert r["tt_match_primary"] and r["cfn_match_primary"] and r["cf_match_primary"]
    assert r["target_opening_224"] == pytest.approx(80.0 * 224 / 1024, abs=1e-3)


def test_counterfactual_kept_for_invalid_decodes_and_cf_unchanged():
    pred, tgt = maps()
    pred["width"][:] = -0.1
    r = classify_sample(pred, tgt, GT, LCFG)
    assert r["valid"] is False and r["cat_primary"] == "invalid_decode"
    assert r["cf_match_primary"] is None                                  # original eligibility unchanged
    assert r["cfn_match_primary"] is True                                 # non-void: evaluated anyway


def test_void_pick_has_no_opening_or_counterfactual():
    pred, tgt = maps()
    pred["q"][0, 0] = 5.0
    pred["cos2"][0, 0], pred["width"][0, 0] = 1.0, 0.2                 # a VALID decode in a void cell
    r = classify_sample(pred, tgt, GT, LCFG)
    assert r["valid"] and r["target_void"] and r["target_opening_224"] is None
    assert r["cfn_match_primary"] is None and r["cf_match_primary"] is None and r["cat_primary"] == "void_cell"
    assert r["tt_match_primary"] is True


# ---------------------------------------------------------------- Protocol A: edit locations
def _slots(out):
    return [int(s) for s in GraspDataset(out, "train", "M", mode="eval", splits=SPLITS, data_cfg=DCFG).slots]


def _check(out, models):
    return protocol_a_check(models, out, "train", _slots(out), splits=SPLITS, data_cfg=DCFG, batch_size=2)


def test_a0_only_check_verifies_the_edit_locations(cache):  # noqa: F811
    rep = _check(cache, {"A0": build_model("A0", 0), "M": build_model("M", 0)})
    assert rep["verdict"] == "PASS", rep["failures"]
    n = len(_slots(cache))
    for cond in ("none", "mid"):
        a0 = rep["conditions"][cond]["A0"]
        assert a0["edit_locations_ok"] and a0["flag_plane_changed_px"]["edit"] == 2304 * n
        assert isinstance(a0["oracle_vacuous"], bool)
        assert rep["conditions"][cond]["M"]["flag_channel_is_plane"] is True


def test_spec_mask_and_edit_function_agree_both_ways():
    for shape in ((224, 224), (100, 120), (49, 51)):
        m = edit_mask(shape)
        assert m.sum() == 2304 and np.array_equal(m, fe.centre_box48_flip(np.zeros(shape, bool)))


def test_a0_only_check_fails_when_the_edit_moves(cache, monkeypatch):  # noqa: F811
    def shifted(flag):                                                 # same 2,304 px, wrong place
        out = np.asarray(flag).astype(bool, copy=True)
        out[:48, :48] = ~out[:48, :48]
        return out
    monkeypatch.setitem(FLAG_EDITS, "centre_box48_flip", shifted)     # the lookup the dataset uses
    monkeypatch.setattr(fe, "centre_box48_flip", shifted)              # and the function itself
    rep = _check(cache, {"A0": build_model("A0", 0)})
    assert rep["verdict"] == "FAIL" and any("box" in f for f in rep["failures"])


def test_return_flag_is_eval_and_flag_capable_only(cache):  # noqa: F811
    ds = GraspDataset(cache, "train", "Astar", mode="eval", splits=SPLITS, data_cfg=DCFG)
    with pytest.raises(ValueError, match="return_flag"):
        ds.sample(0, int(ds.slots[0]), return_flag=True)
    a0 = GraspDataset(cache, "train", "A0", mode="eval", splits=SPLITS, data_cfg=DCFG)
    s = a0.sample(0, int(a0.slots[0]), return_flag=True)
    assert s["flag_plane"].dtype == torch.bool and "flag_plane" not in a0[(0, int(a0.slots[0]))]


# ---------------------------------------------------------------- pilot analysis
def test_bootstrap_basics():
    a, b = [1] * 6, [1] * 6
    r = object_bootstrap(a, b, list("aabbcc"), resamples=200)
    assert r["diff_pp"] == 0.0 and r["ci95"] == [0.0, 0.0] and r["n_objects"] == 3
    r = object_bootstrap([1] * 6, [0] * 6, list("aabbcc"), resamples=200)
    assert r["diff_pp"] == 100.0 and r["ci95"] == [100.0, 100.0]
    x = object_bootstrap([1, 0, 1, 1, 0, 1], [0, 0, 1, 0, 1, 1], list("aabbcc"), resamples=500)
    assert x == object_bootstrap([1, 0, 1, 1, 0, 1], [0, 0, 1, 0, 1, 1], list("aabbcc"), resamples=500)
    with pytest.raises(ValueError):
        object_bootstrap([1], [1, 0], ["a"])


def test_decisions_are_as_registered():
    assert decide_loss({"diff_pp": 1.0, "ci95": [0.2, 1.8]}) == "CONFIG_WINS"
    assert decide_loss({"diff_pp": 1.0, "ci95": [-0.1, 2.0]}) == "SQ"
    assert decide_loss({"diff_pp": -2.0, "ci95": [-3.0, -1.0]}) == "SQ"
    assert decide_loss({"diff_pp": 0.0, "ci95": [0.0, 0.0]}) == "SQ"                 # tie -> SQ
    assert decide_budget(None) == "PENDING"
    assert decide_budget({"diff_pp": 1.2, "ci95": [0.3, 2.0]}) == "BUDGET_TOO_SHORT"
    assert decide_budget({"diff_pp": 0.9, "ci95": [0.1, 1.7]}) == "E"
    assert decide_budget({"diff_pp": 1.5, "ci95": [-0.2, 3.0]}) == "E"


def vrow(match, cat="", void=False, opening="10.0"):
    return {"match_primary": str(match), "cat_primary": cat, "target_void": str(void),
            "target_opening_224": "" if void else opening}


def test_narrow_share_and_miss_table():
    rows = {1: vrow(True), 2: vrow(False, "target_at_cell_miss", opening="1.3"),
            3: vrow(False, "regression_miss", opening="2.0"), 4: vrow(False, "void_cell", void=True),
            5: vrow(False, "regression_miss", opening="6.0")}
    t = miss_table(rows, "primary")
    assert t["misses"] == 4 and t["by_bin"]["<4"] == {"target_at_cell_miss": 1, "regression_miss": 1}
    assert t["by_bin"]["void"] == {"void_cell": 1} and t["by_bin"]["4-8"] == {"regression_miss": 1}
    n = narrow_share(rows)
    assert (n["narrow"], n["void_picks"], n["share"], n["decision"]) == (2, 1, 0.5, "OK")      # > 0.5 needed
    rows[5] = vrow(False, "regression_miss", opening="3.9")
    assert narrow_share(rows)["decision"] == "NARROW_DOMINATES"


def arm(text, epochs=4, **kw):
    r = {"kind": "pilot", "model": "A0", "seed": 1, "batch_size": 32, "n_train": 100, "train_slots_sha256": "t",
         "val_split": "val", "git_commit": "c", "manifest_sha256": "m", "labels_sha256": "l",
         "train_data_sha256": "d", "holes": "train", "flag_setting": "deployed", "epochs": epochs,
         "config": {"loss": {"q_reduction": text}}}
    r.update(kw)
    return r, {"completed_epochs": r["epochs"]}


def test_check_runs_pairing():
    check_runs(arm(ALL), arm(SPLIT), arm(SPLIT, epochs=8))
    check_runs(arm(ALL), arm(SPLIT), None)
    bad = [(arm(SPLIT), arm(SPLIT), None), (arm(ALL), arm(SPLIT, epochs=5), None),
           (arm(ALL), arm(SPLIT, seed=2), None), (arm(ALL), arm(SPLIT), arm(SPLIT, epochs=6)),
           (arm(ALL, kind="smoke"), arm(SPLIT, kind="smoke"), None)]
    for c, s, t in bad:
        with pytest.raises(ValueError):
            check_runs(c, s, t)
    r, f = arm(SPLIT)
    with pytest.raises(ValueError, match="not finished"):
        check_runs(arm(ALL), (r, {"completed_epochs": 2}), None)


def test_check_runs_compares_the_recorded_configs():
    lr2 = {"loss": {"q_reduction": SPLIT}, "optim": {"lr": 2e-3}}
    with pytest.raises(ValueError, match="beyond loss.q_reduction"):
        check_runs(arm(ALL), arm(SPLIT, config=lr2), None)
    other = {"loss": {"q_reduction": SPLIT}, "model": {"c": 32}}
    with pytest.raises(ValueError, match="SQ-2E"):
        check_runs(arm(ALL), arm(SPLIT), arm(SPLIT, epochs=8, config=other))


def test_csv_sha_must_match_the_run_log(tmp_path):
    p = tmp_path / "epoch_003.csv"
    p.write_text("condition,slot\nnone,1\n")
    check_csv_sha(p, hashlib.sha256(p.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="csv_sha256"):
        check_csv_sha(p, "0" * 64)
