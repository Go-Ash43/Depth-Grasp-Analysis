"""Protocol A invariant (SPEC_trajectory "Implementation assertions": Delta_mask = 0 for R and A0 under
flag-only edits; SPEC_hypotheses "Protocols"; step 5 proposal rev 2 section 1, rev 3 section 5).

For each hole condition (default clean "none" and "mid"), every model present among R / A0 / M is run on
three variants of the SAME eval-mode inputs (corruption_seed, epoch 0, same slots, same order, same batch
composition), all built by GraspDataset.sample():
  base    flag_setting="deployed"
  oracle  flag_setting="oracle"       (deployed vs oracle; VACUOUS if no flag pixel changes)
  edit    deployed + flag_edit "centre_box48_flip" (deliberate, exactly EDIT_PIXELS changes per scene)
Required (else FAIL):
  - m_synth bit-identical across variants (the flag can never move a hole);
  - R / A0: x and all four output maps bit-identical across variants (torch.equal);
  - M: RGB and depth channels bit-identical; the edit changes the flag channel by exactly EDIT_PIXELS
    in every scene;
  - every model buffer (BatchNorm statistics) unchanged by the whole check.
M's output differences are REPORTED (max |delta| per map, top-1 cell changes), never required to be
non-zero: a model that ignores the flag is a valid null result, not a broken pipeline.
Edit LOCATIONS (proj_v4 5.13; every model incl. flag-blind R / A0, so an A0-only check is not vacuous): the
flag plane each variant would carry (GraspDataset.sample(..., return_flag=True), eval only, never fed to a
model) must differ between base and edit in EXACTLY the centre 48 x 48 box of every scene; for M the
flag channel of x must equal that plane and its base / edit difference must be exactly the box too.
Reused by every later evaluation (Module 2 flipped-flag settings plug in as further variants).
"""
import numpy as np
import torch

from data.dataset import MODELS, GraspDataset, collate
from data.flag_edits import EDIT_PIXELS
from eval.val_epoch import buffer_snapshot, buffers_equal

EDIT = "centre_box48_flip"
VARIANTS = {"base": ("deployed", None), "oracle": ("oracle", None), "edit": ("deployed", EDIT)}
MAPS = ("q", "cos2", "sin2", "width")


def _batches(ds, slots, batch_size):
    for i in range(0, len(slots), batch_size):
        yield collate([ds[(0, int(s))] for s in slots[i:i + batch_size]])


EDIT_BOX = 48                                     # data/flag_edits.py docstring: the specification


def edit_mask(shape):
    """Where centre_box48_flip must change the flag, built from its SPECIFICATION (rows / cols (H - 48) // 2
    ... + 47), not by calling the edit function, so an off-by-one inside the edit cannot move the expected
    mask with it."""
    if EDIT != "centre_box48_flip":
        raise ValueError(f"no specification mask for edit {EDIT!r}")
    h, w = shape
    m = np.zeros((h, w), bool)
    i0, j0 = (h - EDIT_BOX) // 2, (w - EDIT_BOX) // 2
    m[i0:i0 + EDIT_BOX, j0:j0 + EDIT_BOX] = True
    return m


def flag_plane_check(dss, slots):
    """Flag planes of base / oracle / edit per slot (eval, epoch 0). -> report: edit_locations_ok (base vs
    edit differs in exactly the edit box in every scene), changed-pixel and changed-scene counts, and the
    base planes (for the M channel check)."""
    rep = {"edit_locations_ok": True, "flag_plane_changed_px": {"oracle": 0, "edit": 0},
           "scenes_flag_plane_changed": {"oracle": 0, "edit": 0}}
    base_planes = {}
    for s in slots:
        p = {v: dss[v].sample(0, int(s), return_flag=True)["flag_plane"].numpy() for v in ("base", "oracle", "edit")}
        base_planes[int(s)] = p["base"]
        for v in ("oracle", "edit"):
            d = p["base"] != p[v]
            rep["flag_plane_changed_px"][v] += int(d.sum())
            rep["scenes_flag_plane_changed"][v] += int(d.any())
        if not np.array_equal(p["base"] != p["edit"], edit_mask(p["base"].shape)):
            rep["edit_locations_ok"] = False
    return rep, base_planes


def protocol_a_check(models, cache_dir, split, slots, *, conditions=("none", "mid"), batch_size=32,
                     device="cpu", splits=None, data_cfg=None, labels_cfg=None):
    """models: {"R"|"A0"|"M": nn.Module}. -> report dict with "verdict" PASS / FAIL and "failures"."""
    if not models or any(n not in ("R", "A0", "M") for n in models):
        raise ValueError(f"models must be a non-empty subset of R / A0 / M (Astar has no flag), got {list(models)}")
    slots = [int(s) for s in slots]
    conditions = tuple(conditions)
    if not slots or not conditions or int(batch_size) <= 0:
        raise ValueError(f"need slots, conditions and batch_size > 0 (got {len(slots)} slots, "
                         f"conditions {conditions}, batch_size {batch_size})")
    failures, report = [], {"edit": EDIT, "edit_pixels": EDIT_PIXELS[EDIT], "n_slots": len(slots),
                            "conditions": {}}
    snaps = {n: buffer_snapshot(m) for n, m in models.items()}
    modes = {n: m.training for n, m in models.items()}
    for m in models.values():
        m.eval()
    try:
        for cond in conditions:
            crep = {}
            for name, model in models.items():
                dss = {v: GraspDataset(cache_dir, split, name, mode="eval", holes=cond, flag_setting=fs,
                                       flag_edit=fe, splits=splits, data_cfg=data_cfg, labels_cfg=labels_cfg)
                       for v, (fs, fe) in VARIANTS.items()}
                crep[name] = _check_model(name, model, dss, slots, batch_size, device, failures, cond)
            report["conditions"][cond] = crep
    finally:
        for n, m in models.items():
            m.train(modes[n])
    for n, m in models.items():
        if not buffers_equal(m, snaps[n]):
            failures.append(f"{n}: a model buffer changed during the check")
    report["failures"] = failures
    report["verdict"] = "FAIL" if failures else "PASS"
    return report


def _check_model(name, model, dss, slots, batch_size, device, failures, cond):
    has_flag = "flag" in MODELS[name]
    rep = {"x_identical": {"oracle": True, "edit": True}, "outputs_identical": {"oracle": True, "edit": True},
           "m_synth_identical": True, "flag_changed_px": {"oracle": 0, "edit": 0},
           "scenes_flag_changed": {"oracle": 0, "edit": 0}, "edit_px_ok": True,
           "max_abs_output_delta": {v: {k: 0.0 for k in MAPS} for v in ("oracle", "edit")},
           "top1_cell_changes": {"oracle": 0, "edit": 0}}
    fp_rep, base_planes = flag_plane_check(dss, slots)
    rep.update(fp_rep)
    rep["flag_channel_is_plane"] = True if has_flag else None
    gens = {v: _batches(d, slots, batch_size) for v, d in dss.items()}
    n_eval = {"base": 0, "oracle": 0, "edit": 0}
    nonfinite = set()
    with torch.inference_mode():
        for base in gens["base"]:
            others = {v: next(gens[v]) for v in ("oracle", "edit")}
            out_b = model(base["x"].to(device))
            n_eval["base"] += base["x"].shape[0]
            if not all(bool(torch.isfinite(out_b[k]).all()) for k in MAPS):
                nonfinite.add("base")
            for v, ob in others.items():
                n_eval[v] += ob["x"].shape[0]
                if not torch.equal(ob["m_synth"], base["m_synth"]) or ob["slot"].tolist() != base["slot"].tolist():
                    rep["m_synth_identical"] = False
                xb, xo = base["x"], ob["x"]
                if has_flag:
                    if not torch.equal(xb[:, :-1], xo[:, :-1]):
                        rep["x_identical"][v] = False
                    per = (xb[:, -1] != xo[:, -1]).flatten(1).sum(1)
                    rep["flag_changed_px"][v] += int(per.sum())
                    rep["scenes_flag_changed"][v] += int((per > 0).sum())
                    if v == "edit" and not bool((per == EDIT_PIXELS[EDIT]).all()):
                        rep["edit_px_ok"] = False
                    if v == "edit":
                        box = torch.from_numpy(edit_mask(tuple(xb.shape[-2:])))
                        for bi, s in enumerate(base["slot"].tolist()):
                            if not torch.equal(xb[bi, -1] != xo[bi, -1], box):
                                rep["edit_locations_ok"] = False
                            if not torch.equal(xb[bi, -1] != 0, torch.from_numpy(base_planes[int(s)])):
                                rep["flag_channel_is_plane"] = False
                elif not torch.equal(xb, xo):
                    rep["x_identical"][v] = False
                out_o = model(xo.to(device))
                if not all(bool(torch.isfinite(out_o[k]).all()) for k in MAPS):
                    nonfinite.add(v)
                same = all(torch.equal(out_b[k], out_o[k]) for k in MAPS)
                if not same:
                    rep["outputs_identical"][v] = False
                for k in MAPS:
                    d = float((out_b[k] - out_o[k]).abs().max())
                    rep["max_abs_output_delta"][v][k] = max(rep["max_abs_output_delta"][v][k], d)
                cb = out_b["q"].flatten(1).argmax(1)
                co = out_o["q"].flatten(1).argmax(1)
                rep["top1_cell_changes"][v] += int((cb != co).sum())
        for v in ("oracle", "edit"):
            try:
                next(gens[v])
                rep["m_synth_identical"] = False            # variants yielded different batch counts
            except StopIteration:
                pass
    tag = f"{cond}/{name}"
    rep["n_evaluated"] = n_eval
    if any(n != len(slots) for n in n_eval.values()):
        failures.append(f"{tag}: evaluated {n_eval}, expected {len(slots)} per variant")
    if nonfinite:
        failures.append(f"{tag}: non-finite output maps under {sorted(nonfinite)}")
    if not rep["m_synth_identical"]:
        failures.append(f"{tag}: m_synth or slot order differs across flag variants")
    if not rep["edit_locations_ok"]:
        failures.append(f"{tag}: the edit did not change exactly the centre {EDIT} box of the flag")
    if rep["flag_channel_is_plane"] is False:
        failures.append(f"{tag}: M's flag channel differs from the dataset's flag plane")
    if has_flag:
        for v in ("oracle", "edit"):
            if not rep["x_identical"][v]:
                failures.append(f"{tag}: RGB/depth channels changed under the {v} flag variant")
        if not rep["edit_px_ok"]:
            failures.append(f"{tag}: the edit did not change exactly {EDIT_PIXELS[EDIT]} flag pixels per scene")
    else:
        for v in ("oracle", "edit"):
            if not rep["x_identical"][v]:
                failures.append(f"{tag}: flag-blind input x changed under the {v} flag variant")
            if not rep["outputs_identical"][v]:
                failures.append(f"{tag}: flag-blind outputs changed under the {v} flag variant")
    rep["oracle_vacuous"] = rep["flag_plane_changed_px"]["oracle"] == 0     # from the plane: every model
    return rep


def seeded_slots(ds_slots, n, seed):
    """n slots drawn without replacement by a seeded generator, sorted (all if n >= len)."""
    s = np.asarray(ds_slots, np.int64)
    if n >= len(s):
        return sorted(int(v) for v in s)
    return sorted(int(v) for v in np.random.default_rng(int(seed)).choice(s, n, replace=False))
