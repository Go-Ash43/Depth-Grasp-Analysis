"""data/lies.py: M_lie, lie_eff_224 (RULES R2), the stereo clamp (R4), and count-for-count agreement
with the frozen audit's inline definitions (scripts/audit_train.py, audit_scene)."""
import numpy as np
import pytest

from data.lies import clamp_ds, clamp_truthful_changed, deployed_224, lie_eff_224, m_lie
from data.mask_ops import resize_mask_any, resize_mask_majority
from tests.synthetic import (ABOVE_CLAMP, AUDIT_CFG, EXACT_TAU, LIE_BLOCK, SHARED_HOLE, SHARED_LIE,
                             hand_scene, load_audit_module, noisy_scene)

at = load_audit_module()
S = (224, 224)
LO, HI, TAU = 790.0, 1680.0, 12.0          # configs/frozen_day1.json


def _lie(sc, tau=TAU):
    return m_lie(sc["depth_stereo"], sc["depth_perfect"], sc["m_sensor"], tau)


def _cell(p):
    return tuple(int(v * 7 // 32) for v in p)


# ------------------------------------------------------------------ M_lie
def test_exactly_tau_is_not_a_lie():
    sc = hand_scene()
    assert _lie(sc, 12)[EXACT_TAU] == 0                 # |D_s - D*| == tau: strict '>'
    assert _lie(sc, 6)[EXACT_TAU] == 1
    assert _lie(sc, np.nextafter(np.float32(12), np.float32(0)))[EXACT_TAU] == 1


def test_holes_are_never_lies_whatever_they_hold():
    sc = hand_scene()
    lie = _lie(sc)
    assert not (lie.astype(bool) & sc["m_sensor"].astype(bool)).any()
    for fill in (-1.0, 0.0, 1e6):                       # raw sentinel, zero, absurd value at a hole
        ds = sc["depth_stereo"].copy()
        ds[sc["m_sensor"].astype(bool)] = fill
        np.testing.assert_array_equal(m_lie(ds, sc["depth_perfect"], sc["m_sensor"], TAU), lie)


def test_valid_stereo_above_clamp_is_a_lie():
    sc = hand_scene()
    assert _lie(sc)[ABOVE_CLAMP] == 1                   # 2000 vs 1450 mm


def test_hand_scene_lie_count():
    sc = hand_scene()
    # 32x32 block (+30) + shared-cell pixel (+100) + above-clamp pixel; the exact-tau pixel is not a lie
    assert int(_lie(sc, 12).sum()) == 1024 + 1 + 1
    assert int(_lie(sc, 6).sum()) == 1024 + 1 + 1 + 1


def test_m_lie_rejects_bad_input():
    sc = hand_scene()
    ds = sc["depth_stereo"].copy()
    ds[0, 0] = np.nan                                   # valid pixel, NaN
    with pytest.raises(ValueError, match="valid"):
        m_lie(ds, sc["depth_perfect"], sc["m_sensor"], TAU)
    with pytest.raises(ValueError):
        m_lie(sc["depth_stereo"], sc["depth_perfect"][:-1], sc["m_sensor"], TAU)
    for bad in (0, -1, np.nan, np.inf):
        with pytest.raises(ValueError):
            m_lie(sc["depth_stereo"], sc["depth_perfect"], sc["m_sensor"], bad)


# ------------------------------------------------------------------ lie_eff_224 (R2)
def test_shared_cell_gives_no_lie_eff():
    sc = hand_scene()
    lie = _lie(sc)
    c = _cell(SHARED_LIE)
    assert _cell(SHARED_HOLE) == c                      # the two pixels really share one 224 cell
    assert lie[SHARED_LIE] == 1 and resize_mask_any(lie, S)[c] == 1
    assert deployed_224(sc["m_sensor"], S)[c] == 1
    assert lie_eff_224(lie, sc["m_sensor"], S)[c] == 0  # same flag under deployed and oracle


def test_hand_scene_lie_eff_cells():
    sc = hand_scene()
    eff = lie_eff_224(_lie(sc), sc["m_sensor"], S)
    assert eff[_cell(ABOVE_CLAMP)] == 1 and eff[_cell(EXACT_TAU)] == 0
    assert eff[105:112, 105:112].all()                  # aligned lie block -> exactly cells 105..111
    assert int(eff.sum()) == 49 + 1                     # block + above-clamp pixel; shared cell excluded
    assert int(lie_eff_224(_lie(sc, 6), sc["m_sensor"], S).sum()) == 49 + 1 + 1


def test_fraction_rule_drops_single_pixel_lies():
    sc = hand_scene()
    frac = lie_eff_224(_lie(sc), sc["m_sensor"], S, rule="fraction")
    assert int(frac.sum()) == 49 and frac[105:112, 105:112].all()   # only the fully covered block


def test_synthetic_hole_removes_cell():
    sc = hand_scene()
    synth = np.zeros(S, np.uint8)
    synth[_cell(ABOVE_CLAMP)] = 1
    eff = lie_eff_224(_lie(sc), sc["m_sensor"], S, m_synth_224=synth)
    assert eff[_cell(ABOVE_CLAMP)] == 0 and int(eff.sum()) == 49


def test_lie_eff_rejects_overlap_and_bad_rule():
    sc = hand_scene()
    lie = _lie(sc)
    bad = lie.copy()
    bad[SHARED_HOLE] = 1                                # a "lie" on a hole pixel: upstream bug
    with pytest.raises(ValueError, match="overlaps"):
        lie_eff_224(bad, sc["m_sensor"], S)
    with pytest.raises(ValueError):
        lie_eff_224(lie, sc["m_sensor"], S, rule="majority")


# ------------------------------------------------------------------ clamp (R4)
def test_clamp_leaves_holes_alone():
    sc = hand_scene()
    hole = sc["m_sensor"].astype(bool)
    for fill in (np.nan, -1.0, 0.0):                    # a hole must stay a hole, never become 790 mm
        ds = sc["depth_stereo"].copy()
        ds[hole] = fill
        out, changed = clamp_ds(ds, sc["m_sensor"], LO, HI)
        np.testing.assert_array_equal(out[hole], ds[hole])
        assert not changed[hole].any()


def test_clamp_values():
    sc = hand_scene()
    ds = sc["depth_stereo"]
    out, changed = clamp_ds(ds, sc["m_sensor"], LO, HI)
    assert out[ABOVE_CLAMP] == HI and changed[ABOVE_CLAMP] == 1
    assert int(changed.sum()) == 1
    keep = ~changed.astype(bool) & ~sc["m_sensor"].astype(bool)
    np.testing.assert_array_equal(out[keep], ds[keep])  # in-range values bit-identical
    assert out.dtype == ds.dtype and ds[ABOVE_CLAMP] == 2000.0   # input not modified
    low = ds.copy()
    low[0, 0] = 600.0
    assert clamp_ds(low, sc["m_sensor"], LO, HI)[0][0, 0] == LO


def test_clamp_rejects_bad_input():
    sc = hand_scene()
    ds = sc["depth_stereo"].copy()
    ds[0, 0] = np.inf
    with pytest.raises(ValueError, match="valid"):
        clamp_ds(ds, sc["m_sensor"], LO, HI)
    with pytest.raises(ValueError):
        clamp_ds(sc["depth_stereo"], sc["m_sensor"], HI, LO)


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("tau", [6.0, 12.0, 24.0])
def test_m_lie_unchanged_by_clamp(seed, tau):
    # D* in 1390..1520 mm lies inside [LO + 2 tau, HI - 2 tau]: every clamped pixel was > 2 tau off
    # and stays > 2 tau off, so computing M_lie from clamped D_s would not change it (R4 margin).
    sc = noisy_scene(seed)
    clamped, changed = clamp_ds(sc["depth_stereo"], sc["m_sensor"], LO, HI)
    assert changed.sum() > 0
    raw = m_lie(sc["depth_stereo"], sc["depth_perfect"], sc["m_sensor"], tau)
    np.testing.assert_array_equal(m_lie(clamped, sc["depth_perfect"], sc["m_sensor"], tau), raw)
    assert clamp_truthful_changed(sc["depth_stereo"], sc["depth_perfect"], sc["m_sensor"], LO, HI, tau) == 0


def test_clamp_truthful_changed_detects_a_truthful_pixel():
    z = np.zeros((8, 8), np.uint8)
    dstar = np.full((8, 8), 1675.0, np.float32)         # D* near the top of the range
    ds = dstar.copy()
    ds[2, 2] = 1685.0                                   # 10 mm off (<= 2 tau): truthful, but clamped
    ds[3, 3] = 2000.0                                   # a lie: clamping it is fine
    assert clamp_truthful_changed(ds, dstar, z, LO, HI, TAU) == 1
    assert clamp_truthful_changed(hand_scene()["depth_stereo"], hand_scene()["depth_perfect"],
                                  hand_scene()["m_sensor"], LO, HI, TAU) == 0


def test_clamp_truthful_changed_rejects_nonfinite_dstar():
    # NaN D* next to a clamped valid pixel would compare False and report 0 (a silent pass)
    z = np.zeros((4, 4), np.uint8)
    ds = np.full((4, 4), 1685.0, np.float32)
    dstar = np.full((4, 4), 1675.0, np.float32)
    dstar[0, 0] = np.nan
    with pytest.raises(ValueError, match="perfect depth"):
        clamp_truthful_changed(ds, dstar, z, LO, HI, TAU)


# ------------------------------------------------------------------ same numbers as the frozen audit
@pytest.mark.parametrize("make", [hand_scene, lambda: noisy_scene(0), lambda: noisy_scene(1)],
                         ids=["hand", "noisy0", "noisy1"])
def test_matches_audit_scene(make):
    sc = make()
    row = at.audit_scene(sc, at.make_hists(AUDIT_CFG), AUDIT_CFG)
    assert row["status"] == "ok"
    obj, obj224 = sc["obj_mask"], resize_mask_majority(sc["obj_mask"], S)
    for t in AUDIT_CFG["taus"]:
        k = f"{t:g}"
        lie = m_lie(sc["depth_stereo"], sc["depth_perfect"], sc["m_sensor"], t).astype(bool)
        eff = lie_eff_224(lie, sc["m_sensor"], S).astype(bool)
        frac = lie_eff_224(lie, sc["m_sensor"], S, rule="fraction").astype(bool)
        assert row[f"lie_img_1024@{k}"] == lie.sum()
        assert row[f"lie_obj_1024@{k}"] == (lie & obj).sum()
        assert row[f"lie_eff_img_224@{k}"] == eff.sum()
        assert row[f"lie_eff_obj_224@{k}"] == (eff & obj224).sum()
        assert row[f"lie_frac_obj_224@{k}"] == (frac & obj224).sum()


# ------------------------------------------------------------------ worker-side R2 from the 224 cache
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("rule", ["any", "fraction"])
def test_lie_eff_from_224_equals_native(seed, rule):
    """Cached resize of M_lie, minus the deployed flag (incl. a random M_synth), == lie_eff_224."""
    from data.lies import lie_eff_from_224
    from data.mask_ops import resize_mask_fraction
    sc = noisy_scene(seed)
    lie = _lie(sc)
    synth = (np.random.default_rng(seed).random(S) < 0.05).astype(np.uint8)
    cached = resize_mask_any(lie, S) if rule == "any" else resize_mask_fraction(lie, S, 0.5)
    dep = deployed_224(sc["m_sensor"], S, synth)
    np.testing.assert_array_equal(lie_eff_from_224(cached, dep),
                                  lie_eff_224(lie, sc["m_sensor"], S, synth, rule=rule))


def test_lie_eff_from_224_rejects_bad_masks():
    from data.lies import lie_eff_from_224
    with pytest.raises(ValueError):
        lie_eff_from_224(np.full(S, 255, np.uint8), np.zeros(S, np.uint8))
    with pytest.raises(ValueError):
        lie_eff_from_224(np.zeros(S, np.uint8), np.zeros((10, 10), np.uint8))
