"""Synthetic holes M_synth for training and for H1's independently placed evaluation holes
(docs/proj_v3.md section 2 D1, and the "cover, don't reject" holes decision, notes/holes_decision_v1.md).

One sample = 1-3 whole ellipses (aspect 1-3, any orientation), centred on object cells:

  centres            object & ~M_sensor                 (NO lie mask: placement never looks at M_lie)
  lie cells          IGNORED. An ellipse that covers lie cells keeps them: they become part of M_synth,
                     so they are flagged and completed like the rest of the hole. Nothing is clipped,
                     so no unflagged island can trace M_lie, and since neither centres nor shapes depend
                     on M_lie, hole placement is statistically independent of it.
                     DEVIATION from SPEC_hypotheses (M_synth row, "placed only on correct valid
                     pixels"), recorded in docs/proj_v3.md: rejection made placement depend on M_lie
                     (85-97 % of ellipses rejected, p10 achieved fraction 0 at every severity). The
                     covered lie fraction is logged per sample by the dataset (lie_any@12 in M_synth).
  budget             new hole cells ON the object, M_sensor excluded, / object cells
  cap                at least min_valid_obj_frac of the object's cells stay valid after
                     M_sensor | M_synth; a budget above the cap is truncated and recorded.

So M_synth == (union of the accepted whole ellipses) & ~M_sensor, exactly, and this function takes no
lie input at all (independence by construction).

Sizing is exact, not iterative: for a sampled centre, aspect r and orientation phi every cell gets the
ellipse "radius" rho = sqrt(p^2 / r + q^2 * r) (p, q = offset along / across phi); the ellipse
{rho <= u} has semi-axes u*sqrt(r), u/sqrt(r). u is set to the n-th smallest rho among the cells that
would be NEW on-object hole cells, so the ellipse adds n of them (plus exact ties, trimmed back if
that would pass the cap). Pixel convention: integer = cell centre (data/geometry.py).

Severities at evaluation share one stream per scene (the dataset key has no severity), so low, mid and
high start from the same centre / aspect / orientation draws and are roughly nested. Deliberate: it
pairs the severity comparison (H1 severity trend).

Nothing is silent: every call returns a record with target, cap, achieved, the accepted ellipses as
(cx, cy, aspect, phi, u) and, if the budget was not met within tolerance, the cause:
cap / no_eligible_centre / tie_cap / under_tolerance / over_tolerance.
"""
import numpy as np

from data.mask_ops import _as_mask


def ellipse_mask(hw, cx, cy, aspect, phi, u):
    """Cells of one whole ellipse as recorded in rec["ellipses"] (for tests and overlays)."""
    return _rho(hw, cx, cy, aspect, phi) <= u


def _rho(hw, cx, cy, aspect, phi):
    yy, xx = np.mgrid[0:hw[0], 0:hw[1]]
    dx, dy = xx - cx, yy - cy
    c, s = np.cos(phi), np.sin(phi)
    p = dx * c + dy * s
    q = -dx * s + dy * c
    return np.sqrt(p * p / aspect + q * q * aspect)


def synth_holes(rng, m_sensor, obj, budget_frac, cfg):
    """-> (m_synth uint8 (H, W), record dict). cfg = configs/train_data.json["holes"].

    rng: a np.random.Generator used for nothing else (the dataset's holes stream).
    budget_frac: fraction of the object's cells to turn into new hole cells (0 -> no holes).
    Takes no lie mask: placement is independent of M_lie by construction."""
    ms = _as_mask(m_sensor, name="m_sensor")
    ob = _as_mask(obj, ms.shape, "obj")
    hw = ms.shape
    lo, hi = (int(v) for v in cfg["n_ellipses"])
    a_lo, a_hi = (float(v) for v in cfg["aspect"])
    tol, tries = float(cfg["budget_tol"]), int(cfg["max_tries_per_ellipse"])
    if not (1 <= lo <= hi and 1.0 <= a_lo <= a_hi and 0 <= tol < 1 and tries >= 1
            and 0 < float(cfg["min_valid_obj_frac"]) <= 1):
        raise ValueError(f"invalid holes cfg {cfg}")
    if not 0 <= budget_frac < 1:
        raise ValueError(f"budget_frac must be in [0, 1), got {budget_frac}")

    n_obj = int(ob.sum())
    target = float(budget_frac) * n_obj
    cap = int(np.floor((1.0 - float(cfg["min_valid_obj_frac"])) * n_obj)) - int((ms & ob).sum())
    rec = {"budget_frac": float(budget_frac), "n_obj": n_obj, "target": target, "cap": max(cap, 0),
           "n_ellipses_planned": 0, "n_ellipses": 0, "ellipses": [], "tie_retries": 0, "achieved": 0,
           "achieved_frac": 0.0, "within_tol": budget_frac == 0, "shortfall_cause": None}
    union = np.zeros(hw, bool)
    if budget_frac == 0:
        return union.astype(np.uint8), rec
    if n_obj == 0:                                   # R1 excludes these scenes; refuse loudly
        raise ValueError("object has no cells at 224 (R1 should have excluded the scene)")
    goal = min(target, cap)
    centres = np.argwhere(ob & ~ms)                  # (row, col); no lie mask (condition 1)
    k = int(rng.integers(lo, hi + 1))
    rec["n_ellipses_planned"] = k
    cause = "cap" if cap < target else None
    if goal < 0.5:
        cause = cause or "under_tolerance"
    elif len(centres) == 0:
        cause = "no_eligible_centre"
    else:
        relevant_base = ob & ~ms                     # cells that count toward the budget
        have = 0
        for i in range(k):
            need = (goal - have) / (k - i)
            n = max(1, int(round(need)))
            n = min(n, int(np.floor(goal)) - have) if goal >= 1 else n
            if n < 1:
                break
            placed = False
            for _ in range(tries):
                cy, cx = centres[int(rng.integers(len(centres)))]
                aspect = float(rng.uniform(a_lo, a_hi))
                phi = float(rng.uniform(0.0, np.pi))
                rho = _rho(hw, float(cx), float(cy), aspect, phi)
                new = relevant_base & ~union
                vals = rho[new]
                if len(vals) == 0:
                    break
                m = min(n, len(vals))
                u = np.partition(vals, m - 1)[m - 1]     # m-th smallest, O(N) (no full sort)
                ell = rho <= u
                if int((ell & new).sum()) + have > max(cap, 0):   # exact ties past the cap: step back
                    smaller = vals[vals < u]
                    if len(smaller) == 0:
                        rec["tie_retries"] += 1
                        continue
                    u = smaller.max()
                    ell = rho <= u
                union |= ell
                rec["n_ellipses"] += 1
                rec["ellipses"].append((float(cx), float(cy), aspect, phi, float(u)))
                placed = True
                break
            have = int((union & relevant_base).sum())
            if not placed:
                cause = cause or "tie_cap"
                break
    m_synth = union & ~ms
    achieved = int((m_synth & ob).sum())
    rec["achieved"] = achieved
    rec["achieved_frac"] = achieved / n_obj
    rec["within_tol"] = abs(achieved - target) <= tol * target
    if not rec["within_tol"]:
        if achieved > target:                        # e.g. one cell of exact ties on a tiny object
            rec["shortfall_cause"] = "over_tolerance"
        else:
            rec["shortfall_cause"] = cause or "under_tolerance"
    if achieved > max(cap, 0):
        raise AssertionError(f"M_synth puts {achieved} object cells in holes, above the cap {cap}")
    return m_synth.astype(np.uint8), rec


def covered_lie_frac(m_synth, lie):
    """Share of the hole's cells that were natural lies (lie_any@12 inside M_synth), condition 2.
    0.0 for an empty hole."""
    h = np.asarray(m_synth).astype(bool)
    n = int(h.sum())
    return float((h & np.asarray(lie).astype(bool)).sum() / n) if n else 0.0
