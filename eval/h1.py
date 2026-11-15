"""H1 statistics (P-E1 rev 4, docs/proj_v7.md sections 13-15; SPEC_hypotheses "H1", "Decision rules" L119;
SPEC_trajectory "Statistics, adapted to Jacquard").

Pure functions only (no file access); scripts/h1_harness.py does the I/O and every identity check.

Two-level crossed bootstrap (SPEC_trajectory clarification): each replicate draws ONE seed
resample u (multinomial, S draws over the S seeds) and ONE object resample w (multinomial, O draws over the
O objects). The object weights are shared by every sampled seed and by both arms of a paired contrast;
repeat-draw multiplicities are kept (never deduplicated). Statistic, in percentage points:

    d(u, w) = 100 * sum_s u_s sum_o w_o (A_so - B_so) / (sum_s u_s * sum_o w_o n_o)

A_so / B_so = matches of arm A / B over the scenes of object o in seed s; n_o = scenes of object o (the same
in every seed: every model and seed sees the identical scene set, checked by the harness). The point estimate
is d(1, 1). With S = 3 seeds there are 10 possible seed count vectors, with unequal multinomial probabilities.

Intervals: percentile, numpy.percentile method "linear" (the default): 95 % = [2.5, 97.5], 90 % = [5, 95].

Reading (rev 4, point 3): independent flags
    difference          95 % CI excludes 0 (an endpoint exactly 0 does NOT exclude 0), with its direction
    equivalent          90 % CI strictly inside (-3, +3) pp (an endpoint exactly +/-3 is NOT inside)
    practically_small   abs(point) < 3 pp
    inconclusive        neither difference nor equivalent
    imprecise           abs(point) < 3 pp but not equivalent: "promising but imprecise" qualifier; never
                        implies equivalence
A null H1 is worded "equivalent within +/-3 pp; mechanism unresolved" ONLY when equivalent is True (the
artifact detectability classifier is cut, docs/proj_v6.md section 8).
"""
import numpy as np

MARGIN_PP = 3.0
RESAMPLES = 10_000
BOOT_SEED = 20261115
Q95 = (2.5, 97.5)
Q90 = (5.0, 95.0)


def _object_table(objects):
    """objects: per-scene object keys -> (inverse index, number of objects, scenes per object)."""
    keys, inv = np.unique(np.asarray(objects, dtype=object).astype(str), return_inverse=True)
    n = np.bincount(inv, minlength=len(keys)).astype(float)
    return inv, len(keys), n


def crossed_bootstrap(a, b, objects, resamples=RESAMPLES, seed=BOOT_SEED):
    """a, b: 0/1 arrays [S, N] (seed x scene, paired by column); b may be None (single-arm accuracy).
    objects: length-N object keys. -> dict with point (pp), ci95, ci90, per-arm accuracies, counts."""
    a = np.asarray(a, float)
    b = np.zeros_like(a) if b is None else np.asarray(b, float)
    if a.ndim != 2 or a.shape != b.shape or a.shape[1] != len(objects) or a.size == 0:
        raise ValueError(f"a, b must be equal [S, N] with N = len(objects); got {a.shape}, {b.shape}, {len(objects)}")
    if not (np.isin(a, (0.0, 1.0)).all() and np.isin(b, (0.0, 1.0)).all()):
        raise ValueError("a and b must be 0/1 match indicators")
    S = a.shape[0]
    inv, n_obj, n = _object_table(objects)
    D = np.stack([np.bincount(inv, a[s] - b[s], n_obj) for s in range(S)])      # [S, O]
    A = np.stack([np.bincount(inv, a[s], n_obj) for s in range(S)])
    B = np.stack([np.bincount(inv, b[s], n_obj) for s in range(S)])
    point = 100.0 * D.sum() / (S * n.sum())
    rng = np.random.default_rng(seed)
    U = rng.multinomial(S, np.full(S, 1.0 / S), size=resamples).astype(float)       # [B, S]
    W = rng.multinomial(n_obj, np.full(n_obj, 1.0 / n_obj), size=resamples).astype(float)   # [B, O]
    num = np.einsum("bs,sb->b", U, D @ W.T)                                            # sum_s u_s sum_o w_o D_so
    reps = 100.0 * num / (S * (W @ n))                                                 # sum_s u_s = S always
    return {"point_pp": float(point),
            "ci95": [float(x) for x in np.percentile(reps, Q95)],
            "ci90": [float(x) for x in np.percentile(reps, Q90)],
            "acc_a_pct": float(100.0 * A.sum() / (S * n.sum())),
            "acc_b_pct": float(100.0 * B.sum() / (S * n.sum())),
            "n_seeds": int(S), "n_scenes": int(a.shape[1]), "n_objects": int(n_obj),
            "resamples": int(resamples), "boot_seed": int(seed), "quantile_method": "linear"}


def reading(point, ci95, ci90, margin=MARGIN_PP):
    """Independent conclusion flags (module docstring). Strict inequalities at 0 and at the margin."""
    lo95, hi95 = ci95
    lo90, hi90 = ci90
    if not (lo95 <= hi95 and lo90 <= hi90):
        raise ValueError(f"malformed intervals {ci95}, {ci90}")
    direction = "positive" if lo95 > 0 else ("negative" if hi95 < 0 else None)
    difference = direction is not None
    equivalent = (-margin < lo90) and (hi90 < margin)
    small = abs(point) < margin
    out = {"difference": difference, "direction": direction, "equivalent": equivalent,
           "practically_small": small, "inconclusive": not difference and not equivalent,
           "imprecise": small and not equivalent}
    out["label"] = label(out)
    return out


def label(r):
    """One descriptive label from the flags (for tables; the flags are the record)."""
    if r["difference"] and r["equivalent"]:
        return f"detectable ({r['direction']}), equivalent within +/-{MARGIN_PP:g} pp"
    if r["difference"]:
        return f"detectable ({r['direction']}), " + ("practically small" if r["practically_small"]
                                                      else f"at least {MARGIN_PP:g} pp in the point estimate")
    if r["equivalent"]:
        return f"equivalent within +/-{MARGIN_PP:g} pp; mechanism unresolved"
    return "inconclusive" + ("; promising but imprecise" if r["imprecise"] else "")


def contrast(a, b, objects, resamples=RESAMPLES, seed=BOOT_SEED):
    """crossed_bootstrap + reading."""
    r = crossed_bootstrap(a, b, objects, resamples, seed)
    r["reading"] = reading(r["point_pp"], r["ci95"], r["ci90"])
    return r


def tertile_cuts(values):
    """Cut points q1, q2 = numpy.quantile(values, [1/3, 2/3], method "linear") over the covering scenes."""
    v = np.asarray(values, float)
    if v.size == 0:
        return None
    q1, q2 = np.quantile(v, [1 / 3, 2 / 3], method="linear")
    return float(q1), float(q2)


def lie_strata(n_synth_cells, covered):
    """Per-scene stratum label (rev 4 point 4), from the sidecar's n_synth_cells and covered_lie_frac:
      "no_hole"          n_synth_cells == 0
      "hole_no_lie"      n_synth_cells > 0 and covered == 0
      "lie_t1/t2/t3"     covered > 0, tertiles: [min, q1], (q1, q2], (q2, max]; ties go to the lower bin.
    -> (labels array, cuts or None)."""
    n = np.asarray(n_synth_cells, np.int64)
    c = np.asarray(covered, float)
    if n.shape != c.shape or (n < 0).any() or (c < 0).any() or (c > 1).any():
        raise ValueError("n_synth_cells and covered must be equal-shape, non-negative, covered <= 1")
    if ((n == 0) & (c != 0)).any():
        raise ValueError("covered_lie_frac > 0 for a scene without a synthetic hole")
    lab = np.empty(n.shape, object)
    lab[n == 0] = "no_hole"
    lab[(n > 0) & (c == 0)] = "hole_no_lie"
    cov = (n > 0) & (c > 0)
    cuts = tertile_cuts(c[cov])
    if cuts is not None:
        q1, q2 = cuts
        lab[cov & (c <= q1)] = "lie_t1"
        lab[cov & (c > q1) & (c <= q2)] = "lie_t2"
        lab[cov & (c > q2)] = "lie_t3"
    return lab, cuts


STRATA = ("no_hole", "hole_no_lie", "lie_t1", "lie_t2", "lie_t3")


def stratified_contrast(a, b, objects, strata, resamples=RESAMPLES, seed=BOOT_SEED):
    """M - A0 within each stratum (no verdict). NA when a stratum has no scenes or one object."""
    objects = np.asarray(objects, dtype=object)
    out = {}
    for s in STRATA:
        m = np.asarray(strata) == s
        n_sc, n_ob = int(m.sum()), len(set(objects[m])) if m.any() else 0
        if n_sc == 0 or n_ob < 2:
            out[s] = {"n_scenes": n_sc, "n_objects": n_ob, "estimate": "NA"}
            continue
        r = crossed_bootstrap(np.asarray(a)[:, m], np.asarray(b)[:, m], objects[m], resamples, seed)
        out[s] = {"n_scenes": n_sc, "n_objects": n_ob, **r}
    return out
