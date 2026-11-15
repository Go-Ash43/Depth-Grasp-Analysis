"""eval/h1.py: crossed two-level bootstrap, reading flags at every boundary, lie strata (P-E1 rev 4)."""
import numpy as np
import pytest

from eval import h1


def _toy(seed=0, S=3, n_obj=12, per=3):
    rng = np.random.default_rng(seed)
    objects = [f"P/o{o}" for o in range(n_obj) for _ in range(per)]
    a = (rng.random((S, n_obj * per)) < 0.8).astype(float)
    b = (rng.random((S, n_obj * per)) < 0.7).astype(float)
    return a, b, objects


def test_point_is_plain_pooled_difference():
    a, b, o = _toy()
    r = h1.crossed_bootstrap(a, b, o, resamples=200)
    assert r["point_pp"] == pytest.approx(100 * (a.mean() - b.mean()))
    assert r["acc_a_pct"] == pytest.approx(100 * a.mean()) and r["acc_b_pct"] == pytest.approx(100 * b.mean())
    assert (r["n_seeds"], r["n_scenes"], r["n_objects"]) == (3, 36, 12)


def test_replicates_equal_explicit_loop_with_shared_object_weights_and_multiplicities():
    a, b, o = _toy(1)
    B, seed = 300, 7
    r = h1.crossed_bootstrap(a, b, o, resamples=B, seed=seed)
    keys, inv = np.unique(np.asarray(o), return_inverse=True)
    rng = np.random.default_rng(seed)
    U = rng.multinomial(3, [1 / 3] * 3, size=B)
    W = rng.multinomial(len(keys), [1 / len(keys)] * len(keys), size=B)
    reps = []
    for u, w in zip(U, W):
        num = den = 0.0
        for s in range(3):                         # every sampled seed uses the SAME object weights w
            for k in range(u[s]):                  # seed multiplicity
                for j in range(len(o)):
                    num += w[inv[j]] * (a[s, j] - b[s, j])
                    den += w[inv[j]]
        reps.append(100 * num / den)
    assert r["ci95"] == pytest.approx(list(np.percentile(reps, [2.5, 97.5])))
    assert r["ci90"] == pytest.approx(list(np.percentile(reps, [5, 95])))


def test_single_arm_and_one_seed():
    a, _, o = _toy(2)
    r = h1.crossed_bootstrap(a[:1], None, o, resamples=100)
    assert r["point_pp"] == pytest.approx(100 * a[0].mean()) and r["n_seeds"] == 1
    assert r["ci95"][0] <= r["point_pp"] <= r["ci95"][1]


def test_refuses_bad_shapes_and_non_binary():
    a, b, o = _toy()
    with pytest.raises(ValueError):
        h1.crossed_bootstrap(a, b[:, :-1], o)
    with pytest.raises(ValueError):
        h1.crossed_bootstrap(a * 0.5 + 0.25, b, o)


@pytest.mark.parametrize("point, ci95, ci90, want", [
    (1.0, [0.2, 1.8], [0.3, 1.7], dict(difference=True, direction="positive", equivalent=True, inconclusive=False)),
    (1.0, [0.0, 2.0], [0.1, 1.9], dict(difference=False, equivalent=True, inconclusive=False)),     # 0 not excluded
    (-1.0, [-2.0, -0.1], [-1.9, -0.2], dict(difference=True, direction="negative", equivalent=True)),
    (0.5, [-3.5, 4.0], [-3.0, 3.5], dict(difference=False, equivalent=False, inconclusive=True, imprecise=True)),
    (0.5, [-2.9, 2.9], [-3.0, 2.0], dict(equivalent=False, inconclusive=True)),                    # -3 not inside
    (0.5, [-2.0, 3.2], [-1.5, 3.0], dict(equivalent=False)),                                        # +3 not inside
    (4.0, [1.0, 7.0], [1.5, 6.5], dict(difference=True, equivalent=False, practically_small=False, imprecise=False)),
    (-4.0, [-7.0, 1.0], [-6.5, 0.5], dict(difference=False, equivalent=False, inconclusive=True,
                                          practically_small=False, imprecise=False)),
    (-2.0, [-2.5, -0.5], [-2.4, -0.6], dict(difference=True, practically_small=True, equivalent=True)),
])
def test_reading_flags_and_boundaries(point, ci95, ci90, want):
    r = h1.reading(point, ci95, ci90)
    for k, v in want.items():
        assert r[k] == v, (k, r)
    assert isinstance(r["label"], str)
    if not r["equivalent"]:
        assert "equivalent" not in r["label"]        # imprecise never implies equivalence


def test_labels():
    assert h1.reading(0.5, [-1, 2], [-0.8, 1.8])["label"] == "equivalent within +/-3 pp; mechanism unresolved"
    assert h1.reading(0.5, [-3.5, 4], [-3.2, 3.5])["label"] == "inconclusive; promising but imprecise"
    assert h1.reading(5, [-1, 10], [-0.5, 9])["label"] == "inconclusive"
    assert "practically small" in h1.reading(2, [0.5, 3.5], [0.6, 3.4])["label"]


def test_lie_strata_and_tertile_ties():
    n = np.array([0, 0, 5, 5, 5, 5, 5, 5, 5])
    c = np.array([0, 0, 0, 0.1, 0.1, 0.1, 0.2, 0.3, 0.3])
    lab, cuts = h1.lie_strata(n, c)
    assert list(lab[:3]) == ["no_hole", "no_hole", "hole_no_lie"]
    q1, q2 = np.quantile([0.1, 0.1, 0.1, 0.2, 0.3, 0.3], [1 / 3, 2 / 3])
    assert cuts == pytest.approx((q1, q2))
    for v, l in zip(c[3:], lab[3:]):               # ties at a cut go to the lower bin
        assert l == ("lie_t1" if v <= q1 else "lie_t2" if v <= q2 else "lie_t3")
    assert h1.lie_strata(np.array([0, 3]), np.array([0, 0]))[1] is None
    with pytest.raises(ValueError):
        h1.lie_strata(np.array([0]), np.array([0.2]))  # covered lie without a hole


def test_stratified_contrast_na():
    a, b, o = _toy(3, n_obj=4, per=2)
    strata = np.array(["no_hole"] * 2 + ["hole_no_lie"] * 6)      # no_hole = one object only -> NA
    out = h1.stratified_contrast(a, b, o, strata, resamples=50)
    assert out["no_hole"]["estimate"] == "NA" and out["lie_t1"]["estimate"] == "NA"
    assert out["hole_no_lie"]["n_scenes"] == 6 and "ci95" in out["hole_no_lie"]
