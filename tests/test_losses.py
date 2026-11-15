"""models/losses.py: shapes, finiteness, void masking (values and gradients), pooled reductions."""
import pytest
import torch
import torch.nn.functional as F

from models.losses import TERMS, LossAccumulator, grasp_loss

W = {"q": 1.0, "cos2": 1.0, "sin2": 1.0, "width": 1.0}
S = 224


def _batch(b=2, seed=0, nonvoid_frac=0.1):
    g = torch.Generator().manual_seed(seed)
    t = {k: torch.rand(b, 1, S, S, generator=g) for k in TERMS}
    t["void"] = torch.rand(b, 1, S, S, generator=g) > nonvoid_frac
    p = {k: torch.rand(b, 1, S, S, generator=g, requires_grad=True) for k in TERMS}
    return p, t


def test_hand_computed_pooled_value():
    p, t = _batch()
    loss, st = grasp_loss(p, t, W)
    nv = ~t["void"]
    want = F.smooth_l1_loss(p["q"], t["q"], reduction="sum") / p["q"].numel()
    for k in ("cos2", "sin2", "width"):
        want = want + F.smooth_l1_loss(p[k][nv], t[k][nv], reduction="sum") / int(nv.sum())
    assert torch.allclose(loss, want, rtol=1e-6)
    assert st["q"][1] == 2 * S * S and st["cos2"][1] == int(nv.sum())


def test_weights_scale_terms():
    p, t = _batch()
    l1, st = grasp_loss(p, t, W)
    l2, _ = grasp_loss(p, t, {**W, "width": 3.0})
    extra = 2.0 * float(st["width"][0]) / st["width"][1]
    assert abs(float(l2.detach()) - float(l1.detach()) - extra) < 1e-5


def test_void_gradients_zero_for_angle_width_but_q_supervised():
    p, t = _batch()
    loss, _ = grasp_loss(p, t, W)
    loss.backward()
    v = t["void"]
    for k in ("cos2", "sin2", "width"):
        assert torch.all(p[k].grad[v] == 0)
        assert torch.any(p[k].grad[~v] != 0)
    assert torch.any(p["q"].grad[v] != 0)


def test_void_target_values_do_not_matter():
    p, t = _batch()
    l1, _ = grasp_loss(p, t, W)
    t2 = {k: v.clone() for k, v in t.items()}
    for k in ("cos2", "sin2", "width"):
        t2[k][t["void"]] = 123.0
    l2, _ = grasp_loss(p, t2, W)
    assert torch.equal(l1, l2)


@pytest.mark.parametrize("bad", ["pred_squeezed", "target_squeezed", "void_float", "size"])
def test_shape_contract(bad):
    p, t = _batch()
    if bad == "pred_squeezed":
        p["cos2"] = p["cos2"][:, 0]
    elif bad == "target_squeezed":
        t["width"] = t["width"][:, 0]
    elif bad == "void_float":
        t["void"] = t["void"].float()
    else:
        p = {k: v[..., :200, :200] for k, v in p.items()}
        t = {k: v[..., :200, :200] for k, v in t.items()}
    with pytest.raises(ValueError):
        grasp_loss(p, t, W)


@pytest.mark.parametrize("k", TERMS)
def test_non_finite_output_rejected_even_in_void(k):
    p, t = _batch()
    with torch.no_grad():
        idx = t["void"].nonzero()[0]
        p[k][tuple(idx)] = float("nan")
    with pytest.raises(FloatingPointError):
        grasp_loss(p, t, W)


def test_all_void_batch_refused():
    p, t = _batch()
    t["void"][:] = True
    with pytest.raises(ValueError):
        grasp_loss(p, t, W)


def test_epoch_accumulation_is_pooled_not_mean_of_means():
    p, t = _batch(b=2, seed=1)
    t["void"][1] = True
    t["void"][1, 0, :3, :3] = False                # image 1: 9 supervised pixels; image 0: ~5,000
    acc = LossAccumulator()
    for b in range(2):
        _, st = grasp_loss({k: v[b:b + 1] for k, v in p.items()}, {k: v[b:b + 1] for k, v in t.items()}, W)
        acc.add(st)
    _, whole = grasp_loss(p, t, W)
    res = acc.result(W)
    for k in TERMS:
        assert abs(res[k] - float(whole[k][0]) / whole[k][1]) < 1e-6
