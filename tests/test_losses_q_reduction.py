"""grasp_loss q_reduction (proj_v4 section 5.8): "all" must be BIT-identical to the loss before the option
existed; "split" must equal the diagnostic SQ (reference copy _split_q_loss below)."""
import copy

import pytest
import torch
import torch.nn.functional as F

from models.losses import MASKED, TERMS, grasp_loss


def _split_q_loss(pred_q, tgt_q, void, beta):
    """Reference SQ, verbatim from the retired overfit-20 diagnostic: batch-pooled sum_nonvoid / n_nv +
    sum_void / n_void of smooth-L1 (beta). Masked cells get exactly zero gradient from the other region's
    term (torch.where). -> (loss, {"q_nonvoid": (sum, n), "q_void": (sum, n)}); an empty void region
    contributes 0 with n = 0; an empty non-void region raises."""
    if void.dtype != torch.bool or pred_q.shape != tgt_q.shape or pred_q.shape != void.shape:
        raise ValueError(f"need equal shapes and bool void, got {tuple(pred_q.shape)} {tuple(tgt_q.shape)} "
                         f"{tuple(void.shape)} {void.dtype}")
    sl = F.smooth_l1_loss(pred_q, tgt_q, reduction="none", beta=beta)
    nonvoid = ~void
    n_nv, n_v = int(nonvoid.sum()), int(void.sum())
    if n_nv == 0:
        raise ValueError("batch has no non-void pixel")
    zero = torch.zeros_like(sl)
    s_nv = torch.where(nonvoid, sl, zero).sum()
    s_v = torch.where(void, sl, zero).sum()
    loss = s_nv / n_nv + (s_v / n_v if n_v else s_v * 0.0)
    return loss, {"q_nonvoid": (s_nv.detach(), n_nv), "q_void": (s_v.detach(), n_v)}


W ={"q": 1.0, "cos2": 1.0, "sin2": 1.0, "width": 1.0}


def old_grasp_loss(pred, batch, weights, beta=1.0):
    """Literal copy of the reduction before q_reduction existed (commit 0b863fe), validation omitted."""
    void = batch["void"]
    nonvoid = ~void
    n_all = void.numel()
    n_nv = int(nonvoid.sum())
    stats, loss = {}, 0.0
    sl = F.smooth_l1_loss(pred["q"], batch["q"], reduction="none", beta=beta)
    s = sl.sum()
    stats["q"] = (s.detach(), n_all)
    loss = loss + float(weights["q"]) * s / n_all
    for k in MASKED:
        sl = F.smooth_l1_loss(pred[k], batch[k], reduction="none", beta=beta)
        s = torch.where(nonvoid, sl, torch.zeros_like(sl)).sum()
        stats[k] = (s.detach(), n_nv)
        loss = loss + float(weights[k]) * s / n_nv
    return loss, stats


def make(seed=0, size=224, b=2, void_frac=0.9):
    g = torch.Generator().manual_seed(seed)
    batch = {k: torch.rand(b, 1, size, size, generator=g) * 2 - 1 for k in TERMS}
    batch["void"] = torch.rand(b, 1, size, size, generator=g) < void_frac
    pred = {k: (torch.rand(b, 1, size, size, generator=g) * 3 - 1.5).requires_grad_(True) for k in TERMS}
    return pred, batch


def grads(pred):
    return {k: pred[k].grad.clone() for k in TERMS}


@pytest.mark.parametrize("kw", [{}, {"q_reduction": "all"}])
def test_all_is_bit_identical_to_the_previous_loss(kw):
    for weights in (W, {"q": 2.5, "cos2": 0.3, "sin2": 1.0, "width": 4.0}):
        p1, batch = make(1)
        p2 = {k: v.detach().clone().requires_grad_(True) for k, v in p1.items()}
        new, st_new = grasp_loss(p1, batch, weights, 1.0, **kw)
        old, st_old = old_grasp_loss(p2, batch, weights, 1.0)
        assert torch.equal(new, old)
        new.backward()
        old.backward()
        for k in TERMS:
            assert torch.equal(p1[k].grad, p2[k].grad), k
            assert torch.equal(st_new[k][0], st_old[k][0]) and st_new[k][1] == st_old[k][1]


def diag_sq(pred, batch, weights):
    """The diagnostic SQ: config grasp_loss with Q weight 0, plus _split_q_loss (reference copy above)."""
    w0 = dict(weights, q=0.0)
    loss, st = grasp_loss(pred, batch, w0, 1.0)
    lq, _ = _split_q_loss(pred["q"], batch["q"], batch["void"], 1.0)
    return loss + lq, st


def test_split_equals_the_diagnostic_sq_in_value_and_gradients():
    # Q-only weights: both sums reduce to the same expression, so the value is bit-equal
    q_only = {"q": 1.0, "cos2": 0.0, "sin2": 0.0, "width": 0.0}
    p1, batch = make(2)
    p2 = {k: v.detach().clone().requires_grad_(True) for k, v in p1.items()}
    a, _ = grasp_loss(p1, batch, q_only, 1.0, q_reduction="split")
    b, _ = diag_sq(p2, batch, q_only)
    assert torch.equal(a, b)
    # full weights: every gradient bit-equal (contributions are disjoint or exact zeros); the scalar value
    # can differ only by summation order
    p1, batch = make(3)
    p2 = {k: v.detach().clone().requires_grad_(True) for k, v in p1.items()}
    a, st_a = grasp_loss(p1, batch, W, 1.0, q_reduction="split")
    b, st_b = diag_sq(p2, batch, W)
    assert torch.allclose(a, b, rtol=1e-6, atol=0)
    a.backward()
    b.backward()
    for k in TERMS:
        assert torch.equal(p1[k].grad, p2[k].grad), k
    assert torch.equal(st_a["q"][0], st_b["q"][0]) and st_a["q"][1] == st_b["q"][1]   # reported q unchanged


def test_split_formula_and_reported_q_stat():
    p, batch = make(4, size=224, b=1)
    q_only = {"q": 1.0, "cos2": 0.0, "sin2": 0.0, "width": 0.0}
    loss, st = grasp_loss(p, batch, q_only, 1.0, q_reduction="split")
    sl = F.smooth_l1_loss(p["q"].detach(), batch["q"], reduction="none", beta=1.0)
    v = batch["void"]
    assert float(loss.detach()) == pytest.approx(float(sl[~v].mean() + sl[v].mean()), rel=1e-6)
    assert st["q"][1] == v.numel() and float(st["q"][0]) == pytest.approx(float(sl.sum()), rel=1e-6)


def test_split_with_no_void_pixel_is_the_nonvoid_mean():
    p, batch = make(5, size=224, b=1)
    batch["void"] = torch.zeros_like(batch["void"])
    q_only = {"q": 1.0, "cos2": 0.0, "sin2": 0.0, "width": 0.0}
    loss, _ = grasp_loss(p, batch, q_only, 1.0, q_reduction="split")
    sl = F.smooth_l1_loss(p["q"].detach(), batch["q"], reduction="none", beta=1.0)
    assert float(loss.detach()) == pytest.approx(float(sl.mean()), rel=1e-6)


def test_unknown_reduction_is_refused():
    p, batch = make(6, size=224, b=1)
    with pytest.raises(ValueError, match="q_reduction"):
        grasp_loss(p, batch, W, 1.0, q_reduction="mean")
    p2 = copy.deepcopy(p)
    assert p2["q"].grad is None
