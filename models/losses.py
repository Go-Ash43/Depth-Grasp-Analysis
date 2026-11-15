"""Training loss (step 5 proposal rev 2 section 5; configs/train_v1.json "loss").

  L = w_q L_q + w_cos L_cos + w_sin L_sin + w_w L_w,   each term smooth-L1 (beta 1), as GR-ConvNet.
  L_q      = sum over ALL batch pixels / their count   (Q is trained in void cells too; Stephan et al.)
  L_k      = sum over void == 0 batch pixels / their count, k in cos2, sin2, width  (void-masked
             targets, data/labels.py: angle/width are undefined where Q < q_void)

The non-void reduction is POOLED over the batch, not a mean of per-image means: an image with few
supervised pixels gets less weight than under per-image averaging. Deliberate and recorded in the config.

Contract: every prediction and target is exactly [B, 1, size, size] (no broadcasting), void is bool of the
same shape, every prediction value is finite (checked BEFORE masking, so a broken head can never hide
inside void cells), and the batch has at least one non-void pixel (R1 guarantees grasps per ok scene).
Masked cells use torch.where, so their gradient is exactly zero. grasp_loss returns per-term (sum, count)
so an epoch total can be pooled exactly (sum of sums / sum of counts), never an average of batch means.
"""
import torch
import torch.nn.functional as F

TERMS = ("q", "cos2", "sin2", "width")
MASKED = ("cos2", "sin2", "width")
Q_REDUCTIONS = ("all", "split")


def grasp_loss(pred, batch, weights, beta=1.0, size=224, q_reduction="all"):
    """pred: {term: [B,1,size,size]}; batch: same keys plus "void" (bool). ->
    (loss tensor, {term: (sum tensor detached, count int)}).

    q_reduction (proj_v4 sections 5.7-5.9):
      "all"   (default; unchanged)  L_q = sum over ALL batch pixels / count.
      "split" (SQ)                  L_q = sum over void==0 pixels / their count + sum over void==1 pixels /
                                    their count, batch-pooled; an empty void region contributes 0. Gives a
                                    non-void Q pixel the same coefficient as a cos2/sin2/width pixel.
    The reported q stat is ALWAYS (sum over all pixels, all-pixel count), so logs stay comparable."""
    if q_reduction not in Q_REDUCTIONS:
        raise ValueError(f"q_reduction must be one of {Q_REDUCTIONS}, got {q_reduction!r}")
    void = batch["void"]
    if void.dtype != torch.bool or void.dim() != 4 or void.shape[1:] != (1, size, size):
        raise ValueError(f"void must be bool [B,1,{size},{size}], got {void.dtype} {tuple(void.shape)}")
    shape = void.shape
    for k in TERMS:
        if tuple(pred[k].shape) != tuple(shape) or tuple(batch[k].shape) != tuple(shape):
            raise ValueError(f"{k}: prediction {tuple(pred[k].shape)} / target {tuple(batch[k].shape)} "
                             f"must both be {tuple(shape)}")
    nonvoid = ~void
    n_all = void.numel()
    finite = torch.stack([torch.isfinite(pred[k]).all() for k in TERMS])
    n_nv = nonvoid.sum()
    ok = torch.cat([finite, (n_nv > 0)[None]])
    if not bool(ok.all()):                            # one host sync for both checks
        bad = [k for k, f in zip(TERMS, finite.tolist()) if not f]
        if bad:
            raise FloatingPointError(f"non-finite model output in {bad}")
        raise ValueError("batch has no non-void pixel (R1 should make this impossible)")
    n_nv = int(n_nv)
    stats, loss = {}, 0.0
    sl = F.smooth_l1_loss(pred["q"], batch["q"], reduction="none", beta=beta)
    s = sl.sum()
    stats["q"] = (s.detach(), n_all)
    if q_reduction == "all":
        loss = loss + float(weights["q"]) * s / n_all
    else:                                                      # "split"
        n_v = n_all - n_nv
        zero = torch.zeros_like(sl)
        s_nv = torch.where(nonvoid, sl, zero).sum()
        s_v = torch.where(void, sl, zero).sum()
        loss = loss + float(weights["q"]) * (s_nv / n_nv + (s_v / n_v if n_v else s_v * 0.0))
    for k in MASKED:
        sl = F.smooth_l1_loss(pred[k], batch[k], reduction="none", beta=beta)
        s = torch.where(nonvoid, sl, torch.zeros_like(sl)).sum()
        stats[k] = (s.detach(), n_nv)
        loss = loss + float(weights[k]) * s / n_nv
    return loss, stats


class LossAccumulator:
    """Pooled epoch losses: per term, sum of sums / sum of counts."""

    def __init__(self):
        self.sums = {k: 0.0 for k in TERMS}
        self.counts = {k: 0 for k in TERMS}
        self._pending = []

    def add(self, stats):
        self._pending.append({k: stats[k][0] for k in TERMS})      # tensors; synced in result()
        for k in TERMS:
            self.counts[k] += int(stats[k][1])

    def result(self, weights):
        for p in self._pending:
            for k in TERMS:
                self.sums[k] += float(p[k])
        self._pending = []
        terms = {k: (self.sums[k] / self.counts[k] if self.counts[k] else None) for k in TERMS}
        total = None if any(v is None for v in terms.values()) else \
            sum(float(weights[k]) * terms[k] for k in TERMS)
        return {"total": total, **terms, "counts": dict(self.counts)}
