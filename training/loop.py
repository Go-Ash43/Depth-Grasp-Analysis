"""One training epoch (step 5 proposal rev 2/3; configs/train_v1.json).

- One scheduler step immediately after every optimizer step (cosine horizon fixed at run creation).
- Pooled per-term losses (models.losses.LossAccumulator), never an average of batch means.
- hole_frac / covered_lie_frac per severity from the CPU batch (training holes, run seed).
- Timing: normal execution; cuda.synchronize only at the phase boundaries (start / end of the epoch).
  loader_wait_s = CPU time blocked in next(loader), unsynchronised and approximate. sync_timing=True
  synchronises after every step (diagnostic only: changes the overlap, never used for the budget).
"""
import time

import torch

from models.losses import LossAccumulator, grasp_loss
from training.runcfg import q_reduction

TARGETS = ("q", "cos2", "sin2", "width", "void")


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def train_epoch(model, loader, optimizer, scheduler, device, cfg, sync_timing=False, step_hook=None,
                progress_every=0, log=print):
    """progress_every > 0: every N steps log step, running pooled losses ("monitor loss": Q pooled over all
    pixels, not the SQ objective), samples/s, loader wait, lr, and the WINDOW since the last line: mean and
    max of the per-step pooled Q and the step of the max (spike check, proj_v4 5.13). The window keeps the
    per-step Q sums as detached device tensors and syncs once per line, never per step (no extra stall, so
    epoch timings stay valid). Changes no computed value."""
    weights, beta = cfg["loss"]["weights"], float(cfg["loss"]["beta"])
    q_red = q_reduction(cfg)                          # "all" (config loss) or "split" (SQ), proj_v4 5.13
    model.train()
    acc = LossAccumulator()
    sev = {}
    hole_sum = cov_sum = 0.0
    n = steps = 0
    wait = 0.0
    win = []                                          # progress window: per-step (q sum tensor, count)
    sync(device)
    t0 = time.perf_counter()
    it = iter(loader)
    while True:
        tw = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            break
        wait += time.perf_counter() - tw
        x = batch["x"].to(device, non_blocking=True)
        tg = {k: batch[k].to(device, non_blocking=True) for k in TARGETS}
        pred = model(x)
        loss, stats = grasp_loss(pred, tg, weights, beta, q_reduction=q_red)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()                              # one scheduler step per optimizer step
        steps += 1
        acc.add(stats)
        b = batch["x"].shape[0]
        n += b
        for i in range(b):
            s = batch["severity"][i]
            e = sev.setdefault(s, {"n": 0, "hole_frac_sum": 0.0, "covered_lie_frac_sum": 0.0})
            e["n"] += 1
            e["hole_frac_sum"] += float(batch["hole_frac"][i])
            e["covered_lie_frac_sum"] += float(batch["covered_lie_frac"][i])
        hole_sum += float(batch["hole_frac"].sum())
        cov_sum += float(batch["covered_lie_frac"].sum())
        if sync_timing:
            sync(device)
        if step_hook is not None:
            step_hook(steps)
        if progress_every:
            win.append(stats["q"])                     # (detached device sum, int count): no sync here
            if steps % progress_every == 0:
                el = time.perf_counter() - t0
                run = acc.result(weights)              # flushes pending sums; totals stay exact
                sums = torch.stack([s_ for s_, _ in win]).tolist()      # the line's one sync
                qs = [s_ / c_ for s_, (_, c_) in zip(sums, win)]
                k = max(range(len(qs)), key=qs.__getitem__)
                first = steps - len(qs) + 1
                log(f"  step {steps}: monitor loss (Q pooled over all pixels) {run['total']:.5f} (q {run['q']:.2e} "
                    f"cos2 {run['cos2']:.2e} sin2 {run['sin2']:.2e} width {run['width']:.2e}) | "
                    f"Q window steps {first}-{steps}: mean {sum(qs) / len(qs):.2e} max {qs[k]:.2e} @ step "
                    f"{first + k} | {n / el:.1f} samples/s | loader wait {wait:.0f}s of {el:.0f}s | "
                    f"lr {optimizer.param_groups[0]['lr']:.2e}")
                win = []
    sync(device)
    wall = time.perf_counter() - t0
    if steps == 0:
        raise RuntimeError("no optimizer step in this epoch (empty loader)")
    per_sev = {s: {"n": e["n"], "hole_frac_mean": e["hole_frac_sum"] / e["n"],
                   "covered_lie_frac_mean": e["covered_lie_frac_sum"] / e["n"]} for s, e in sorted(sev.items())}
    return {"steps": steps, "samples": n, "loss": acc.result(weights),
            "hole_frac_mean": hole_sum / n, "covered_lie_frac_mean": cov_sum / n, "per_severity": per_sev,
            "train_s": wall, "loader_wait_s": wait, "samples_per_s": n / wall if wall > 0 else None,
            "lr_end": float(optimizer.param_groups[0]["lr"])}
