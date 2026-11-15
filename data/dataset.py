"""Detector dataset over the 224-px cache (docs/proj_v3.md; SPEC_trajectory Module 1 item 2;
handoff "Pipeline" step 5: holes, completion and flags are made per epoch in data-loader workers).

What it guarantees (each one tested in tests/test_dataset.py):
- Input whitelist (D4). x is built ONLY from MODELS[model]: R = RGB; A0 = RGB + completed depth;
  M = A0 + flag; Astar = RGB + D* (no synthetic holes, no flag). D* and lie masks never appear in a
  batch as their own keys; D* enters x only for Astar, M_lie enters the flag only under the named
  "oracle" flag setting (lie_any at the frozen tau 12, configs/train_data.json oracle_tau_mm).
- Protocol A (SPEC_hypotheses "Protocols"): depth is completed over the TRUE invalid set
  M_sensor | M_synth (complete_protocol_a), so the flag setting can never change the depth; with no
  synthetic hole the cached ds_comp is used (bit-identical to complete_protocol_a(ds, M_sensor)).
- Random streams per sample, keyed SeedSequence([key_seed, split, stream, epoch, slot]): holes 0, flags 1
  (reserved for Module 2 flips), augmentation 2, sampler order 3. key_seed = the run seed in "train"
  mode (so R / A0 / M with one seed see identical holes) and configs/train_data.json corruption_seed
  in "eval" mode with epoch 0 (identical corruption instances across models AND model seeds).
  Results do not depend on worker count or order.
- The epoch reaches the workers: items are (epoch, slot) pairs from EpochSampler, which lives in the
  main process; a bare integer index is refused (a persistent worker's own copy of an epoch counter
  would silently go stale, review fix 2).
- The cache is opened lazily in each process (pickling an np.memmap would copy the array).
- Synthetic holes (data/holes.py) never look at M_lie: lie cells inside a hole are covered (flagged and
  completed). Each sample carries covered_lie_frac (lie_any@12 cells in M_synth / M_synth cells) next to
  hole_frac. Analyses that count natural lies under a hole condition must use
  data.lies.lie_eff_from_224 with the deployed flag INCLUDING M_synth (RULES R2), never the raw
  lie_any array (notes/holes_decision_v1.md, condition 3).
- Augmentation: none for training (no-augmentation decision, docs/proj_v3.md; configs/train_data.json flip_axis null, so
  augment=True is refused). The flip code (data/augment.py) stays for a possible later revisit and
  is applied jointly to inputs, masks, targets and the GT grasps carried for matching.
- Normalisation (D3): RGB (rgb / 255 - 0.5) / 0.25; depth by the frozen train constants in
  configs/frozen_day1.json (ds_comp_mm, dstar_mm for Astar); flag 0 / 1. Never per-image statistics.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from data.augment import flip_array, flip_grasps, flip_targets
from data.cache_layout import lie_name, open_cache
from data.flag_edits import FLAG_EDITS
from data.holes import covered_lie_frac, synth_holes
from data.labels import load_labels_cfg, targets_from_cfg
from data.mask_ops import complete_protocol_a

REPO = Path(__file__).resolve().parents[1]
SRC_HW = (1024, 1024)
MODELS = {"R": ("rgb",), "A0": ("rgb", "depth"), "M": ("rgb", "depth", "flag"), "Astar": ("rgb", "dstar")}
FLAG_SETTINGS = ("deployed", "oracle")
MODES = ("train", "eval")
STREAM_HOLES, STREAM_FLAGS, STREAM_AUGMENT, STREAM_ORDER = 0, 1, 2, 3
SEVERITY_ORDER = ("low", "mid", "high")
BATCH_KEYS = frozenset({"x", "q", "cos2", "sin2", "width", "void", "slot", "epoch", "flip", "m_synth",
                        "severity", "hole_frac", "covered_lie_frac", "gt_grasps"})


def load_data_cfg(path=None):
    return json.loads(Path(path or REPO / "configs" / "train_data.json").read_text())


def load_frozen(path=None):
    return json.loads(Path(path or REPO / "configs" / "frozen_day1.json").read_text())


SPLIT_INDEX = {"train": 0, "val": 1, "test_module": 2, "test_sealed": 3}


def stream_rng(key_seed, stream, epoch, slot, split="train"):
    """The one way a per-sample generator is made: SeedSequence([key_seed, split, stream, epoch, slot]).
    The split is part of the key, so slot k of val and slot k of test never share a stream."""
    if split not in SPLIT_INDEX:
        raise ValueError(f"unknown split {split!r}; choose from {tuple(SPLIT_INDEX)}")
    parts = [int(key_seed), SPLIT_INDEX[split], int(stream), int(epoch), int(slot)]
    if any(p < 0 for p in parts):
        raise ValueError(f"seed, stream, epoch and slot must be >= 0, got {parts}")
    return np.random.default_rng(np.random.SeedSequence(parts))


class GraspDataset(Dataset):
    def __init__(self, cache_dir, split, model, *, mode, seed=None, holes="none", flag_setting="deployed",
                 augment=False, flip_axis=None, data_cfg=None, labels_cfg=None, frozen=None, splits=None,
                 allow_smoke=False, allow_sealed=False, flag_edit=None):
        if split == "test_sealed" and not allow_sealed:
            raise ValueError("test_sealed is opened once, for the final sealed-test run: pass allow_sealed=True")
        if mode == "train" and split != "train":
            raise ValueError(f"train mode reads the train split only, got {split!r}")
        if model not in MODELS:
            raise ValueError(f"model must be one of {tuple(MODELS)}, got {model!r}")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if split not in SPLIT_INDEX:
            raise ValueError(f"split must be one of {tuple(SPLIT_INDEX)}, got {split!r}")
        if flag_setting not in FLAG_SETTINGS:
            raise ValueError(f"flag_setting must be one of {FLAG_SETTINGS}, got {flag_setting!r}")
        self.data_cfg = data_cfg if data_cfg is not None else load_data_cfg()
        self.labels_cfg = labels_cfg if labels_cfg is not None else load_labels_cfg()
        frozen = frozen if frozen is not None else load_frozen()
        sev = self.data_cfg["holes"]["severities"]
        if sorted(sev) != sorted(SEVERITY_ORDER):
            raise ValueError(f"severities must be {SEVERITY_ORDER}, got {sorted(sev)}")
        allowed_holes = ("none", "train", *SEVERITY_ORDER)
        if holes not in allowed_holes:
            raise ValueError(f"holes must be one of {allowed_holes}, got {holes!r}")
        if model == "Astar" and (holes != "none" or flag_setting != "deployed"):
            raise ValueError("Astar (perfect depth) takes no synthetic holes and no flag (D4)")
        if flag_edit is not None:                     # Protocol A intervention: eval only, never Astar
            if flag_edit not in FLAG_EDITS or mode != "eval" or model == "Astar":
                raise ValueError(f"flag_edit {flag_edit!r} needs mode='eval', a non-Astar model and "
                                 f"one of {tuple(FLAG_EDITS)}")
        self.flag_edit = flag_edit
        if mode == "train":
            if seed is None or int(seed) < 0:
                raise ValueError("train mode needs a run seed >= 0")
            self.key_seed = int(seed)
        else:
            if holes == "train" or augment:
                raise ValueError("eval mode: fixed severities only and no augmentation")
            self.key_seed = int(self.data_cfg["corruption_seed"])   # never the run seed (review fix 3)
        if augment:
            if self.data_cfg["augment"]["train"] != "flip":   # "none" is binding, not just a null axis
                raise ValueError(f"augment=True but configs/train_data.json augment.train = "
                                 f"{self.data_cfg['augment']['train']!r} (no augmentation, docs/proj_v3.md)")
            flip_axis = flip_axis or self.data_cfg["augment"]["flip_axis"]
            if flip_axis is None:
                raise ValueError("augment=True but configs/train_data.json augment.flip_axis is null: "
                                 "training uses no augmentation (no-augmentation decision, docs/proj_v3.md)")
        self.cache_dir, self.split, self.model, self.mode = cache_dir, split, model, mode
        self.holes, self.flag_setting, self.augment, self.flip_axis = holes, flag_setting, augment, flip_axis
        self.splits, self.allow_smoke = splits, allow_smoke
        self.radius = float(frozen["telea_radius"])
        nm = frozen["normalisation"]
        self.depth_norm = (float(nm["ds_comp_mm"]["median"]), float(nm["ds_comp_mm"]["robust_sigma"]))
        self.dstar_norm = (float(nm["dstar_mm"]["median"]), float(nm["dstar_mm"]["robust_sigma"]))
        rn = self.data_cfg["rgb_norm"]
        self.rgb_norm = (float(rn["mean"]), float(rn["std"]))
        self.covered_name = self.data_cfg["holes"]["covered_lie_mask"]   # logging only, never placement
        self.oracle_name = lie_name("any", float(self.data_cfg["oracle_tau_mm"]))
        self._cache = None
        c = self._c()                                 # validate once in the constructing process
        for n in (self.covered_name, self.oracle_name):
            if n not in c["arrays"]:
                raise ValueError(f"cache lacks {n}")
        self.slots = np.array([i for i, r in enumerate(c["index"]) if r["status"] == "ok"], np.int64)
        self._ok = set(int(s) for s in self.slots)
        self._cache = None                            # never pickled into workers

    def __getstate__(self):
        st = dict(self.__dict__)
        st["_cache"] = None
        return st

    def _c(self):
        if self._cache is None:
            self._cache = open_cache(self.cache_dir, self.split, allow_smoke=self.allow_smoke,
                                     splits=self.splits)
        return self._cache

    def __len__(self):
        return len(self.slots)

    def sampler(self, shuffle, order_seed=None):
        return EpochSampler(self.slots, shuffle=shuffle, strict=self.mode == "train",
                            seed=self.key_seed if order_seed is None else order_seed)

    def __getitem__(self, key):
        if not (isinstance(key, tuple) and len(key) == 2):
            raise TypeError("items are (epoch, slot) pairs from EpochSampler; a bare index would freeze "
                            "the epoch inside persistent workers")
        epoch, slot = int(key[0]), int(key[1])
        if slot not in self._ok:
            raise KeyError(f"slot {slot} is not an ok slot of {self.split}")
        if self.mode == "eval" and epoch != 0:
            raise ValueError("eval mode uses epoch 0 (corruption instances fixed across runs)")
        return self.sample(epoch, slot)

    # ------------------------------------------------------------------ one sample
    def sample(self, epoch, slot, force_flip=None, return_flag=False):
        """force_flip: None = from the augmentation stream (train) / never (eval); True / False forces
        it (overlays only; needs a flip axis). return_flag=True (eval-mode invariant checks ONLY, e.g.
        Protocol A on flag-blind models) adds "flag_plane" (bool [H, W], the plane the flag channel would
        carry). It is never set by __getitem__, so training / validation batches keep the BATCH_KEYS
        whitelist and the plane never reaches a detector through this path."""
        if return_flag and (self.mode != "eval" or self.model == "Astar"):
            raise ValueError("return_flag is for eval-mode, flag-capable (non-Astar) datasets only")
        c = self._c()
        A = c["arrays"]
        ms = np.asarray(A["m_sensor"][slot]).astype(bool)
        synth = np.zeros_like(ms)
        severity, hole_frac, covered = "clean", 0.0, 0.0
        if self.holes != "none":
            rng_h = stream_rng(self.key_seed, STREAM_HOLES, epoch, slot, self.split)
            if self.holes == "train":
                sev = None if rng_h.random() < float(self.data_cfg["holes"]["p_clean"]) \
                    else SEVERITY_ORDER[int(rng_h.integers(len(SEVERITY_ORDER)))]
            else:
                sev = self.holes
            if sev is not None:
                m, rec = synth_holes(rng_h, ms, np.asarray(A["obj"][slot]),        # no lie input
                                     float(self.data_cfg["holes"]["severities"][sev]), self.data_cfg["holes"])
                synth = m.astype(bool)
                severity, hole_frac = sev, float(rec["achieved_frac"])
                covered = covered_lie_frac(synth, np.asarray(A[self.covered_name][slot]))   # condition 2
        true_inv = ms | synth
        chans = MODELS[self.model]
        planes = {"rgb": np.asarray(A["rgb"][slot])}
        if "depth" in chans:                          # Protocol A: the TRUE invalid set, never the flag
            planes["depth"] = (complete_protocol_a(np.asarray(A["ds"][slot]), true_inv, self.radius)
                               if synth.any() else np.asarray(A["ds_comp"][slot]))
        if self.model != "Astar":                     # every flag-capable path builds the flag, so
            f = true_inv                              # Protocol A exercises R/A0's flag-blind x too
            if self.flag_setting == "oracle":         # named oracle condition only
                f = f | np.asarray(A[self.oracle_name][slot]).astype(bool)
            if self.flag_edit is not None:
                f = FLAG_EDITS[self.flag_edit](f)
            planes["flag"] = f                        # enters x only if "flag" in MODELS[model]
        if "dstar" in chans:                          # Astar only (D4)
            planes["dstar"] = np.asarray(A["dstar"][slot])
        gt = np.array(c["grasps"](slot), np.float64)
        tg = targets_from_cfg(gt, self.labels_cfg)
        flip = 0
        if force_flip is not None:
            flip = int(bool(force_flip))
        elif self.augment:
            flip = int(stream_rng(self.key_seed, STREAM_AUGMENT, epoch, slot, self.split).random() < 0.5)
        if flip:
            if self.flip_axis is None:
                raise ValueError("a flip needs a flip axis")
            planes = {k: flip_array(v, self.flip_axis) for k, v in planes.items()}
            synth = flip_array(synth, self.flip_axis)
            tg = flip_targets(tg, self.flip_axis)
            gt = flip_grasps(gt, self.flip_axis, SRC_HW)  # matching happens in the flipped frame
        out = {"x": torch.from_numpy(self._x(planes)),
               "q": torch.from_numpy(np.ascontiguousarray(tg["q"], np.float32))[None],
                "cos2": torch.from_numpy(np.ascontiguousarray(tg["cos2"], np.float32))[None],
                "sin2": torch.from_numpy(np.ascontiguousarray(tg["sin2"], np.float32))[None],
                "width": torch.from_numpy(np.ascontiguousarray(tg["width"], np.float32))[None],
                "void": torch.from_numpy(np.ascontiguousarray(tg["void"]).astype(bool))[None],
                "slot": slot, "epoch": epoch, "flip": flip,
                "m_synth": torch.from_numpy(np.ascontiguousarray(synth, np.uint8)),
                "severity": severity, "hole_frac": hole_frac, "covered_lie_frac": covered, "gt_grasps": gt}
        if return_flag:
            out["flag_plane"] = torch.from_numpy(np.ascontiguousarray(planes["flag"]).astype(bool))
        return out

    def _x(self, planes):
        out = []
        for ch in MODELS[self.model]:
            if ch == "rgb":
                m, s = self.rgb_norm
                out.append(((planes["rgb"].astype(np.float32) / 255.0 - m) / s).transpose(2, 0, 1))
            elif ch == "depth":
                m, s = self.depth_norm
                out.append(((planes["depth"].astype(np.float64) - m) / s).astype(np.float32)[None])
            elif ch == "dstar":
                m, s = self.dstar_norm
                out.append(((planes["dstar"].astype(np.float64) - m) / s).astype(np.float32)[None])
            elif ch == "flag":
                out.append(planes["flag"].astype(np.float32)[None])
        x = np.ascontiguousarray(np.concatenate(out, 0))
        if not np.isfinite(x).all():
            raise ValueError("non-finite value in the detector input")
        return x


class EpochSampler(Sampler):
    """Yields (epoch, slot) pairs. Lives in the main process, so set_epoch() always takes effect,
    persistent workers or not. Order: fixed (shuffle=False) or a permutation keyed by (seed, epoch)."""

    def __init__(self, slots, shuffle, seed, strict=False):
        """strict (train mode): refuse to iterate an epoch that was already iterated, so a train.py
        that forgets set_epoch() fails loudly instead of reusing every hole and flip."""
        self.slots = np.asarray(slots, np.int64)
        self.shuffle, self.seed, self.epoch = bool(shuffle), int(seed), 0
        self.strict, self._done = bool(strict), set()

    def set_epoch(self, epoch):
        if int(epoch) < 0:
            raise ValueError("epoch must be >= 0")
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.slots)

    def __iter__(self):
        """A generator: the strict check runs when the epoch is first READ, not when iter() is called
        (PyTorch may call iter() on the sampler more than once while starting workers)."""
        e = self.epoch
        if self.strict:
            if e in self._done:
                raise RuntimeError(f"epoch {e} already iterated: call set_epoch() each epoch")
            self._done.add(e)
        order = self.slots
        if self.shuffle:
            order = self.slots[stream_rng(self.seed, STREAM_ORDER, e, 0).permutation(len(self.slots))]
        yield from ((e, int(s)) for s in order)


def collate(batch):
    """Stack tensors; keep variable-length GT grasps and severity names as lists."""
    keys = set(batch[0])
    if keys != BATCH_KEYS:
        raise AssertionError(f"batch keys {sorted(keys)} != whitelist {sorted(BATCH_KEYS)}")
    out = {k: torch.stack([b[k] for b in batch]) for k in ("x", "q", "cos2", "sin2", "width", "void", "m_synth")}
    for k in ("slot", "epoch", "flip"):
        out[k] = torch.tensor([b[k] for b in batch], dtype=torch.int64)
    out["hole_frac"] = torch.tensor([b["hole_frac"] for b in batch], dtype=torch.float32)
    out["covered_lie_frac"] = torch.tensor([b["covered_lie_frac"] for b in batch], dtype=torch.float32)
    out["severity"] = [b["severity"] for b in batch]
    out["gt_grasps"] = [b["gt_grasps"] for b in batch]
    return out
