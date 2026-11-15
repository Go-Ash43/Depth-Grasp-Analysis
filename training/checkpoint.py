"""Checkpoints, the epoch log and crash recovery (step 5 proposal rev 3 section 1).

Commit order for epoch e (scripts/train.py):
  1 train (one scheduler step after every optimizer step)      2 validate
  3 val/epoch_eee.csv   (tmp, fsync, os.replace, fsync dir)     4 ckpt/epoch_eee.pt  (same)
  5 append the log line {run_id, epoch, ckpt_sha256, csv_sha256, ...}, flush, os.fsync
Epoch e is COMMITTED only after step 5's fsync.

Recovery (recover): parse log.jsonl (a torn FINAL record is archived; a malformed interior record is an
error; epochs must be 0..k with one run_id); verify the committed checkpoints and the last CSV against the
logged sha256s and the checkpoint metadata; archive (move, never delete) every orphan with epoch > k
(e.g. a finished checkpoint whose log line was never written) and every *.tmp into recovery/<stamp>/;
record the event in recovery.json. Training then replays epoch k + 1 (k = -1: from the seeded init).

Checkpoints hold tensors and primitive values only and load with torch.load(weights_only=True).
"""
import fcntl
import hashlib
import io
import json
import os
import socket
import time
import uuid
from pathlib import Path

import torch

LOG = "log.jsonl"
LOCK = ".lock"


def acquire_run_lock(run_dir):
    """Exclusive, non-blocking flock on <run>/.lock for a fresh run or a resume (recovery included).
    Returns the fd; the caller releases it with release_run_lock in a finally block.

    LIMITATION: flock reliably excludes processes on ONE machine. On a network volume it may not exclude
    a process on a different pod: use one pod per run directory."""
    p = Path(run_dir) / LOCK
    fd = os.open(p, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        held = p.read_text().strip()
        raise SystemExit(f"{run_dir} is in use by another process ({held or 'holder unknown'})") from None
    except BaseException:
        os.close(fd)
        raise
    try:
        os.ftruncate(fd, 0)
        os.write(fd, f"pid {os.getpid()} host {socket.gethostname()} since {time.strftime('%F %T')}\n".encode())
        os.fsync(fd)
    except BaseException:
        release_run_lock(fd)
        raise
    return fd


def release_run_lock(fd):
    """Explicit unlock and close (tests call main() repeatedly in one process)."""
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def fsync_dir(d):
    fd = os.open(str(d), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(path, data):
    """<name>.tmp, flush + fsync, os.replace, fsync the directory. Returns sha256 of data."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)
    return sha256_bytes(data)


def atomic_write_json(path, obj):
    return atomic_write_bytes(path, (json.dumps(obj, indent=1, default=str) + "\n").encode())


def ckpt_path(run_dir, epoch):
    return Path(run_dir) / "ckpt" / f"epoch_{int(epoch):03d}.pt"


def csv_path(run_dir, epoch):
    return Path(run_dir) / "val" / f"epoch_{int(epoch):03d}.csv"


# ------------------------------------------------------------------ state
def capture_state(model, optimizer, scheduler, generators, meta):
    """generators: {name: torch.Generator}. meta: primitives only."""
    return {"meta": dict(meta), "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": {"torch_cpu": torch.get_rng_state(),
                    "cuda": list(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else [],
                    "generators": {k: g.get_state() for k, g in generators.items()}}}


def restore_state(state, model, optimizer, scheduler, generators):
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["rng"]["torch_cpu"])
    if state["rng"]["cuda"]:
        if not torch.cuda.is_available() or torch.cuda.device_count() != len(state["rng"]["cuda"]):
            raise RuntimeError("checkpoint has CUDA RNG states for a different device count")
        torch.cuda.set_rng_state_all(state["rng"]["cuda"])
    if set(state["rng"]["generators"]) != set(generators):
        raise RuntimeError(f"generator names differ: {sorted(state['rng']['generators'])} vs {sorted(generators)}")
    for k, g in generators.items():
        g.set_state(state["rng"]["generators"][k])


def save_checkpoint(path, state):
    buf = io.BytesIO()
    torch.save(state, buf)
    return atomic_write_bytes(path, buf.getvalue())


def load_checkpoint(path, expect_sha=None):
    data = Path(path).read_bytes()
    sha = sha256_bytes(data)
    if expect_sha is not None and sha != expect_sha:
        raise RuntimeError(f"{path}: sha256 {sha} != logged {expect_sha}")
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)


# ------------------------------------------------------------------ log
def append_log(run_dir, record):
    """Append one JSON line, flush and fsync: the epoch is committed when this returns."""
    p = Path(run_dir) / LOG
    with open(p, "a") as fh:
        fh.write(json.dumps(record, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def read_log(run_dir):
    """-> (records, torn_tail or None). Only the final record may be torn."""
    p = Path(run_dir) / LOG
    if not p.exists():
        return [], None
    text = p.read_text()
    if not text:
        return [], None
    parts = text.split("\n")
    complete, tail = parts[:-1], parts[-1]      # tail == "" when the file ends with a newline
    records, torn = [], (tail or None)
    for n, line in enumerate(complete):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if n == len(complete) - 1 and torn is None:
                torn = line                     # last newline-terminated record unparsable: torn
            else:
                raise RuntimeError(f"{p}: malformed log record at line {n + 1} (not the final record)")
    return records, torn


def recover(run_dir, run_id, expect_meta):
    """-> (k, report). k = last committed epoch (-1 if none). Archives orphans and a torn tail.
    expect_meta: checkpoint meta keys that must equal run_config values (model, seed, ...)."""
    run_dir = Path(run_dir)
    records, torn = read_log(run_dir)
    for i, r in enumerate(records):
        if r.get("run_id") != run_id:
            raise RuntimeError(f"log record {i} has run_id {r.get('run_id')!r}, expected {run_id!r}")
        if r.get("epoch") != i:
            raise RuntimeError(f"log record {i} has epoch {r.get('epoch')!r}: epochs must run 0..k")
    k = len(records) - 1
    for r in records:                                # every committed checkpoint must be intact
        st = load_checkpoint(ckpt_path(run_dir, r["epoch"]), r["ckpt_sha256"])
        meta = st["meta"]
        if meta.get("run_id") != run_id or meta.get("epoch") != r["epoch"] \
                or meta.get("global_step") != r["global_step"] or meta.get("csv_sha256") != r["csv_sha256"]:
            raise RuntimeError(f"checkpoint {r['epoch']}: metadata does not match the log")
        for key, want in expect_meta.items():
            if meta.get(key) != want:
                raise RuntimeError(f"checkpoint {r['epoch']}: meta {key}={meta.get(key)!r} != run {want!r}")
    if k >= 0:
        got = sha256_bytes(csv_path(run_dir, k).read_bytes())
        if got != records[k]["csv_sha256"]:
            raise RuntimeError(f"val CSV of epoch {k}: sha256 {got} != logged {records[k]['csv_sha256']}")
    stamp = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]     # unique per recovery
    arch = run_dir / "recovery" / stamp
    moved = []

    def ensure_arch():
        if not arch.exists():
            arch.mkdir(parents=True, exist_ok=False)  # never merge into an earlier archive

    def archive(p):
        ensure_arch()
        rel = p.relative_to(run_dir)
        dst = arch / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(p, dst)
        moved.append(rel.as_posix())

    for sub in ("ckpt", "val"):
        d = run_dir / sub
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.name.endswith(".tmp"):
                archive(p)
                continue
            stem = p.name.split(".")[0]
            if stem.startswith("epoch_") and stem[6:].isdigit() and int(stem[6:]) > k:
                archive(p)
    for p in sorted(run_dir.glob("*.tmp")):
        archive(p)
    if torn is not None:
        ensure_arch()                                 # also unique when only the log tail is torn
        (arch / "log.torn_tail.txt").write_text(torn)
        good = "".join(json.dumps(r, default=str) + "\n" for r in records)
        atomic_write_bytes(run_dir / LOG, good.encode())
        moved.append("log.jsonl (torn final record)")
    report = {"time": stamp, "last_committed_epoch": k, "archived": moved,
              "archive_dir": str(arch.relative_to(run_dir)) if moved else None}
    return k, report
