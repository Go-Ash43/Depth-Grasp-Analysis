"""Environment record for training and evaluation runs: locked package versions and hardware.

Used by training/runcfg.py env_record(); the record's comparison key (runcfg.env_key) lets a resumed run or
an evaluation refuse a changed environment. Functions copied verbatim from the retired pod setup check, so
records written before the move compare equal.
"""
import importlib.metadata as md
import math
import os
import platform
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
LOCK = REPO / "requirements-lock.txt"


# ------------------------------------------------------------------ package versions
def parse_lock(path=LOCK):
    """requirements-lock.txt -> {name: version} (pins 'name==version' only; comments, options skipped)."""
    pins = {}
    for line in Path(path).read_text().splitlines():
        s = line.split("#", 1)[0].strip()
        if not s or s.startswith("-"):
            continue
        if "==" not in s:
            raise ValueError(f"{path}: not an exact pin: {line!r}")
        name, ver = s.split("==", 1)
        pins[name.strip()] = ver.split(";", 1)[0].strip()
    return pins


def check_versions(pins, expect_python):
    bad = {}
    for name, want in sorted(pins.items()):
        try:
            got = md.version(name)
        except md.PackageNotFoundError:
            got = None
        if got != want:
            bad[name] = {"locked": want, "installed": got}
    py = ".".join(platform.python_version_tuple()[:2])
    return {"n_pins": len(pins), "mismatches": bad, "python": platform.python_version(),
            "python_ok": py == expect_python, "ok": not bad and py == expect_python}


# ------------------------------------------------------------------ hardware
def _read(p):
    try:
        return Path(p).read_text().strip()
    except OSError:
        return None


def usable_cpus():
    out = {"os_cpu_count": os.cpu_count()}
    try:
        out["affinity"] = len(os.sched_getaffinity(0))
    except AttributeError:
        out["affinity"] = None
    quota = None
    v2 = _read("/sys/fs/cgroup/cpu.max")
    if v2:
        q, per = v2.split()[:2]
        quota = None if q == "max" else float(q) / float(per)
    else:
        q, per = _read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"), _read("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        if q and per and int(q) > 0:
            quota = int(q) / int(per)
    out["cgroup_quota"] = quota
    cands = [c for c in (out["os_cpu_count"], out["affinity"], math.floor(quota) if quota else None) if c]
    out["usable"] = max(1, min(cands))
    return out


def memory():
    total = None
    mi = _read("/proc/meminfo")
    if mi:
        for line in mi.splitlines():
            if line.startswith("MemTotal:"):
                total = int(line.split()[1]) * 1024
    lim = _read("/sys/fs/cgroup/memory.max") or _read("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    lim = int(lim) if lim and lim.isdigit() and int(lim) < (1 << 60) else None
    usable = min(x for x in (total, lim) if x) if (total or lim) else None
    return {"meminfo_total_gb": total and round(total / 1e9, 1), "cgroup_limit_gb": lim and round(lim / 1e9, 1),
            "usable_gb": usable and round(usable / 1e9, 1)}


def hardware():
    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            gpus.append({"name": p.name, "memory_gb": round(p.total_memory / 1e9, 1),
                         "capability": f"{p.major}.{p.minor}"})
    try:
        drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        drv = []
    return {"gpus": gpus, "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            "driver": sorted(set(drv)) or None, "cv2": cv2.__version__, "numpy": np.__version__,
            "cpus": usable_cpus(), "ram": memory(), "platform": platform.platform()}
