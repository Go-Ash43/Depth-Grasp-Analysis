"""Per-machine paths: this workstation (configs/paths.json) or another machine such as a RunPod pod.

configs/paths.json stays the default and is never edited for another machine. A machine with other
paths sets the environment variable DEPTH_LIES_PATHS to a JSON file OUTSIDE the repo holding a subset
of the same keys (e.g. {"cache_dir": "/root/cache", "runs_dir": "/workspace/runs"}); the keys it names
override the defaults, an unknown key is an error (a typo must not silently fall back to a local path).
"""
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_PATHS = REPO / "configs" / "paths.json"
ENV_VAR = "DEPTH_LIES_PATHS"


def load_paths(env=None):
    """-> dict of paths. env: mapping used instead of os.environ (tests)."""
    paths = json.loads(DEFAULT_PATHS.read_text())
    override = (os.environ if env is None else env).get(ENV_VAR)
    if override:
        p = Path(override).expanduser()
        extra = json.loads(p.read_text())
        if not isinstance(extra, dict):
            raise ValueError(f"{p}: must hold a JSON object of path keys")
        unknown = sorted(set(extra) - set(paths))
        if unknown:
            raise ValueError(f"{p}: unknown path keys {unknown}; allowed {sorted(paths)}")
        paths = {**paths, **extra}
    return paths


def resolve_manifest(p, env=None):
    """splits.json "manifest": an absolute path is used as is; a bare file name is looked up in manifest_dir."""
    p = Path(p).expanduser()
    return p if p.is_absolute() else Path(load_paths(env)["manifest_dir"]).expanduser() / p


def paths_source(env=None):
    """Where the paths came from, for config JSONs."""
    override = (os.environ if env is None else env).get(ENV_VAR)
    return {"default": str(DEFAULT_PATHS), "override": override or None}
