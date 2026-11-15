"""Run provenance shared by scripts: git state, output guards, config identity. No dataset access."""
import hashlib
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Config identity (configs/legacy_hashes.json). Artifacts (cache summaries, run configs) record the sha256 of
# the config files they were built with. Provenance-only text (creation dates, local paths, free-text notes)
# was later removed from four configs; the bridge lets the recorded pre-removal hashes still identify them.
CONFIG_BRIDGE = REPO / "configs" / "legacy_hashes.json"
PROVENANCE_KEYS = frozenset({"created", "frozen_on", "date", "source", "audit_dir", "manifest"})


def _strip_provenance(obj):
    """Drop PROVENANCE_KEYS and every "*_note" key at any depth; everything else is kept as is."""
    if isinstance(obj, dict):
        return {k: _strip_provenance(v) for k, v in obj.items()
                if k not in PROVENANCE_KEYS and not k.endswith("_note")}
    if isinstance(obj, list):
        return [_strip_provenance(v) for v in obj]
    return obj


def content_sha256(path):
    """sha256 of a JSON config's settings: provenance keys removed, keys sorted, compact separators.
    Formatting, key order and provenance text do not change it; any setting does."""
    d = json.loads(Path(path).read_text())
    canon = json.dumps(_strip_provenance(d), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canon.encode()).hexdigest()


def config_matches(path, recorded_sha256, bridge=None):
    """Is the config file at `path` the one an artifact recorded as `recorded_sha256`?

    Yes if the file's bytes hash to it (the normal case: every artifact built after the cleanup).
    Otherwise yes ONLY through the bridge: `recorded_sha256` is listed there as a pre-cleanup file hash for
    this file name, and this file's content_sha256 equals the content hash listed for it (only provenance
    text differs). A changed setting, an unlisted hash or a missing bridge file -> False.
    """
    path = Path(path)
    if not recorded_sha256:
        return False
    if hashlib.sha256(path.read_bytes()).hexdigest() == recorded_sha256:
        return True
    bp = Path(bridge) if bridge is not None else CONFIG_BRIDGE
    if not bp.is_file():
        return False
    want = json.loads(bp.read_text()).get(path.name, {}).get(recorded_sha256)
    return want is not None and content_sha256(path) == want


def git_state():
    """(commit hash, dirty bool); (None, None) if git or the repo is unavailable."""
    try:
        head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                                    capture_output=True, text=True, check=True).stdout.strip())
        return head, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def git_fields():
    """{"git_commit", "git_dirty"} for a config JSON."""
    head, dirty = git_state()
    return {"git_commit": head, "git_dirty": dirty}


def refuse_inside(path, *roots):
    """Raise SystemExit if `path` is one of `roots` or inside one (resolved, parent check)."""
    p = Path(path).expanduser().resolve()
    for r in roots:
        r = Path(r).expanduser().resolve()
        if p == r or r in p.parents:
            raise SystemExit(f"refusing to write inside the dataset root {r}: {p}")
    return p


def refuse_existing(paths, overwrite):
    """Raise SystemExit if any of `paths` exists and overwrite is False."""
    found = [str(p) for p in paths if Path(p).exists()]
    if found and not overwrite:
        raise SystemExit(f"{len(found)} output(s) already exist, e.g. {found[0]}; "
                         f"use a fresh --out or pass --overwrite")
