"""data.provenance.config_matches / content_sha256 and the configs/legacy_hashes.json bridge; data.paths
resolve_manifest. A bridged hash must accept ONLY a provenance-text change, never a changed setting."""
import hashlib
import json
import subprocess

import pytest

from data.paths import ENV_VAR, resolve_manifest
from data.provenance import CONFIG_BRIDGE, REPO, config_matches, content_sha256

OLD = {"version": 1, "created": "2000-01-01", "source": "somewhere",
       "targets": {"q_void": 0.05, "w_norm_note": "old note"}, "seed": 7}


def _write(p, d, **kw):
    p.write_text(json.dumps(d, **kw))
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _bridge(tmp_path, name, old_sha, content):
    b = tmp_path / "bridge.json"
    b.write_text(json.dumps({name: {old_sha: content}}))
    return b


def test_identical_bytes_match_without_bridge(tmp_path):
    p = tmp_path / "labels.json"
    sha = _write(p, OLD)
    assert config_matches(p, sha, bridge=tmp_path / "absent.json")


def test_content_hash_ignores_provenance_format_and_order(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    _write(a, OLD)
    new = {"seed": 7, "targets": {"w_norm_note": "rewritten", "q_void": 0.05}, "version": 1}   # no created/source
    _write(b, new, indent=3)
    assert content_sha256(a) == content_sha256(b)


def test_provenance_only_change_matches_through_bridge(tmp_path):
    p = tmp_path / "labels.json"
    old_sha = _write(p, OLD)
    _write(p, {k: v for k, v in OLD.items() if k not in ("created", "source")})
    b = _bridge(tmp_path, "labels.json", old_sha, content_sha256(p))
    assert config_matches(p, old_sha, bridge=b)
    assert not config_matches(p, old_sha, bridge=tmp_path / "absent.json")     # no bridge -> refused


def test_setting_change_refused_even_for_a_bridged_hash(tmp_path):
    p = tmp_path / "labels.json"
    old_sha = _write(p, OLD)
    _write(p, {k: v for k, v in OLD.items() if k != "created"})
    b = _bridge(tmp_path, "labels.json", old_sha, content_sha256(p))
    _write(p, {**OLD, "created": None, "seed": 8})                              # a setting changed
    assert not config_matches(p, old_sha, bridge=b)


@pytest.mark.parametrize("recorded", ["0" * 64, None, ""])
def test_unlisted_or_missing_hash_refused(tmp_path, recorded):
    p = tmp_path / "labels.json"
    old_sha = _write(p, OLD)
    _write(p, {**OLD, "created": "later"})
    b = _bridge(tmp_path, "labels.json", old_sha, content_sha256(p))
    assert not config_matches(p, recorded, bridge=b)


def test_bridge_is_keyed_by_file_name(tmp_path):
    p = tmp_path / "train_data.json"
    old_sha = _write(p, OLD)
    _write(p, {**OLD, "created": "later"})
    b = _bridge(tmp_path, "labels.json", old_sha, content_sha256(p))            # entry for another file
    assert not config_matches(p, old_sha, bridge=b)


# ------------------------------------------------------------------ the repo's bridge
def _bridge_entries():
    d = json.loads(CONFIG_BRIDGE.read_text())
    return [(name, old, content) for name, m in d.items() if name != "about" for old, content in m.items()]


def test_repo_bridge_lists_the_four_configs():
    assert sorted(n for n, _, _ in _bridge_entries()) == ["frozen_day1.json", "labels.json", "splits.json",
                                                          "train_data.json"]


@pytest.mark.parametrize("name, old, content", _bridge_entries())
def test_repo_configs_match_their_bridged_hash(name, old, content):
    """Fails if a SETTING in a bridged config is edited: the recorded artifacts would no longer match it."""
    p = REPO / "configs" / name
    assert content_sha256(p) == content
    assert config_matches(p, old)


@pytest.mark.parametrize("name, old, content", _bridge_entries())
def test_repo_bridge_against_the_pre_cleanup_file(tmp_path, name, old, content):
    """The listed old hash is the pre-cleanup file (git history) and has the same content hash."""
    try:
        blob = subprocess.run(["git", "-C", str(REPO), "rev-list", "--all", "--objects", "--", f"configs/{name}"],
                              capture_output=True, text=True, check=True).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git history unavailable")
    for obj in blob:
        try:
            data = subprocess.run(["git", "-C", str(REPO), "cat-file", "blob", obj],
                                  capture_output=True, check=True).stdout
        except subprocess.CalledProcessError:
            continue
        if hashlib.sha256(data).hexdigest() == old:
            f = tmp_path / name
            f.write_bytes(data)
            assert content_sha256(f) == content
            return
    pytest.skip(f"pre-cleanup {name} not in this clone's history")


# ------------------------------------------------------------------ manifest path in splits.json
def test_resolve_manifest_absolute_and_bare(tmp_path):
    o = tmp_path / "paths.json"
    o.write_text(json.dumps({"manifest_dir": str(tmp_path / "man")}))
    env = {ENV_VAR: str(o)}
    assert resolve_manifest("manifest.csv", env) == tmp_path / "man" / "manifest.csv"
    assert resolve_manifest(str(tmp_path / "x.csv"), env) == tmp_path / "x.csv"
