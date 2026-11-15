"""RULES R3 skip limits (SkipBudget) and the narrowed SCENE_ERRORS (resource faults must crash)."""
import errno

import pytest

import data.jacquard as jq
from data.jacquard import SkipBudget, try_load_scene


def _rec(i):
    return {"obj_dir": f"o{i}", "view": 0, "error_type": "SceneDataError", "message": "x"}


def test_total_limit_is_more_than_50():
    b = SkipBudget(max_total=50, max_run=10)
    for i in range(50):
        b.skipped(_rec(i))
        b.loaded()                                   # never 10 in a row
    with pytest.raises(RuntimeError, match="> 50"):
        b.skipped(_rec(50))


def test_run_limit_is_10_in_a_row_and_resets():
    b = SkipBudget(max_total=50, max_run=10)
    for i in range(9):
        b.skipped(_rec(i))
    b.loaded()                                       # resets the run
    for i in range(9):
        b.skipped(_rec(i))
    with pytest.raises(RuntimeError, match="in a row"):
        b.skipped(_rec(99))


def test_summary():
    b = SkipBudget()
    b.skipped(_rec(0)); b.skipped(_rec(1)); b.loaded(); b.skipped(_rec(2))
    s = b.summary()
    assert s["n_skipped"] == 3 and s["max_run_seen"] == 2 and len(s["skips"]) == 3
    assert s["max_total"] == 50 and s["max_run"] == 10


@pytest.fixture
def root(tmp_path, monkeypatch):
    (tmp_path / "fakeobj").mkdir()
    monkeypatch.setattr(jq, "data_root", lambda: str(tmp_path))
    return tmp_path


@pytest.mark.parametrize("exc", [OSError(errno.EMFILE, "Too many open files"),
                                 OSError(errno.EIO, "Input/output error"),
                                 PermissionError(errno.EACCES, "Permission denied")])
def test_resource_errors_propagate(root, monkeypatch, exc):
    def boom(obj_dir, view):
        raise exc
    monkeypatch.setattr(jq, "load_scene", boom)
    with pytest.raises(type(exc)):
        try_load_scene(root / "fakeobj", 0)
