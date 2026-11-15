"""try_load_scene: expected data errors become skip records; code bugs and a vanished drive must crash."""
import cv2
import pytest

import data.jacquard as jq
from data.jacquard import try_load_scene

GOOD = "560.71;478.67;20.33;106.5;25.03"


@pytest.fixture
def root(tmp_path, monkeypatch):
    """Fake dataset root with one object folder; data_root() points at it."""
    (tmp_path / "fakeobj").mkdir()
    monkeypatch.setattr(jq, "data_root", lambda: str(tmp_path))
    return tmp_path


def _grasps(root, text):
    (root / "fakeobj" / "0_fakeobj_grasps.txt").write_text(text)
    return root / "fakeobj"


def _assert_skip(res, error_type):
    scene, rec = res
    assert scene is None
    assert set(rec) == {"obj_dir", "view", "error_type", "message"}
    assert rec["error_type"] == error_type and rec["view"] == 0


def test_success_passes_scene_through(root, monkeypatch):
    sentinel = {"obj_id": "fakeobj"}
    monkeypatch.setattr(jq, "load_scene", lambda obj_dir, view: sentinel)
    scene, rec = try_load_scene(root / "fakeobj", 0)
    assert scene is sentinel and rec is None


def test_missing_image_is_skipped(root):
    obj = _grasps(root, GOOD)                    # valid grasps, but no image files -> FileNotFoundError
    _assert_skip(try_load_scene(obj, 0), "FileNotFoundError")


def test_missing_grasp_file_is_skipped(root):
    _assert_skip(try_load_scene(root / "fakeobj", 0), "FileNotFoundError")


def test_bad_grasp_row_is_skipped(root):
    obj = _grasps(root, f"{GOOD}\n10;20;30;-4;5")
    _assert_skip(try_load_scene(obj, 0), "SceneDataError")


def test_empty_grasps_is_skipped(root):
    obj = _grasps(root, "0;0;0;0;0\n")
    _assert_skip(try_load_scene(obj, 0), "EmptyGraspsError")


def test_cv2_error_is_skipped(root, monkeypatch):
    def boom(obj_dir, view):
        raise cv2.error("decode failed")
    monkeypatch.setattr(jq, "load_scene", boom)
    _assert_skip(try_load_scene(root / "fakeobj", 0), "error")


@pytest.mark.parametrize("exc", [ValueError("shape bug"), TypeError("bad arg"), KeyError("k"), IndexError("i")])
def test_code_bugs_propagate(root, monkeypatch, exc):
    def boom(obj_dir, view):
        raise exc
    monkeypatch.setattr(jq, "load_scene", boom)
    with pytest.raises(type(exc)):
        try_load_scene(root / "fakeobj", 0)


def test_explicit_root_resolves_relative_obj_dir(tmp_path, monkeypatch):
    other = tmp_path / "other_dataset"
    (other / "fakeobj").mkdir(parents=True)
    (other / "fakeobj" / "0_fakeobj_grasps.txt").write_text("0;0;0;0;0\n")
    monkeypatch.setattr(jq, "data_root", lambda: str(tmp_path / "default_root_unused"))
    _assert_skip(try_load_scene("fakeobj", 0, root=other), "EmptyGraspsError")   # found under `other`


def test_absolute_obj_dir_outside_root_is_a_code_bug(tmp_path):
    (tmp_path / "a" / "fakeobj").mkdir(parents=True)
    (tmp_path / "b").mkdir()
    with pytest.raises(ValueError, match="not inside the dataset root"):
        try_load_scene(tmp_path / "a" / "fakeobj", 0, root=tmp_path / "b")


@pytest.mark.parametrize("rel", ["../escaped", "fakeobj/../../escaped"])
def test_dotdot_cannot_escape_root(root, rel):
    (root.parent / "escaped").mkdir(exist_ok=True)
    with pytest.raises(ValueError, match="not inside the dataset root"):
        try_load_scene(rel, 0)
    with pytest.raises(ValueError, match="not inside the dataset root"):
        try_load_scene(root / rel, 0)                          # absolute form, same escape


def test_symlink_out_of_root_is_rejected(root):
    outside = root.parent / "outside_obj"
    outside.mkdir(exist_ok=True)
    (root / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="not inside the dataset root"):
        try_load_scene("link", 0)


def test_binary_grasp_file_is_skipped(root):
    (root / "fakeobj" / "0_fakeobj_grasps.txt").write_bytes(b"\xff\xfe\x00\x81garbage")
    _assert_skip(try_load_scene(root / "fakeobj", 0), "SceneDataError")


def test_drive_check_uses_explicit_root_not_default(tmp_path, monkeypatch):
    # default root exists, the scene's own root does not: must abort, not skip
    monkeypatch.setattr(jq, "data_root", lambda: str(tmp_path))
    gone = tmp_path / "unplugged_drive"
    with pytest.raises(RuntimeError, match="missing"):
        try_load_scene("fakeobj", 0, root=gone)


def test_missing_root_aborts_instead_of_skipping(tmp_path, monkeypatch):
    gone = tmp_path / "unplugged_drive"           # never created
    monkeypatch.setattr(jq, "data_root", lambda: str(gone))
    with pytest.raises(RuntimeError, match="missing"):
        try_load_scene(gone / "fakeobj", 0)
