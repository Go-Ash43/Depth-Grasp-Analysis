"""data/provenance.py: dataset-root guard and overwrite guard used by every script."""
import pytest

from data.provenance import git_fields, refuse_existing, refuse_inside


def test_refuse_inside(tmp_path):
    root = tmp_path / "ds"
    root.mkdir()
    for p in (root, root / "a" / "b.json", tmp_path / "ds" / ".." / "ds" / "x"):
        with pytest.raises(SystemExit):
            refuse_inside(p, root)
    assert refuse_inside(tmp_path / "ds_other" / "x", root) == (tmp_path / "ds_other" / "x").resolve()


def test_refuse_existing(tmp_path):
    f = tmp_path / "out.json"
    refuse_existing([f], overwrite=False)                # absent: fine
    f.write_text("{}")
    with pytest.raises(SystemExit):
        refuse_existing([f], overwrite=False)
    refuse_existing([f], overwrite=True)


def test_git_fields_keys():
    assert set(git_fields()) == {"git_commit", "git_dirty"}
