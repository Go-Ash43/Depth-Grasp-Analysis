"""All-zero grasp rows (16 rows / 14 scenes found by the dataset audit, docs/proj_v3.md) are dropped; others raise."""
import numpy as np
import pytest

from data.jacquard import EmptyGraspsError, SceneDataError, load_scene, parse_grasps

GOOD = "560.71;478.67;20.33;106.5;25.03"


def test_zero_row_dropped_and_counted():
    g, n = parse_grasps(f"{GOOD}\n0;0;0;0;0\n0.0;0.0;0.0;0.0;0.0\n{GOOD}", return_dropped=True)
    assert n == 2 and g.shape == (2, 5)
    np.testing.assert_array_equal(g[0], [560.71, 478.67, 20.33, 106.5, 25.03])


def test_default_return_is_array_without_zero_rows():
    g = parse_grasps(f"0;0;0;0;0\n{GOOD}")
    assert isinstance(g, np.ndarray) and g.shape == (1, 5)


@pytest.mark.parametrize("line", ["10;20;30;0;5", "10;20;30;40;0", "10;20;30;-4;5", "0;0;0;0;5", "10;20;nan;40;5",
                                  "10;20;30;40", "10;20;30;40;5;6", "10;20;abc;40;5"])
def test_other_degenerate_rows_raise(line):
    # SceneDataError specifically: try_load_scene skips only that, a plain ValueError must crash
    with pytest.raises(SceneDataError):
        parse_grasps(f"{GOOD}\n{line}")


def test_theta_zero_at_origin_with_size_is_kept():
    # only the full all-zero pattern is dropped; x=y=theta=0 with a real size is a valid grasp
    assert parse_grasps("0;0;0;40;10").shape == (1, 5)


@pytest.mark.parametrize("text", ["", "\n  \n", "0;0;0;0;0\n", "0;0;0;0;0\n0;0;0;0;0"])
def test_scene_without_valid_grasps_is_dismissed(tmp_path, text):
    obj = tmp_path / "fakeobj"
    obj.mkdir()
    (obj / "0_fakeobj_grasps.txt").write_text(text)   # grasps are read first, so no images are needed
    with pytest.raises(EmptyGraspsError):
        load_scene(obj, 0)
