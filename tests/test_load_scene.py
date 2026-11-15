"""load_scene enforces what the dataset audit (docs/proj_v3.md) checks: object-mask format and
grasp centres inside the image. Builds a tiny fake scene on disk (no dataset files)."""
import cv2
import numpy as np
import pytest

from data.jacquard import IMG_SIZE, SceneDataError, load_scene

N = IMG_SIZE
GOOD = "512.0;512.0;30.0;80.0;20.0"


def obj_mask_png(value=255):
    m = np.zeros((N, N), np.uint8)
    m[400:600, 450:580] = value
    return m


def make_scene(tmp_path, mask=None, grasps=GOOD, rgb=None, depth=None, stereo=None):
    obj = tmp_path / "fakeobj"
    obj.mkdir(exist_ok=True)
    rgb = np.full((N, N, 3), 128, np.uint8) if rgb is None else rgb
    depth = np.full((N, N), 0.8, np.float32) if depth is None else depth
    if stereo is None:
        stereo = np.full((N, N), 0.8, np.float32)
        stereo[10:20, 10:20] = -1.0
    assert cv2.imwrite(str(obj / "0_fakeobj_RGB.png"), rgb)
    assert cv2.imwrite(str(obj / "0_fakeobj_mask.png"), obj_mask_png() if mask is None else mask)
    assert cv2.imwrite(str(obj / "0_fakeobj_perfect_depth.tiff"), depth)
    assert cv2.imwrite(str(obj / "0_fakeobj_stereo_depth.tiff"), stereo)
    (obj / "0_fakeobj_grasps.txt").write_text(grasps)
    return obj


@pytest.mark.parametrize("value", [1, 255])
def test_valid_scene_loads(tmp_path, value):
    s = load_scene(make_scene(tmp_path, mask=obj_mask_png(value)), 0)
    assert s["obj_mask"].dtype == bool and s["obj_mask"].sum() == 200 * 130
    assert s["m_sensor"].sum() == 100 and s["grasps"].shape == (1, 5)


def test_three_channel_mask_rejected(tmp_path):
    m = np.repeat(obj_mask_png()[..., None], 3, axis=2)
    with pytest.raises(SceneDataError, match="single-channel"):
        load_scene(make_scene(tmp_path, mask=m), 0)


def test_mixed_mask_values_rejected(tmp_path):
    m = obj_mask_png()
    m[0, 0] = 128
    with pytest.raises(SceneDataError, match="not in"):
        load_scene(make_scene(tmp_path, mask=m), 0)


@pytest.mark.parametrize("value", [1, 255])
def test_inverted_mask_rejected(tmp_path, value):
    inverted = np.where(obj_mask_png(value) > 0, 0, value).astype(np.uint8)   # background = "object"
    with pytest.raises(SceneDataError, match="polarity"):
        load_scene(make_scene(tmp_path, mask=inverted), 0)


def test_object_touching_border_kept(tmp_path):
    m = obj_mask_png()
    m[0, :300] = 255                                           # object touches the top edge: ~7% of border
    assert load_scene(make_scene(tmp_path, mask=m), 0)["obj_mask"][0, 0]


@pytest.mark.parametrize("fill,word", [(0, "empty"), (255, "full-frame")])
def test_empty_or_full_mask_rejected(tmp_path, fill, word):
    with pytest.raises(SceneDataError, match=word):
        load_scene(make_scene(tmp_path, mask=np.full((N, N), fill, np.uint8)), 0)


# pixel-centre convention (data/geometry.py): inside iff -0.5 <= x < N - 0.5
@pytest.mark.parametrize("line", [f"{N - 0.5};512;30;80;20", "512;-0.51;30;80;20", f"512;{N + 3};30;80;20"])
def test_grasp_centre_outside_image_rejected(tmp_path, line):
    with pytest.raises(SceneDataError, match="outside"):
        load_scene(make_scene(tmp_path, grasps=f"{GOOD}\n{line}"), 0)


@pytest.mark.parametrize("line", [f"{N - 0.51};0;30;80;20", "-0.5;-0.5;30;80;20"])
def test_grasp_centre_on_edge_pixels_kept(tmp_path, line):
    s = load_scene(make_scene(tmp_path, grasps=f"{GOOD}\n{line}"), 0)
    assert s["grasps"].shape == (2, 5)


@pytest.mark.parametrize("rgb", [np.full((N, N), 128, np.uint8),          # grayscale
                                 np.full((N, N, 4), 128, np.uint8)])      # RGBA
def test_non_rgb_colour_image_rejected(tmp_path, rgb):
    with pytest.raises(SceneDataError, match="3-channel"):
        load_scene(make_scene(tmp_path, rgb=rgb), 0)


def test_corrupt_image_is_scene_data_error_not_missing(tmp_path):
    obj = make_scene(tmp_path)
    (obj / "0_fakeobj_RGB.png").write_bytes(b"\x89PNG not really a png")
    with pytest.raises(SceneDataError, match="decode"):
        load_scene(obj, 0)


def test_missing_image_is_file_not_found(tmp_path):
    obj = make_scene(tmp_path)
    (obj / "0_fakeobj_RGB.png").unlink()
    with pytest.raises(FileNotFoundError):
        load_scene(obj, 0)


@pytest.mark.parametrize("which", ["stereo", "depth"])
@pytest.mark.parametrize("value", [0.0, 3e36])            # 0 = sentinel collision; 3e36 m overflows float32 mm
def test_zero_or_overflowing_depth_rejected(tmp_path, which, value):
    d = np.full((N, N), 0.8, np.float32)
    d[500, 500] = value
    with pytest.raises(SceneDataError):
        load_scene(make_scene(tmp_path, **{which: d}), 0)
