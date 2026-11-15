"""Synthetic 1024 x 1024 scenes in load_scene's output format, and the audit module, for tests that
must not touch the dataset.

hand_scene() aligns every feature to the 1024 -> 224 grid: 32 source px = 7 cells exactly, so source
pixel p lies in cell floor(p * 7 / 32), and the expected counts can be worked out by hand.
"""
import importlib.util
from pathlib import Path

import numpy as np

from data.jacquard import grasp_corners

N = 1024
TABLE_MM, OBJ_MM = 1500.0, 1450.0

# audit_scene / summarise config (defaults of scripts/audit_train.py, small grids)
AUDIT_CFG = {"size": 224, "taus": [6, 12, 24], "tau_candidates": [12], "telea_radius": 3.0,
             "depth_max": 5000.0, "depth_step": 0.5, "serr_lim": 1000.0, "serr_step": 0.1,
             "small_thresholds": [200], "crops": [1024]}

# Hand scene features: (row y, col x) at 1024 and the 224 cell they fall in.
HOLE_BLOCK = (slice(384, 416), slice(384, 416))     # 32 x 32 sensor hole on the object -> cells 84..90
LIE_BLOCK = (slice(480, 512), slice(480, 512))      # 32 x 32, D_s = D* + 30 mm -> cells 105..111
SHARED_HOLE, SHARED_LIE = (550, 550), (551, 551)    # hole + lie (D* + 100 mm) in one cell (120, 120)
EXACT_TAU = (600, 600)                              # D_s = D* + 12 mm exactly -> cell (131, 131)
ABOVE_CLAMP = (620, 330)                            # valid D_s = 2000 mm (> 1680) -> cell (135, 72)


def load_audit_module():
    """scripts/audit_train.py as a module, without editing or moving the frozen audit code."""
    spec = importlib.util.spec_from_file_location(
        "audit_train", Path(__file__).resolve().parents[1] / "scripts" / "audit_train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _scene(dstar, ds, m_sensor, obj, grasp):
    ds = ds.copy()
    ds[m_sensor.astype(bool)] = np.nan                  # load_scene: NaN at holes
    g = np.array([grasp], np.float64)
    return {"depth_perfect": dstar, "depth_stereo": ds, "m_sensor": m_sensor.astype(np.uint8),
            "obj_mask": obj, "grasps": g, "grasp_corners": grasp_corners(g)}


def hand_scene():
    """Object = source 320..639 square (cells 70..139: 4900 cells at 224, 102400 px at 1024)."""
    obj = np.zeros((N, N), bool)
    obj[320:640, 320:640] = True
    dstar = np.full((N, N), TABLE_MM, np.float32)
    dstar[obj] = OBJ_MM
    ds = dstar.copy()
    ms = np.zeros((N, N), np.uint8)
    ms[HOLE_BLOCK] = 1
    ds[LIE_BLOCK] += 30.0
    ms[SHARED_HOLE] = 1
    ds[SHARED_LIE] += 100.0
    ds[EXACT_TAU] += 12.0                               # 1462 - 1450 = 12 exactly in float32
    ds[ABOVE_CLAMP] = 2000.0
    return _scene(dstar, ds, ms, obj, [480.0, 480.0, 30.0, 40.0, 10.0])


def small_scene():
    """Second grid-aligned scene with a much smaller, heavily holed object, so pooled rates
    (sum of counts / sum of pixels) differ from an unweighted mean of per-scene rates.
    Object = source 640..703 square (cells 140..153: 196 cells, 4096 px); 32 x 32 hole block at
    640..671 (cells 140..146: 49 cells, 1024 px); no lies."""
    obj = np.zeros((N, N), bool)
    obj[640:704, 640:704] = True
    dstar = np.full((N, N), TABLE_MM, np.float32)
    dstar[obj] = OBJ_MM
    ds = dstar.copy()
    ms = np.zeros((N, N), np.uint8)
    ms[640:672, 640:672] = 1
    return _scene(dstar, ds, ms, obj, [690.0, 690.0, 0.0, 10.0, 4.0])


def noisy_scene(seed):
    """Random scene: sloped table, elliptical object, Gaussian stereo noise, gross outliers on both
    sides of the clamp range (up to ~2e5 mm), random and block holes. D* stays in 1390..1520 mm."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float32)
    dstar = (TABLE_MM + 0.02 * (xx - N / 2)).astype(np.float32)
    obj = ((xx - 500) / 180) ** 2 + ((yy - 540) / 120) ** 2 <= 1
    dstar[obj] = (OBJ_MM - 40 * (1 - ((xx[obj] - 500) / 180) ** 2)).astype(np.float32)
    ds = (dstar + rng.normal(0, 6, (N, N))).astype(np.float32)
    idx = rng.choice(N * N, 400, replace=False)
    ds.ravel()[idx[:150]] = rng.uniform(1700, 5000, 150)          # far outliers, above hi
    ds.ravel()[idx[150:300]] = rng.uniform(550, 780, 150)         # near outliers, below lo
    ds.ravel()[idx[300:]] = rng.uniform(1e4, 2e5, 100)            # gross outliers
    ms = (rng.random((N, N)) < 0.02).astype(np.uint8)
    ms[500:560, 450:530] = 1
    return _scene(dstar, ds, ms, obj, [500.0, 540.0, -20.0, 60.0, 12.0])
