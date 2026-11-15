"""scripts/audit_report.py: the histogram quantile and clamp count behind the frozen Phase-0 values."""
import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "audit_report", Path(__file__).resolve().parents[1] / "scripts" / "audit_report.py")
ar = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ar)

EDGES = np.arange(0.0, 11.0)          # 10 unit bins on [0, 10)
UNIFORM = np.full(10, 10)             # 10 values per bin, n = 100


def test_quantile_uniform_interpolates():
    h = (EDGES, UNIFORM, 0, 0)
    assert ar.quantile(h, 0.5) == 5.0
    assert ar.quantile(h, 0.25) == 2.5
    assert ar.quantile(h, 0.01) == 0.1


def test_quantile_outside_range_is_none():
    h = (EDGES, UNIFORM, 10, 10)      # n = 120: lowest 10 and highest 10 are off the bins
    assert ar.quantile(h, 0.05) is None and ar.quantile(h, 0.95) is None
    assert ar.quantile(h, 0.5) == 5.0


def test_count_outside_edges():
    counts = UNIFORM.copy()
    h = (EDGES, counts, 3, 4)
    # below 2: under (3) + bins [0,1), [1,2) = 23; value exactly 2 sits in bin [2,3) and is kept
    # at/above 8: over (4) + bins [8,9), [9,10) = 24 (includes a value exactly at hi: conservative)
    assert ar.count_outside(h, 2.0, 8.0) == (23, 24)
    assert ar.count_outside(h, 0.0, 10.0) == (3, 4)
