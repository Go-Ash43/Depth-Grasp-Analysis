"""Jaw-contact zone candidates, G0 / G90 controls, exact-area rasterisation and G_theta
(SPEC_hypotheses "Placements" and H4 feature G_theta; notes/MAJOR_v1.md section 6).

Grasp (x, y, theta, L = opening, W = jaw) at the resolution of the flag mask (224). Closing axis
a = (cos t, sin t), plate direction p = (-sin t, cos t), (x right, y down), theta as-is.
Each candidate gives two zones, one per jaw: distance along a from the centre in [d0, d1]
(on the + side and mirrored on the - side), full jaw width W along p:
  Z-end4  [L/4, L/2]            Z-end3  [L/6, L/2]            Z-band  [max(L/2 - band, 0), L/2]
The two zones never overlap (d0 >= 0). For predictions and diagnostic stand-ins W = opening / 2
(the decoder's jaw rule), so the zones have the geometry the analyses will actually use.

Rasterisation is exact: coverage[i, j] = area(zone ∩ cell), cell = [j - .5, j + .5] x [i - .5, i + .5]
(pixel-centre convention). Parts outside the frame are dropped for rasterisation only (frac_out is
recorded); the match IoU never clips. Binary zone cells (coverage >= 0.5) exist for placements only.

G_theta = flagged area / in-frame area of the UNION of both zones (coverage-weighted, no threshold).
The grasp is empty if that in-frame area < empty_min (0.5 cell): G_theta = NaN. Per-jaw empty flags
are diagnostics. G0 = disk at the centre with the union's continuous area (regular 64-gon, radius
corrected so its area is exact). G90 = zones constructed at theta + 90 deg.
"""
import numpy as np
import shapely

from data.mask_ops import _as_mask

CANDIDATES = ("Z-end4", "Z-end3", "Z-band")


def _grasp(g):
    g = np.asarray(g, np.float64).reshape(5)
    if not np.isfinite(g).all() or g[3] <= 0 or g[4] <= 0:
        raise ValueError(f"grasp must be finite with opening, jaw > 0: {g.tolist()}")
    return g


def with_predicted_jaw(g):
    """Replace the jaw by opening / 2 (decoder rule) for diagnostic stand-ins."""
    g = np.array(g, np.float64).reshape(5)
    g[4] = g[3] / 2.0
    return g


def zone_range(opening, candidate, band_px=2.0):
    half = opening / 2.0
    if candidate == "Z-end4":
        return opening / 4.0, half
    if candidate == "Z-end3":
        return opening / 6.0, half
    if candidate == "Z-band":
        return max(half - float(band_px), 0.0), half
    raise ValueError(f"unknown zone candidate {candidate!r}; choose from {CANDIDATES}")


def zone_polygons(grasp, candidate, band_px=2.0):
    """[+ jaw zone, - jaw zone] as shapely polygons in the grasp's pixel coordinates."""
    x, y, t, L, W = _grasp(grasp)
    r = np.deg2rad(t)
    a = np.array([np.cos(r), np.sin(r)])
    p = np.array([-np.sin(r), np.cos(r)])
    c = np.array([x, y])
    d0, d1 = zone_range(L, candidate, band_px)
    out = []
    for sgn in (1.0, -1.0):
        pts = [c + sgn * d0 * a - W / 2 * p, c + sgn * d1 * a - W / 2 * p,
               c + sgn * d1 * a + W / 2 * p, c + sgn * d0 * a + W / 2 * p]
        out.append(shapely.Polygon(pts))
    return out


def g90_polygons(grasp, candidate, band_px=2.0):
    g = _grasp(grasp).copy()
    g[2] += 90.0
    return zone_polygons(g, candidate, band_px)


def g0_polygon(grasp, area, segments=64):
    """Regular `segments`-gon at the grasp centre with area exactly `area` (circumradius corrected)."""
    x, y = _grasp(grasp)[:2]
    if not area > 0:
        raise ValueError("G0 area must be > 0")
    r = np.sqrt(area / (segments / 2.0 * np.sin(2.0 * np.pi / segments)))
    ang = 2.0 * np.pi * np.arange(segments) / segments
    return shapely.Polygon(np.c_[x + r * np.cos(ang), y + r * np.sin(ang)])


def coverage(geom, hw):
    """(H, W) float64 exact area of geom inside each cell; 0 outside the frame."""
    h, w = hw
    cov = np.zeros((h, w), np.float64)
    g = shapely.intersection(geom, shapely.box(-0.5, -0.5, w - 0.5, h - 0.5))
    if g.is_empty or g.area <= 0:
        return cov
    xmin, ymin, xmax, ymax = g.bounds
    j0, j1 = max(int(np.floor(xmin + 0.5)), 0), min(int(np.floor(xmax + 0.5)), w - 1)
    i0, i1 = max(int(np.floor(ymin + 0.5)), 0), min(int(np.floor(ymax + 0.5)), h - 1)
    jj, ii = np.meshgrid(np.arange(j0, j1 + 1, dtype=np.float64), np.arange(i0, i1 + 1, dtype=np.float64))
    boxes = shapely.box(jj - 0.5, ii - 0.5, jj + 0.5, ii + 0.5)
    cov[i0:i1 + 1, j0:j1 + 1] = shapely.area(shapely.intersection(boxes, g))
    return cov


def region_stats(polys, flag, empty_min=0.5):
    """Coverage-weighted flagged fraction over the union of `polys` (in-frame). flag: (H, W) 0/1,
    1 = flagged. -> dict(g, cov, area_cont, area_in, frac_out, empty, flag_area).
    flag_area = flagged in-frame area, measured BEFORE the empty threshold: an empty region can
    still touch flags (contact must be judged from flag_area, never from a finite g)."""
    f = _as_mask(flag, name="flag")
    union = shapely.union_all(list(polys))
    cov = coverage(union, f.shape)
    area_cont = float(union.area)
    area_in = float(cov.sum())
    flag_area = float((cov * f).sum())
    empty = area_in < empty_min
    g = flag_area / area_in if not empty else float("nan")
    return {"g": g, "cov": cov, "area_cont": area_cont, "area_in": area_in, "flag_area": flag_area,
            "frac_out": 1.0 - area_in / area_cont if area_cont > 0 else float("nan"), "empty": bool(empty)}


def zone_features(grasp, flag, candidate, band_px=2.0, empty_min=0.5, g0_segments=64):
    """G_theta, G0, G90 and per-jaw diagnostics for one grasp on one flag mask (same resolution)."""
    zones = zone_polygons(grasp, candidate, band_px)
    main = region_stats(zones, flag, empty_min)
    f = _as_mask(flag, name="flag")
    jaws = []
    for z in zones:                                        # per-jaw resolution diagnostics
        cz = coverage(z, f.shape)
        a_in = float(cz.sum())
        jaws.append({"area_cont": float(z.area), "area_in": a_in, "cells_binary": int((cz >= 0.5).sum()),
                     "empty": bool(a_in < empty_min)})
    g0 = region_stats([g0_polygon(grasp, main["area_cont"], g0_segments)], flag, empty_min)
    g90 = region_stats(g90_polygons(grasp, candidate, band_px), flag, empty_min)
    return {"g_theta": main["g"], "g0": g0["g"], "g90": g90["g"], "empty": main["empty"],
            "area_cont": main["area_cont"], "area_in": main["area_in"], "frac_out": main["frac_out"],
            "flag_area": main["flag_area"], "jaws": jaws,
            "jaw_area_in": [j["area_in"] for j in jaws], "jaw_empty": [j["empty"] for j in jaws],
            "overlap_area": float(zones[0].intersection(zones[1]).area),
            "cells_binary": int((main["cov"] >= 0.5).sum()), "g0_cells_binary": int((g0["cov"] >= 0.5).sum()),
            "cov": main["cov"]}


def selected_zone(cfg):
    """The frozen zone candidate for analyses (Module 2 placements, H4 G_theta). Raises while
    configs/labels.json zones.selected is null: the provisional choice must never be used silently
    before Stage B (RULES R7) passes."""
    z = cfg["zones"]
    sel = z.get("selected")
    if sel is None:
        raise RuntimeError(f"zones.selected is null (provisional {z.get('provisional')!r}): run Stage B "
                           f"(RULES R7) and freeze the zone by an approved diff before using it")
    if sel not in z["candidates"] or sel not in CANDIDATES:
        raise ValueError(f"zones.selected {sel!r} is not a known candidate")
    return sel


def binary_zone(cov, thr=0.5):
    """Placement cells (uint8, 1 = in zone)."""
    return (np.asarray(cov) >= thr).astype(np.uint8)
