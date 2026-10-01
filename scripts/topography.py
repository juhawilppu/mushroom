"""Terrain metrics from the national 10 m elevation model.

The forest inventory describes the stand -- what grows there, on what soil --
but says nothing about where the stand sits in the landscape. Two stands with
identical inventory rows can be a dry crest and the damp hollow below it, and
that is exactly the distinction a chanterelle map and a funnel-chanterelle map
need in order to disagree about anything. This module supplies it.

Source: Maanmittauslaitos 10 m korkeusmalli (lidar-derived ground model,
EPSG:3067), read straight from the openly mirrored nationwide VRT at funet by
HTTP range request -- so no gigabytes of tiles are downloaded, only the windows
actually being looked at.

Metrics, all computed on the same grid:

    tpi_small   elevation minus the mean elevation within 150 m. Negative = a
                local hollow that collects water and cold air, positive = a
                local rise that sheds both.
    tpi_large   the same at 500 m, which captures landform position (valley
                bottom vs hillside vs crest) rather than micro-relief.
    slope       degrees. Flat ground holds water; steep ground drains it.
    northness   +1 on a due-north-facing slope, -1 due south. North-facing
                ground stays cooler and damper through a dry spell.

Slope and northness are derivatives, so they are taken from a lightly smoothed
copy of the elevation model (SLOPE_SMOOTH_M): on the raw 10 m grid a single
cell can be a boulder or a ditch bank, and the value read at one point would
say more about that than about the hillside it sits on.

Every metric is read as a POINT value -- at a sighting, or at a stand's
centroid. Averaging them over a stand polygon was tried first and is a trap:
it shrinks micro-relief and aspect toward zero, so stands look flatter and
more neutral than the sighting points they are compared against, and the
comparison then finds a difference that is purely an artefact of the two sides
being measured differently.

TPI is used rather than a proper topographic wetness index because TWI needs
flow accumulation over an entire catchment, which cannot be computed from the
small windows the sighting calibration reads. TPI plus slope captures the same
"does water gather here or run off" distinction locally, and both are
computable identically for a single point and for a whole municipality, which
is what keeps calibration and map honest with each other.
"""

import json
import os
from pathlib import Path

import numpy as np

# GDAL should not try to list the remote directory before opening a file
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")

import rasterio  # noqa: E402  (import after the GDAL config above)
from rasterio.features import rasterize  # noqa: E402
from rasterio.windows import from_bounds  # noqa: E402

DEM_VRT = (
    "/vsicurl/https://www.nic.funet.fi/index/geodata/mml/dem10m/dem10m_direct.vrt"
)
DEM_RES_M = 10.0

TPI_SMALL_M = 150
TPI_LARGE_M = 500
SLOPE_SMOOTH_M = 50
# Enough margin around a point for the largest kernel to fit whole
POINT_MARGIN_M = TPI_LARGE_M + 200

METRICS = ("tpi_small", "tpi_large", "slope", "northness")


def read_dem(bounds: tuple[float, float, float, float], margin_m: float = POINT_MARGIN_M,
             cache_path: Path | None = None) -> tuple[np.ndarray, "rasterio.Affine"]:
    """Elevation for a bounding box in EPSG:3067, padded so that focal windows
    have real data to work with at the edges of the area of interest."""
    if cache_path and cache_path.exists():
        cached = np.load(cache_path)
        return cached["dem"], rasterio.Affine(*cached["transform"])

    xmin, ymin, xmax, ymax = bounds
    with rasterio.open(DEM_VRT) as src:
        window = from_bounds(xmin - margin_m, ymin - margin_m,
                             xmax + margin_m, ymax + margin_m, src.transform)
        dem = src.read(1, window=window).astype("float32")
        transform = src.window_transform(window)
        dem[dem == src.nodata] = np.nan

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, dem=dem, transform=np.array(transform[:6]))
    return dem, transform


def _box_mean(a: np.ndarray, radius_px: int) -> np.ndarray:
    """Mean over a (2r+1) square window, ignoring NaN, via summed-area tables.

    Written out rather than pulled from scipy because it has to be NaN-aware
    (lakes and the sea are nodata in the elevation model) and because the same
    code then runs on a 2600x2000 municipality grid and on a 141x141 window
    around a single sighting, with identical results.
    """
    valid = np.isfinite(a)
    filled = np.where(valid, a, 0.0).astype("float64")

    def integral(x):
        return np.pad(x, ((1, 0), (1, 0))).cumsum(0).cumsum(1)

    sums, counts = integral(filled), integral(valid.astype("float64"))
    rows, cols = a.shape
    r0 = np.clip(np.arange(rows) - radius_px, 0, rows)
    r1 = np.clip(np.arange(rows) + radius_px + 1, 0, rows)
    c0 = np.clip(np.arange(cols) - radius_px, 0, cols)
    c1 = np.clip(np.arange(cols) + radius_px + 1, 0, cols)

    def window_total(table):
        return (table[np.ix_(r1, c1)] - table[np.ix_(r0, c1)]
                - table[np.ix_(r1, c0)] + table[np.ix_(r0, c0)])

    total, n = window_total(sums), window_total(counts)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, total / n, np.nan)


def terrain_metrics(dem: np.ndarray, res_m: float = DEM_RES_M) -> dict[str, np.ndarray]:
    smoothed = _box_mean(dem, max(1, int(round(SLOPE_SMOOTH_M / res_m))))
    # rows run north to south, so the row derivative is the southward one and
    # the north-facing component of the downhill direction is +dz_drow
    dz_drow, dz_dcol = np.gradient(smoothed, res_m)
    gradient = np.hypot(dz_dcol, dz_drow)
    with np.errstate(invalid="ignore", divide="ignore"):
        northness = np.where(gradient > 0, dz_drow / gradient, 0.0)
    return {
        "tpi_small": (dem - _box_mean(dem, int(round(TPI_SMALL_M / res_m)))).astype("float32"),
        "tpi_large": (dem - _box_mean(dem, int(round(TPI_LARGE_M / res_m)))).astype("float32"),
        "slope": np.degrees(np.arctan(gradient)).astype("float32"),
        "northness": northness.astype("float32"),
    }


def sample_points(metrics: dict[str, np.ndarray], transform, xs, ys) -> dict[str, np.ndarray]:
    """Metric values at map coordinates (EPSG:3067)."""
    inverse = ~transform
    cols, rows = inverse * (np.asarray(xs, dtype="float64"), np.asarray(ys, dtype="float64"))
    rows = np.floor(rows).astype(int)
    cols = np.floor(cols).astype(int)
    shape = next(iter(metrics.values())).shape
    inside = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])
    out = {}
    for name, grid in metrics.items():
        values = np.full(len(rows), np.nan, dtype="float32")
        values[inside] = grid[rows[inside], cols[inside]]
        out[name] = values
    return out


def metrics_at_points(xs, ys, batch_km: float = 3.0) -> dict[str, np.ndarray]:
    """Terrain metrics at scattered points anywhere in Finland.

    Used for the calibration sightings, which are spread over the whole of
    southern Finland. Points are grouped into tiles first so that a cluster of
    sightings in one forest costs one windowed read rather than fifty.
    """
    xs = np.asarray(xs, dtype="float64")
    ys = np.asarray(ys, dtype="float64")
    out = {name: np.full(len(xs), np.nan, dtype="float32") for name in METRICS}
    size = batch_km * 1000
    keys = np.stack([np.floor(xs / size), np.floor(ys / size)], axis=1)

    for key in np.unique(keys, axis=0):
        idx = np.where((keys == key).all(axis=1))[0]
        bounds = (xs[idx].min(), ys[idx].min(), xs[idx].max(), ys[idx].max())
        try:
            dem, transform = read_dem(bounds)
        except Exception as exc:  # a tile can be missing at the coast
            print(f"  DEM read failed for tile {key.tolist()}: {exc}")
            continue
        values = sample_points(terrain_metrics(dem), transform, xs[idx], ys[idx])
        for name in METRICS:
            out[name][idx] = values[name]
    return out
