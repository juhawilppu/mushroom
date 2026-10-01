"""Calibrate a species' habitat scoring model against real sighting data.

Not used to place pins on the map -- used as a check on the model itself.
Pulls real observation coordinates for one mapped species from laji.fi
(FinBIF) across a region ecologically comparable to the mapped area
(southern Finland; the mapped municipalities alone have too few sightings to
be useful), looks up the actual forest stand each sighting landed in via
Metsakeskus's point-queryable WFS, and compares that distribution (fertility
class, development class, species mix, soil, drainage) against the mapped
area's own stand population as the "available habitat" background. Categories that are
over-represented at real sighting locations relative to background support
the model's weighting for that factor; under-represented ones call it into
question.

Requires a free laji.fi API token in .env as LAJI_FI_TOKEN (see README).

    python scripts/calibrate.py --species suppilovahvero
"""

import argparse
import json
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from pyproj import Transformer
from shapely.geometry import Point, shape

import area
import build_map as bm
import species as sp
import topography as topo
from species import PROFILES, SpeciesProfile

ROOT = Path(__file__).resolve().parent.parent

# Uusimaa + neighbouring Kanta-Hame/Paijat-Hame/western Varsinais-Suomi: the
# same southern-Finland managed-forest zone the mapped area sits in, so comparing
# sighting locations against its own stand population is apples-to-apples
BBOX_WGS84 = "60.0:61.3:22.5:26.0:WGS84"
# A sighting is only evidence about a stand if it can be placed IN that stand,
# and stands here are 1-3 ha. The old 1000 m cap let a record 1 km wide be
# matched to whichever stand happened to sit under its centre point, which
# describes the forest someone walked through rather than the forest the
# mushroom grew in. Tightening this costs less data than it sounds: most
# records are already GPS-precise (at 100 m, 820 of 1063 kantarelli and 661 of
# 904 suppilovahvero records survive).
COORDINATE_ACCURACY_MAX_M = 100

LAJI_API = "https://api.laji.fi/v0/warehouse/query/unit/list"
MK_WFS = "https://avoin.metsakeskus.fi/rajapinnat/v1/stand/ows"

WGS84_TO_TM35FIN = Transformer.from_crs("EPSG:4326", "EPSG:3067", always_xy=True)


def load_token() -> str:
    env_path = ROOT / ".env"
    for line in env_path.read_text().splitlines():
        if line.startswith("LAJI_FI_TOKEN="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("LAJI_FI_TOKEN not found in .env")


def cache_path(profile: SpeciesProfile) -> Path:
    # the accuracy cap is part of the identity of the dataset, so tightening it
    # builds a new cache rather than silently reusing the looser one
    return (ROOT / "data" / "cache" /
            f"{profile.slug}_sightings_with_stands_{COORDINATE_ACCURACY_MAX_M}m.json")


def fetch_sightings(token: str, profile: SpeciesProfile) -> list[dict]:
    sightings = []
    page = 1
    while True:
        resp = requests.get(LAJI_API, params={
            "target": profile.laji_target,
            "coordinates": BBOX_WGS84,
            "coordinateAccuracyMax": COORDINATE_ACCURACY_MAX_M,
            "pageSize": 1000,
            "page": page,
            "access_token": token,
        }, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        for r in data["results"]:
            pt = r.get("gathering", {}).get("conversions", {}).get("wgs84CenterPoint")
            if pt:
                sightings.append({
                    "lat": pt["lat"], "lon": pt["lon"],
                    "date": r["gathering"].get("displayDateTime"),
                    # kept so a dataset can be re-filtered without refetching
                    "accuracy_m": r["gathering"].get("interpretations", {}).get("coordinateAccuracy"),
                })
        print(f"  page {page}/{data['lastPage']}: {len(data['results'])} records")
        if page >= data["lastPage"]:
            break
        page += 1
    return sightings


def lookup_stand(x: float, y: float, buffer_m: float = 25) -> dict | None:
    bbox = f"{x-buffer_m},{y-buffer_m},{x+buffer_m},{y+buffer_m},EPSG:3067"
    resp = requests.get(MK_WFS, params={
        "service": "WFS", "version": "2.0.0", "request": "GetFeature",
        "typeNames": "v1:stand", "outputFormat": "application/json",
        "srsName": "EPSG:3067", "bbox": bbox,
    }, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    point = Point(x, y)
    for feature in data.get("features", []):
        geom = shape(feature["geometry"])
        if geom.contains(point):
            return feature["properties"]
    return None


def build_sightings_with_stands(token: str, profile: SpeciesProfile) -> pd.DataFrame:
    path = cache_path(profile)
    if path.exists():
        print(f"Using cached {path}")
        return pd.DataFrame(json.loads(path.read_text()))

    print(f"Fetching {profile.name} sightings from laji.fi ...")
    sightings = fetch_sightings(token, profile)
    print(f"{len(sightings)} sightings with coordinates")

    rows = []
    for i, s in enumerate(sightings):
        x, y = WGS84_TO_TM35FIN.transform(s["lon"], s["lat"])
        try:
            stand = lookup_stand(x, y)
        except requests.RequestException as e:
            print(f"  [{i}] WFS lookup failed: {e}")
            stand = None
        if stand:
            rows.append({**s, **stand})
        if (i + 1) % 50 == 0:
            print(f"  looked up {i + 1}/{len(sightings)} ({len(rows)} matched a stand)")
        time.sleep(0.05)  # be polite to the WFS

    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(df.to_json(orient="records"))
    print(f"Matched {len(df)}/{len(sightings)} sightings to a forest stand. Cached to {path}")
    return df


def sighting_terrain(profile: SpeciesProfile, sightings: pd.DataFrame) -> pd.DataFrame:
    """Landform metrics at each sighting, read from the same elevation model
    the map uses, with the same kernels -- otherwise presence and background
    would not be measuring the same thing."""
    path = (ROOT / "data" / "cache" /
            f"{profile.slug}_sighting_terrain_{COORDINATE_ACCURACY_MAX_M}m.json")
    if path.exists():
        print(f"Using cached {path}")
        return pd.DataFrame(json.loads(path.read_text()))

    print(f"Reading terrain at {len(sightings)} {profile.name} sighting locations ...")
    x, y = WGS84_TO_TM35FIN.transform(sightings["lon"].to_numpy(), sightings["lat"].to_numpy())
    metrics = topo.metrics_at_points(x, y)
    df = pd.DataFrame({k: v.astype(float) for k, v in metrics.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(df.to_json(orient="records"))
    print(f"Cached to {path}")
    return df


def compare_terrain(presence: pd.DataFrame, background: pd.DataFrame) -> None:
    """Terrain metrics are continuous, so they are compared as binned
    distributions rather than by code table."""
    bins = {
        "tpi_small (m vs 150 m surroundings)": [-99, -3, -1, 1, 3, 99],
        "tpi_large (m vs 500 m surroundings)": [-99, -6, -2, 2, 6, 99],
        "slope (deg)": [0, 2, 4, 7, 12, 90],
        "northness (+1 = due north)": [-1.01, -0.5, 0, 0.5, 1.01],
    }
    for label, edges in bins.items():
        column = label.split(" ")[0]
        cut = lambda s: pd.cut(s, edges)  # noqa: E731
        p = cut(presence[column]).value_counts(normalize=True).mul(100)
        b = cut(background[column]).value_counts(normalize=True).mul(100)
        both = pd.concat([p, b], axis=1, keys=["presence_%", "background_%"]).fillna(0)
        both["enrichment"] = (both["presence_%"] / both["background_%"].replace(0, float("nan"))).round(2)
        print(f"\n=== {label} ===")
        print(both.round(1).sort_index().to_string())


def area_background(profile: SpeciesProfile) -> pd.DataFrame:
    """Available habitat: every stand in the mapped area this species' model does not
    exclude outright. The exclusions differ per species (kantarelli drops
    every mire type, suppilovahvero keeps korpi), so the background has to be
    built per species too."""
    layers = bm.load_layers()
    scored = bm.score_stands(*layers, profile=profile)
    return scored[~scored["excluded"]].copy()


def compare(field: str, presence: pd.Series, background: pd.Series, labels: dict | None = None) -> None:
    p = presence.value_counts(normalize=True)
    b = background.value_counts(normalize=True)
    both = pd.concat([p, b], axis=1, keys=["presence_%", "background_%"]).fillna(0) * 100
    both["enrichment"] = (both["presence_%"] / both["background_%"].replace(0, float("nan"))).round(2)
    both = both.sort_values("presence_%", ascending=False).round(1)
    if labels:
        both.index = [labels.get(i, i) for i in both.index]
    print(f"\n=== {field} ===")
    print(both)


def to_code_str(series: pd.Series) -> pd.Series:
    """WFS numeric codes -> the same string-coded space growthplacedata uses."""
    return series.dropna().astype(float).astype(int).astype(str)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", default="kantarelli", choices=sorted(PROFILES),
                        help="which species' model to calibrate (default: kantarelli)")
    profile = PROFILES[parser.parse_args().species]

    token = load_token()
    df = build_sightings_with_stands(token, profile)
    bg = area_background(profile)

    print(f"\n=== {profile.name} ({profile.latin}) ===")
    print(f"{len(df)} sightings matched to a stand; {len(bg)} {area.AREA_NAME} stands as background")

    compare("Kasvupaikka (fertilityclass)", to_code_str(df["FERTILITYCLASS"]), bg["fertilityclass"], sp.FERTILITY_LABELS)
    compare("Kehitysluokka (developmentclass)", df["DEVELOPMENTCLASS"], bg["developmentclass"], sp.DEVELOPMENT_LABELS)
    compare("Vallitseva puulaji (main species)", to_code_str(df["MAINTREESPECIES"]).map(sp.TREESPECIES_LABELS).fillna("Muu"),
            bg["dominant_species"].map(sp.TREESPECIES_LABELS).fillna("Muu"))
    compare("Maalaji (soiltype)", to_code_str(df["SOILTYPE"]), bg["soiltype"], sp.SOIL_LABELS)
    compare("Maaryhma (subgroup, 1=kangas)", to_code_str(df["SUBGROUP"]), bg["subgroup"])
    compare("Kuivatustilanne (drainagestate)", to_code_str(df["DRAINAGESTATE"]), bg["drainagestate"], sp.DRAINAGE_LABELS)

    # the canopy-density curve is calibrated straight off these bands: where
    # sightings pile up relative to background is where the curve should peak
    bands = [0, 150, 300, 600, 900, 1400, 2200, 1e9]
    names = ["<150", "150-300", "300-600", "600-900", "900-1400", "1400-2200", ">2200"]
    compare("Runkoluku (stemcount, kpl/ha)",
            pd.cut(df["STEMCOUNT"], bands, labels=names),
            pd.cut(bg["stemcount"], bands, labels=names))

    compare_terrain(sighting_terrain(profile, df), bg)

    print("\n=== mean PROPORTIONSPRUCE / PROPORTIONPINE / PROPORTIONOTHER at sighting locations ===")
    print(df[["PROPORTIONSPRUCE", "PROPORTIONPINE", "PROPORTIONOTHER"]].mean().round(3))


if __name__ == "__main__":
    main()
