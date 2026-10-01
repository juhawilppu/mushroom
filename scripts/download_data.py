"""Download the source datasets for the mushroom habitat maps.

Everything is fetched per municipality in area.MUNICIPALITIES:

- Suomen metsäkeskus (Finnish Forest Centre) open forest resource data
  (metsävarakuviot), municipality-level GeoPackage.
- GTK (Geological Survey of Finland) glaciofluvial / moraine formation
  polygons (eskers etc.), fetched by bounding box from their ArcGIS REST
  service, clipped to the extent of the forest stand data above.
- Maanmittauslaitos 10 m elevation model, the window covering the
  municipality, read by HTTP range request from the openly mirrored
  nationwide VRT rather than downloaded as tiles.
- laji.fi (FinBIF) real sighting coordinates for every mapped species
  within the municipality, for the "reported here" flags on the map.
  Optional: skipped if no LAJI_FI_TOKEN is set in .env.

All are cached under data/ so re-running this script is a no-op unless
the cache is deleted, and adding a municipality fetches only that one.
"""

import json
import zipfile
from pathlib import Path

import geopandas as gpd
import requests
from pyproj import Transformer
from shapely.geometry import Point

import area
import topography as topo
from species import PROFILES, SpeciesProfile

GTK_FORMATIONS_LAYER_URL = (
    "https://gtkdata.gtk.fi/arcgis/rest/services/Rajapinnat/GTK_Maapera_WFS/"
    "MapServer/60/query"
)

LAJI_API = "https://api.laji.fi/v0/warehouse/query/unit/list"


def download_forest_stand_data(municipality: str) -> Path:
    gpkg_path = area.gpkg_path(municipality)
    if not gpkg_path.exists():
        url = f"https://avoin.metsakeskus.fi/aineistot/MV/Kunta/MV_{municipality}.zip"
        zip_path = area.RAW_DIR / f"MV_{municipality}.zip"
        area.RAW_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url} ...")
        resp = requests.get(url, timeout=300)
        resp.raise_for_status()
        zip_path.write_bytes(resp.content)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(gpkg_path.parent)
        print(f"Extracted to {gpkg_path}")
    else:
        print(f"Using cached {gpkg_path}")
    return gpkg_path


def download_gtk_formations(municipality: str, bounds_epsg3067: tuple[float, float, float, float]) -> Path:
    path = area.gtk_formations_path(municipality)
    if path.exists():
        print(f"Using cached {path}")
        return path

    xmin, ymin, xmax, ymax = bounds_epsg3067
    params = {
        "geometry": f"{xmin},{ymin},{xmax},{ymax}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": 3067,
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "DEPOSIT_TYPE_CLASS,DEPOSIT_TYPE,DEPOSIT_TYPE_NAME",
        "outSR": 3067,
        "f": "geojson",
    }
    print(f"Querying GTK glaciofluvial/moraine formations over {municipality} ...")
    resp = requests.get(GTK_FORMATIONS_LAYER_URL, params=params, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    # the service caps the features per response; a silently truncated answer
    # would leave eskers off the map with nothing to say so
    if data.get("exceededTransferLimit") or data.get("properties", {}).get("exceededTransferLimit"):
        raise RuntimeError(f"GTK returned a truncated result for {municipality}; page the query")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(resp.content)
    print(f"Saved {len(data.get('features', []))} formation polygons to {path}")
    return path


def download_dem(municipality: str, stand_gdf: gpd.GeoDataFrame) -> Path:
    """The elevation window for this municipality, cached as a numpy array.

    Only the window is fetched: the source is a nationwide VRT and GDAL reads
    it by range request, so this costs tens of megabytes rather than the
    hundreds of gigabytes the full model would.
    """
    path = area.dem_path(municipality)
    if path.exists():
        print(f"Using cached {path}")
        return path
    print(f"Reading the 10 m elevation model over {municipality} ...")
    dem, _ = topo.read_dem(tuple(stand_gdf.total_bounds), cache_path=path)
    print(f"Saved a {dem.shape[1]}x{dem.shape[0]} elevation grid to {path}")
    return path


def load_laji_token() -> str | None:
    env_path = area.ROOT / ".env"
    if not env_path.exists():
        return None
    for line in env_path.read_text().splitlines():
        if line.startswith("LAJI_FI_TOKEN="):
            return line.split("=", 1)[1].strip()
    return None


def fetch_laji_results(token: str, profile: SpeciesProfile, coordinates: str) -> list[dict]:
    """Every record inside a bounding box, a page at a time: a populous
    municipality can hold more than the 1000 records one page carries."""
    results, page = [], 1
    while True:
        resp = requests.get(LAJI_API, params={
            "target": profile.laji_target,
            "coordinates": coordinates,
            "pageSize": 1000,
            "page": page,
            "access_token": token,
        }, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        results += data["results"]
        if page >= data.get("lastPage", 1):
            return results
        page += 1


def download_laji_sightings(municipality: str, stand_gdf: gpd.GeoDataFrame,
                            profile: SpeciesProfile) -> Path | None:
    """Real sighting coordinates for one species within this municipality, for
    the "reported here" flags on the map. Optional -- skipped without a token."""
    cache_path = area.laji_sightings_path(municipality, profile.slug)
    if cache_path.exists():
        print(f"Using cached {cache_path}")
        return cache_path

    token = load_laji_token()
    if not token:
        print(f"No LAJI_FI_TOKEN in .env -- skipping {profile.name} sighting flags (optional).")
        return None

    minx, miny, maxx, maxy = stand_gdf.to_crs(4326).total_bounds
    coordinates = f"{miny}:{maxy}:{minx}:{maxx}:WGS84"

    print(f"Querying laji.fi for {profile.name} sightings within {municipality} ...")
    results = fetch_laji_results(token, profile, coordinates)

    to_stand_crs = Transformer.from_crs("EPSG:4326", stand_gdf.crs, always_xy=True)
    stand_union = stand_gdf.union_all()
    sightings = []
    for r in results:
        pt = r.get("gathering", {}).get("conversions", {}).get("wgs84CenterPoint")
        if not pt:
            continue
        x, y = to_stand_crs.transform(pt["lon"], pt["lat"])
        point = Point(x, y)
        if not point.within(stand_union):
            continue  # bbox query can return points just outside the municipality
        containing = stand_gdf[stand_gdf.contains(point)]
        standid = int(containing.iloc[0]["standid"]) if len(containing) else None
        sightings.append({
            "lat": pt["lat"], "lon": pt["lon"],
            "date": r["gathering"].get("displayDateTime"),
            "standid": standid,
        })

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(sightings, ensure_ascii=False))
    print(f"Found {len(sightings)} {profile.name} sightings within {municipality}. Cached to {cache_path}")
    return cache_path


def main() -> None:
    for municipality in area.MUNICIPALITIES:
        print(f"\n=== {municipality} ===")
        gpkg_path = download_forest_stand_data(municipality)
        stand = gpd.read_file(gpkg_path, layer="stand")
        download_gtk_formations(municipality, tuple(stand.total_bounds))
        download_dem(municipality, stand)
        for profile in PROFILES.values():
            download_laji_sightings(municipality, stand, profile)


if __name__ == "__main__":
    main()
