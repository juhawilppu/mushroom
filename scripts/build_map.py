"""Score the forest stands of the mapped area (see area.py) for mushroom
habitat suitability and render the result as one standalone Leaflet HTML map
with a species switcher.

The habitat heuristics themselves live in `scripts/species.py`, one profile
per mushroom; this module is the engine that applies a profile to the forest
inventory and draws the result. Kantarelli wants dry, light-flooded,
well-drained mineral soil near eskers; suppilovahvero wants damp, shady,
moss-floored spruce forest and is at home on peat -- so the two maps
disagree about most of the municipality, which is the point of having both.

Both species ship in a single HTML file: stand geometry is by far the
largest part of the payload and is identical between them, so it is written
once and each species contributes only its own scores. Attributes are
shipped as inventory codes and turned into English labels in the browser,
and the stands themselves are packed (see to_page_payload), which keeps
the page to a few megabytes even with several municipalities on it -- this
thing gets loaded over mobile data, in a forest.

Run scripts/download_data.py first.
"""

import json

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import mapping

import area
import species as sp
import topography as topo
from species import PROFILES, SpeciesProfile

OUTPUT_GEOJSON = area.ROOT / "output" / "scored_stands.geojson"
OUTPUT_HTML = area.ROOT / "output" / "mushroom_map.html"

CURRENT_TREESTAND_CLASS = "2"  # "Nykytilan puusto" = current, as opposed to inventory/forecast

# Only these make it onto the map, in this order: the array index is what the
# browser gets instead of the category name.
MAPPED_CATEGORIES = ["excellent", "high", "medium"]

# Factors whose ratio is species-specific and therefore shipped per species,
# in the order the browser expects them after [score, category].
SCORED_FACTORS = ["fertility", "development", "species", "light", "soil", "terrain"]

# ~1 m at this latitude, against stand outlines already simplified to 2 m --
# invisible on the map, and it takes a third off the size of the file
COORD_DECIMALS = 5


def load_municipality(municipality: str):
    gpkg_path = area.gpkg_path(municipality)
    stand = gpd.read_file(gpkg_path, layer="stand")[["standid", "area", "geometry"]]
    stand = stand.join(compute_terrain(stand, municipality))
    stand["municipality"] = municipality
    growthplace = gpd.read_file(gpkg_path, layer="growthplacedata")[
        ["standid", "maingroup", "subgroup", "fertilityclass", "soiltype", "drainagestate"]
    ]
    treestand = gpd.read_file(gpkg_path, layer="treestand")
    treestand = treestand[treestand["treestandclass"] == CURRENT_TREESTAND_CLASS][
        ["treestandid", "standid", "developmentclass"]
    ]
    # stemcount (stems/ha) is used as the canopy-density proxy: a stand can
    # have high basal area from a few big old trees (open, light) or the same
    # basal area from many small crowded ones (dark) -- stem density tells
    # those apart, basal area alone does not
    treestandsummary = gpd.read_file(gpkg_path, layer="treestandsummary")[
        ["treestandid", "stemcount"]
    ]
    treestratum = gpd.read_file(gpkg_path, layer="treestratum")[
        ["treestandid", "treespecies", "basalarea"]
    ]
    return stand, growthplace, treestand, treestandsummary, treestratum


def load_layers():
    """Every municipality's inventory tables, stacked into one population.

    Stand and tree-stand ids are national, and no stand is published by two
    municipalities, so stacking is all it takes -- and scoring one stacked
    population, rather than each municipality on its own, is what lets the
    ranking treat the whole area as one map.
    """
    per_municipality = [load_municipality(m) for m in area.MUNICIPALITIES]
    stand, *tables = (pd.concat(parts, ignore_index=True) for parts in zip(*per_municipality))
    return (gpd.GeoDataFrame(stand, geometry="geometry", crs=per_municipality[0][0].crs), *tables)


def compute_species_mix(treestand: pd.DataFrame, treestratum: pd.DataFrame,
                        profile: SpeciesProfile) -> pd.DataFrame:
    current_ids = set(treestand["treestandid"])
    tt = treestratum[treestratum["treestandid"].isin(current_ids)].copy()
    tt["basalarea"] = tt["basalarea"].fillna(0)
    tt["weight"] = tt["treespecies"].map(profile.species_weight).fillna(profile.species_weight_default)

    totals = tt.groupby("treestandid")["basalarea"].sum().rename("total_ba")
    weighted = (tt["basalarea"] * tt["weight"]).groupby(tt["treestandid"]).sum().rename("weighted_ba")

    dominant_idx = tt.groupby("treestandid")["basalarea"].idxmax()
    dominant = tt.loc[dominant_idx, ["treestandid", "treespecies"]].set_index("treestandid")

    # Gini-Simpson diversity index (1 - sum of squared species shares) over
    # actual basal-area composition: 0 = pure monoculture, closer to 1 = a
    # genuine "sekametsä" of several well-balanced species. Independent of any
    # species' host weights -- this measures mixedness itself, so it is the
    # same number on both maps even though they value it differently.
    species_ba = tt.groupby(["treestandid", "treespecies"])["basalarea"].sum()
    shares = species_ba / species_ba.groupby(level="treestandid").transform("sum")
    diversity = (1 - (shares ** 2).groupby(level="treestandid").sum()).rename("diversity_index")

    mix = pd.concat([totals, weighted, diversity], axis=1).join(dominant["treespecies"])
    mix["species_fraction"] = (mix["weighted_ba"] / mix["total_ba"]).clip(upper=1).fillna(0)
    mix["diversity_index"] = mix["diversity_index"].fillna(0)
    return mix.rename(columns={"treespecies": "dominant_species"}).reset_index()


# Column names the site factors are read from. Given as a mapping so the same
# scoring code can be pointed at the differently-named columns of the
# Metsakeskus WFS rows that scripts/validate.py scores sightings from.
SITE_COLUMNS = {
    "fertilityclass": "fertilityclass",
    "developmentclass": "developmentclass",
    "soiltype": "soiltype",
    "drainagestate": "drainagestate",
    "stemcount": "stemcount",
    "tpi": "tpi_large",
    "slope": "slope",
}


def site_factor_points(df: pd.DataFrame, profile: SpeciesProfile,
                       columns: dict[str, str] | None = None) -> dict[str, pd.Series]:
    """Points for the four factors that come straight off a stand's own
    inventory attributes, before the species/mixture terms are added.

    Split out of score_stands so that validate.py can score real sighting
    locations with exactly the code the map uses, rather than a lookalike that
    could drift away from it.
    """
    c = columns or SITE_COLUMNS
    points = profile.factor_points
    # Canopy density, via the profile's own stems/ha response curve. It peaks
    # in a band rather than rising monotonically in either direction: a nearly
    # treeless stand has all the light in the world but no living mycorrhizal
    # host, so it must never score as ideal.
    stemcount = df[c["stemcount"]]
    suitability = pd.Series(
        np.interp(stemcount, profile.light.stemcount_knots, profile.light.suitability_knots),
        index=df.index,
    ).where(stemcount.notna())
    # Landform: how the stand sits relative to its surroundings, and how
    # steeply. Two responses combined by the profile's tpi_weight rather than
    # scored separately, because they describe one thing between them -- where
    # water goes -- and splitting them would give terrain two votes.
    terrain = profile.terrain
    tpi_fit = np.interp(df[c["tpi"]], terrain.tpi_knots, terrain.tpi_suitability)
    slope_fit = np.interp(df[c["slope"]], terrain.slope_knots, terrain.slope_suitability)
    landform = pd.Series(
        terrain.tpi_weight * tpi_fit + (1 - terrain.tpi_weight) * slope_fit,
        index=df.index,
    ).where(df[c["tpi"]].notna() & df[c["slope"]].notna())

    return {
        "terrain": points["terrain"] * landform.fillna(0.6),
        "fertility": df[c["fertilityclass"]].map(profile.fertility_points)
                       .fillna(profile.default_fertility_points),
        "development": df[c["developmentclass"]].map(profile.development_points)
                       .fillna(profile.default_development_points),
        "soil": (df[c["soiltype"]].map(profile.soil_points).fillna(profile.soil_default)
                 * df[c["drainagestate"]].map(profile.drainage_multiplier).fillna(profile.drainage_default)),
        "light": points["light"] * suitability.fillna(0.4),
    }


def is_excluded(df: pd.DataFrame, profile: SpeciesProfile,
                columns: dict[str, str] | None = None,
                maingroup: str = "maingroup", subgroup: str = "subgroup") -> pd.Series:
    """Stands this species' model writes off outright: wrong land class, the
    wrong kind of mire, no forest floor yet, or bare rock."""
    c = columns or SITE_COLUMNS
    return (
        (df[maingroup] != "1")
        | df[subgroup].isin(profile.excluded_subgroup)
        | df[c["developmentclass"]].isin(sp.EXCLUDED_DEVELOPMENT)
        | df[c["fertilityclass"]].isin(profile.excluded_fertility)
    )


def score_stands(stand, growthplace, treestand, treestandsummary, treestratum,
                 profile: SpeciesProfile) -> gpd.GeoDataFrame:
    species_mix = compute_species_mix(treestand, treestratum, profile)
    points = profile.factor_points

    df = stand.merge(growthplace, on="standid", how="left")
    df = df.merge(treestand, on="standid", how="left")
    df = df.merge(treestandsummary, on="treestandid", how="left")
    df = df.merge(species_mix, on="treestandid", how="left")

    excluded = is_excluded(df, profile)

    site = site_factor_points(df, profile)
    fertility_score, development_score, soil_score = site["fertility"], site["development"], site["soil"]
    terrain_score = site["terrain"]
    # the species contribution is split in two: host quality (how good the
    # dominant trees are as a mycorrhizal partner, per the profile's weights)
    # and a separate "sekametsä" mixture term. A stand can score well on one
    # without the other, and the two species weigh them very differently.
    species_quality_score = points["species"] * df["species_fraction"].fillna(0.3)
    mixture_score = points["mixture"] * df["diversity_index"].fillna(0)
    light_score = site["light"]

    df["fertility_ratio"] = (fertility_score / points["fertility"]).clip(upper=1).round(2)
    df["development_ratio"] = (development_score / points["development"]).clip(upper=1).round(2)
    df["species_ratio"] = (species_quality_score / points["species"]).round(2)
    df["mixture_ratio"] = df["diversity_index"].fillna(0).round(2)
    df["soil_ratio"] = (soil_score / points["soil"]).clip(upper=1).round(2)
    df["light_ratio"] = (light_score / points["light"]).round(2)
    df["terrain_ratio"] = (terrain_score / points["terrain"]).round(2)

    # Limiting-factor penalty (Liebig's law of the minimum): habitat is
    # limited by its worst attribute, not its average. A plain sum lets a
    # stand offset a fatal weakness -- no light, say -- with two maxed-out
    # factors, which both overrates it ecologically and made some
    # "Excellent" stands (green everywhere, maxed nowhere) score below
    # "High" stands carrying a red factor. Each factor is measured against
    # its own green threshold, so "green everywhere" means no penalty at all.
    weakest = pd.concat(
        [(df[f"{factor}_ratio"] / profile.green_thresholds[factor]).clip(upper=1)
         for factor in sp.LIMITING_FACTORS],
        axis=1,
    ).min(axis=1).fillna(0)

    raw = (
        fertility_score + development_score + species_quality_score
        + mixture_score + soil_score + light_score + terrain_score
    )
    df["score"] = (raw * (sp.LIMITING_FLOOR + (1 - sp.LIMITING_FLOOR) * weakest)).round(1)
    df.loc[excluded, "score"] = 0
    df["excluded"] = excluded

    return gpd.GeoDataFrame(df, geometry="geometry", crs=stand.crs)


def compute_terrain(stand: gpd.GeoDataFrame, municipality: str) -> pd.DataFrame:
    """Landform metrics per stand, from the national 10 m elevation model.

    Purely geometric like the esker lookup, so it is computed once for every
    stand and handed to whichever species profile wants it. Cached, because it
    reads a few tens of megabytes of elevation over HTTP.
    """
    cache_path = area.terrain_path(municipality)
    if cache_path.exists():
        cached = np.load(cache_path)
        return pd.DataFrame({name: cached[name] for name in topo.METRICS}, index=stand.index)

    print(f"Computing terrain metrics for {municipality} from the 10 m elevation model ...")
    dem, transform = topo.read_dem(tuple(stand.total_bounds), cache_path=area.dem_path(municipality))
    # sampled at the centroid, not averaged over the polygon: the sightings
    # this is calibrated against are single points, and a polygon average is
    # not the same quantity (see the note in topography.py)
    centroids = stand.geometry.centroid
    values = topo.sample_points(topo.terrain_metrics(dem), transform,
                                centroids.x.to_numpy(), centroids.y.to_numpy())
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **values)
    return pd.DataFrame(values, index=stand.index)


def compute_near_esker(stand: gpd.GeoDataFrame) -> pd.Series:
    """Proximity to glaciofluvial deposits -- eskers, sandurs, deltas,
    ice-contact deposits. The point is the *substrate*, not elevation: these
    are sorted sand and gravel laid down by glacial meltwater rivers, so they
    drain exceptionally well, which is the soil condition Finnish sources tie
    to chanterelle-friendly forest.

    The GTK layer also carries moraine (unsorted till) and littoral deposits.
    Buffering all of them put 62% of Karkkila "near an esker", which made the
    factor almost meaningless -- and moraine is the opposite of the sorted,
    free-draining substrate we are actually looking for. Restricting to
    genuine glaciofluvial deposits brings it to a selective 25%.

    Purely geometric, so it is computed once and handed to whichever species
    profiles actually use it (suppilovahvero does not -- damp ground is the
    whole point for it, and free-draining sand is not where it fruits).
    """
    paths = [area.gtk_formations_path(m) for m in area.MUNICIPALITIES]
    if not any(path.exists() for path in paths):
        return pd.Series(False, index=stand.index)
    # each municipality's query is a bounding box, so neighbours overlap and a
    # formation near a border arrives twice -- harmless, as they are unioned
    formations = pd.concat(
        [gpd.read_file(path).to_crs(stand.crs) for path in paths if path.exists()],
        ignore_index=True,
    )
    deposit_class = formations["DEPOSIT_TYPE_CLASS"].astype(str)
    glaciofluvial = deposit_class.str.startswith("1") & ~deposit_class.str.startswith("1.5")
    buffered = formations[glaciofluvial].buffer(sp.ESKER_BUFFER_M).union_all()
    return stand.geometry.centroid.within(buffered)


def add_esker_bonus(gdf: gpd.GeoDataFrame, profile: SpeciesProfile) -> gpd.GeoDataFrame:
    if not profile.uses_esker:
        return gdf
    bonus = gdf["near_esker"] & (~gdf["excluded"])
    gdf.loc[bonus, "score"] = gdf.loc[bonus, "score"] + profile.esker_points
    return gdf


def normalize_scores(gdf: gpd.GeoDataFrame, profile: SpeciesProfile) -> gpd.GeoDataFrame:
    """Rescale the raw point total onto a true 0-100 scale.

    The esker bonus used to be added and then clipped at 100, which silently
    penalised exactly the best stands: once the other factors already summed
    near the 100-point maximum, part of the +10 was thrown away, so an
    excellent stand near an esker got less credit for it than a mediocre one.
    Dividing by the profile's real theoretical maximum keeps every factor's
    calibrated weight intact and keeps the displayed "x/100" honest -- and it
    is what lets two species with different point budgets share a scale.
    """
    gdf["score"] = (gdf["score"] / profile.max_raw_score * 100).round(1)
    return gdf


def categorize(gdf: gpd.GeoDataFrame, profile: SpeciesProfile) -> gpd.GeoDataFrame:
    """"Excellent" is a hard rule, not a percentile: every single badged
    factor has to be green (its own "good" threshold, matching what the
    popup actually shows), so an "Excellent" stand is explainable purely by
    "look, everything is green" -- no exceptions or partial credit.

    Everything else ranks against everything else rather than using fixed
    score thresholds: the forest land here is overwhelmingly mesic,
    coarse-mineral-soil spruce/mixed forest, so the raw weighted score
    clusters densely and a fixed cutoff would flag most of the area
    as "high". Relative ranking keeps the map useful for choosing where to go,
    and it is per species, so each map ranks stands against the habitat that
    species actually has available. It is across the whole mapped area, not
    per municipality, so a colour means the same thing on both sides of a
    border.
    """
    gdf["category"] = "excluded"
    non_excluded = ~gdf["excluded"]

    all_green = gdf["near_esker"].copy() if profile.uses_esker else pd.Series(True, index=gdf.index)
    for factor, threshold in profile.green_thresholds.items():
        all_green &= gdf[f"{factor}_ratio"] >= threshold
    excellent = non_excluded & all_green
    gdf.loc[excellent, "category"] = "excellent"

    rest = non_excluded & ~excellent
    ranked = gdf.loc[rest, "score"].rank(pct=True)
    gdf.loc[rest, "category"] = pd.cut(
        ranked, bins=[0, 0.5, 0.85, 1.0], labels=["low", "medium", "high"], include_lowest=True
    )
    return gdf


def build_species_frame(layers, near_esker: pd.Series, profile: SpeciesProfile) -> gpd.GeoDataFrame:
    scored = score_stands(*layers, profile=profile)
    scored["near_esker"] = near_esker.reindex(scored.index).fillna(False)
    scored = add_esker_bonus(scored, profile)
    scored = normalize_scores(scored, profile)
    return categorize(scored, profile)


def round_coords(obj, decimals: int = COORD_DECIMALS):
    if isinstance(obj, (list, tuple)):
        return [round_coords(o, decimals) for o in obj]
    return round(obj, decimals)


def to_geojson_dict(frames: dict[str, gpd.GeoDataFrame]) -> dict:
    """One FeatureCollection carrying every species' scores.

    Geometry and the stand's own inventory attributes are identical between
    species, so they are written once per stand; each species adds a compact
    [score, category, ...factor ratios] array under its own short key, and is
    simply absent from stands its own map does not show. A stand is kept if
    *any* species ranks it, and the browser hides the ones the active species
    has nothing to say about.
    """
    base = next(iter(frames.values()))
    keep = np.zeros(len(base), dtype=bool)
    for frame in frames.values():
        keep |= frame["category"].isin(MAPPED_CATEGORIES).to_numpy()

    shown = base.loc[keep].copy()
    # centroid computed in the planar CRS (before simplify/reproject) so it's a
    # true geometric centroid, used as the Google Maps navigation destination
    centroid_4326 = shown.geometry.centroid.to_crs(4326)
    shown["lat"] = centroid_4326.y.round(6)
    shown["lon"] = centroid_4326.x.round(6)
    shown["geometry"] = shown["geometry"].simplify(2.0)
    shown = shown.to_crs(4326)

    def code(value):
        return None if pd.isna(value) else str(value)

    def number(value, decimals=None):
        if pd.isna(value):
            return None
        return round(float(value), decimals) if decimals is not None else float(value)

    blocks = {}
    for slug, frame in frames.items():
        rows = frame.loc[keep]
        blocks[slug] = [
            None if cat not in MAPPED_CATEGORIES else
            [round(float(score), 1), MAPPED_CATEGORIES.index(cat)]
            + [round(float(r), 2) for r in ratios]
            for cat, score, *ratios in zip(
                rows["category"], rows["score"],
                *[rows[f"{factor}_ratio"] for factor in SCORED_FACTORS],
            )
        ]

    features = []
    for i, (_, row) in enumerate(shown.iterrows()):
        # age, area and basal area don't feed any score (development class
        # already captures stand-age effects) -- not shipped. Neither is
        # anything that can be derived in the browser from what is.
        props = {
            "id": int(row["standid"]),
            "lat": row["lat"], "lon": row["lon"],
            "fc": code(row["fertilityclass"]),
            "dc": code(row["developmentclass"]),
            "ts": code(row["dominant_species"]),
            "st": code(row["soiltype"]),
            "ds": code(row["drainagestate"]),
            "div": number(row["mixture_ratio"], 2),
            "stem": None if pd.isna(row["stemcount"]) else int(row["stemcount"]),
            "esk": int(bool(row["near_esker"])),
            "tpi": number(row["tpi_large"], 1),
            "slp": number(row["slope"], 1),
        }
        for slug, block in blocks.items():
            if block[i] is not None:
                props[PROFILES[slug].map_key] = block[i]
        features.append({
            "type": "Feature",
            "properties": props,
            "geometry": {
                "type": row.geometry.geom_type,
                "coordinates": round_coords(mapping(row.geometry)["coordinates"]),
            },
        })
    return {"type": "FeatureCollection", "features": features}


# Properties the page reads, in the order each stand's packed row carries them.
# The stand id is not among them: it is in the GeoJSON export for GIS use, but
# nothing on the page reads it.
PAGE_FIELDS = (["lat", "lon", "fc", "dc", "ts", "st", "ds", "div", "stem", "esk", "tpi", "slp"]
               + [profile.map_key for profile in PROFILES.values()])


def encode_ring(ring: list) -> str:
    """One polygon ring in Google's encoded polyline format, latitude first.

    Each coordinate is stored as its difference from the previous one, at the
    same 1e-5 degree precision COORD_DECIMALS rounds to, packed five bits to a
    printable character. Stand outlines are short hops between nearby points,
    so a coordinate costs two or three characters instead of the eight or nine
    of a decimal number.
    """
    scale = 10 ** COORD_DECIMALS
    chars, previous = [], (0, 0)
    for lon, lat in ring:
        point = (round(lat * scale), round(lon * scale))
        for value, last in zip(point, previous):
            delta = value - last
            delta = ~(delta << 1) if delta < 0 else delta << 1
            while delta >= 0x20:
                chars.append(chr((0x20 | (delta & 0x1F)) + 63))
                delta >>= 5
            chars.append(chr(delta + 63))
        previous = point
    return "".join(chars)


def to_page_payload(geojson_dict: dict) -> dict:
    """The FeatureCollection repacked for the page, which unpacks it on load.

    Written out as plain GeoJSON, the stands of even three municipalities
    come to over 20 MB -- far too much to load over mobile data in the middle
    of a forest.
    Nearly all of that is repetition: every stand spells out the same property
    names, and every vertex a full decimal coordinate. Here the names are
    listed once, each stand is a row of values in that order, and each ring is
    an encoded polyline.
    """
    features = geojson_dict["features"]
    for feature in features:
        if feature["geometry"]["type"] != "Polygon":
            raise ValueError(f"the page only unpacks Polygons, got {feature['geometry']['type']}")
    return {
        "fields": PAGE_FIELDS,
        "rows": [[f["properties"].get(name) for name in PAGE_FIELDS] for f in features],
        "rings": [[encode_ring(ring) for ring in f["geometry"]["coordinates"]] for f in features],
    }


def load_sightings_geojson(profile: SpeciesProfile) -> dict:
    sightings = []
    for municipality in area.MUNICIPALITIES:
        path = area.laji_sightings_path(municipality, profile.slug)
        if path.exists():
            sightings += json.loads(path.read_text())
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [s["lon"], s["lat"]]},
            "properties": {"date": s["date"]},
        }
        for s in sightings
    ]
    return {"type": "FeatureCollection", "features": features}


def species_config() -> list[dict]:
    """Everything the page needs to know about a species: how to read its
    score block, where its badges turn green, and what to call things."""
    return [
        {
            "slug": profile.slug,
            "name": profile.name,
            "latin": profile.latin,
            "intro": profile.intro,
            "key": profile.map_key,
            "green": profile.green_thresholds,
            "light": {
                "row": profile.light.row_label,
                "good": profile.light.label_good,
                "mid": profile.light.label_mid,
                "poor": profile.light.label_poor,
                "sparseStems": profile.light.sparse_stems,
                "sparse": profile.light.label_sparse,
            },
            "terrain": {"row": profile.terrain.row_label},
            "esker": list(profile.esker_row_labels) if profile.uses_esker else None,
            "sightings": load_sightings_geojson(profile),
        }
        for profile in PROFILES.values()
    ]


def label_config() -> dict:
    return {
        "fertility": sp.FERTILITY_LABELS,
        "development": sp.DEVELOPMENT_LABELS,
        "soil": sp.SOIL_LABELS,
        "drainage": sp.DRAINAGE_LABELS,
        "species": sp.TREESPECIES_LABELS,
        "mixtureBands": sp.MIXTURE_BANDS,
        "mixtureFallback": sp.MIXTURE_FALLBACK,
        "landformBands": sp.LANDFORM_BANDS,
        "landformFallback": sp.LANDFORM_FALLBACK,
        "slopeBands": sp.SLOPE_BANDS,
    }


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Mushroom map</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<style>
  :root {
    --font: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    --ink: #1d2420;        /* body text */
    --ink-soft: #56605a;   /* secondary text */
    --ink-faint: #8a938c;   /* labels, captions */
    --rule: #e9ece7;       /* hairline separators */
    --forest: #1f5136;     /* the app's own accent, used for actions */
  }
  html, body { margin: 0; height: 100%; font-family: var(--font); color: var(--ink);
               -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale; }
  #map { height: 100%; background: #eceee9; }
  /* OSM's own greens and browns compete with the score colours for attention.
     Muting the tiles keeps every road, path and label legible while letting
     the stand polygons read as the data layer they are. */
  .leaflet-tile-pane { filter: saturate(.45) brightness(1.06) contrast(.94); }

  /* --- species switcher ------------------------------------------------ */
  /* Top centre: the one control on the map, and the thing that decides what
     every colour on screen means, so it sits above the map rather than
     inside the legend where it would read as another caption. */
  .species-switch { position: fixed; top: calc(10px + env(safe-area-inset-top, 0px));
                    left: 50%; transform: translateX(-50%); z-index: 1000;
                    display: flex; gap: 3px; padding: 3px; max-width: calc(100vw - 24px);
                    background: rgba(255,255,255,.94); border-radius: 999px;
                    -webkit-backdrop-filter: blur(12px); backdrop-filter: blur(12px);
                    box-shadow: 0 2px 14px rgba(23,35,28,.18), inset 0 0 0 1px rgba(23,35,28,.06); }
  .species-switch button { flex: 0 1 auto; min-width: 0; appearance: none; border: none;
                           padding: 9px 17px; border-radius: 999px; background: none;
                           font: inherit; font-size: 14px; font-weight: 600; color: var(--ink-soft);
                           white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
                           cursor: pointer; transition: background .15s, color .15s; }
  .species-switch button:hover { color: var(--ink); }
  .species-switch button[aria-selected="true"] { background: var(--forest); color: #fff;
                           box-shadow: 0 1px 4px rgba(31,81,54,.32); }

  /* --- legend ---------------------------------------------------------- */
  .legend { position: fixed; left: 0; right: 0; bottom: 0; z-index: 1000;
            background: rgba(255,255,255,.95);
            -webkit-backdrop-filter: blur(12px); backdrop-filter: blur(12px);
            border-top: 1px solid var(--rule); border-radius: 16px 16px 0 0;
            box-shadow: 0 -10px 30px rgba(23,35,28,.13);
            padding: 14px 46px calc(14px + env(safe-area-inset-bottom, 0px)) 18px; }
  .legend-title { display: block; font-size: 11px; font-weight: 700; letter-spacing: .09em;
                  text-transform: uppercase; color: var(--ink-faint); }
  .legend-row { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 18px; margin-top: 10px; }
  .legend-row > span { display: inline-flex; align-items: center; gap: 7px;
                       font-size: 14px; line-height: 1.3; color: var(--ink-soft); }
  .legend .swatch { flex: none; width: 15px; height: 15px; border-radius: 4px;
                    box-shadow: inset 0 0 0 1px rgba(0,0,0,.16); }
  .legend small { display: block; margin-top: 11px; padding-top: 10px; border-top: 1px solid var(--rule);
                  font-size: 12px; line-height: 1.5; color: var(--ink-faint); }
  .legend-close { position: absolute; top: 10px; right: 10px; width: 30px; height: 30px;
                  border: none; border-radius: 50%; background: #f1f3ef; color: var(--ink-faint);
                  font-size: 17px; line-height: 30px; text-align: center; cursor: pointer;
                  transition: background .15s, color .15s; }
  .legend-close:hover { background: #e4e8e2; color: var(--ink); }

  .my-location-dot { width: 16px; height: 16px; border-radius: 50%; background: #1a73e8;
                     border: 2px solid white; box-shadow: 0 0 0 2px rgba(26,115,232,.45), 0 1px 4px rgba(0,0,0,.3); }

  /* --- popup ----------------------------------------------------------- */
  .leaflet-popup-content-wrapper { padding: 0; border-radius: 14px; overflow: hidden;
                                   box-shadow: 0 12px 36px rgba(23,35,28,.22); }
  .leaflet-popup-content { margin: 0; font-size: 15px; line-height: 1.5; min-width: 258px; }
  .leaflet-popup-close-button {
    color: var(--ink-faint) !important; background: none;
    top: 10px !important; right: 8px !important; font-size: 20px !important;
    width: 26px !important; height: 26px !important; line-height: 24px !important; text-align: center;
  }
  .leaflet-popup-close-button:hover { color: var(--ink) !important; }
  /* the category's three tones (fill, ink, tint) arrive as custom properties
     set inline per stand, so one palette in JS drives map, legend and popup */
  .popup-header { padding: 13px 46px 12px 16px; background: var(--cat-tint);
                  border-top: 4px solid var(--cat);
                  display: flex; justify-content: space-between; align-items: center; gap: 12px; }
  .popup-cat { display: inline-flex; align-items: center; gap: 8px; font-size: 12px; font-weight: 700;
               letter-spacing: .07em; text-transform: uppercase; color: var(--cat-ink); }
  .popup-cat::before { content: ""; flex: none; width: 10px; height: 10px; border-radius: 3px; background: var(--cat); }
  .popup-score { font-size: 21px; font-weight: 700; letter-spacing: -.02em; white-space: nowrap;
                 color: var(--cat-ink); font-variant-numeric: tabular-nums; }
  .popup-score em { font-style: normal; font-size: 13px; font-weight: 600; opacity: .6; }
  .popup-body { padding: 4px 16px 15px; }
  /* the per-factor dot leads its row rather than trailing the value: values
     wrap freely ("Near an esker or ice-marginal formation" does), and trailing dots
     ended up orphaned on a line of their own. Leading dots also line up into
     one scannable column of greens and reds. */
  .popup-field { display: flex; align-items: flex-start; gap: 9px;
                 padding: 8px 0; border-bottom: 1px solid var(--rule); }
  .popup-field:last-child { border-bottom: none; }
  .popup-label { display: block; color: var(--ink-faint); font-size: 10.5px; font-weight: 700;
                 text-transform: uppercase; letter-spacing: .08em; }
  .popup-value { display: block; margin-top: 3px; }
  .score-badge { flex: none; width: 10px; height: 10px; margin-top: 3px; border-radius: 50%;
                 box-shadow: inset 0 0 0 1px rgba(0,0,0,.12); }
  .score-badge.blank { background: none; box-shadow: none; }
  .score-badge.good { background: #1f7a3f; }
  .score-badge.mid { background: #dfa32c; }
  .score-badge.poor { background: #cd5b52; }
  .gmaps-btn { display: flex; align-items: center; justify-content: center; gap: 7px;
               margin-top: 14px; padding: 11px 14px; background: var(--forest); color: white !important;
               border-radius: 10px; text-decoration: none; font-size: 14px; font-weight: 600;
               box-shadow: 0 2px 8px rgba(31,81,54,.26); transition: background .15s, transform .08s; }
  .gmaps-btn:hover { background: #18402b; }
  .gmaps-btn:active { transform: translateY(1px); }
  .sighting-flag { font-size: 20px; line-height: 1; text-shadow: 0 1px 2px rgba(0,0,0,.5); }
  /* .leaflet-popup-content carries no padding of its own (the stand popup
     supplies its own), so the sighting popup brings a padded wrapper */
  .popup-note { padding: 14px 40px 14px 16px; }
  .popup-note b { display: block; margin-bottom: 3px; font-size: 12px; font-weight: 700;
                  letter-spacing: .07em; text-transform: uppercase; color: var(--forest); }
  .popup-note span { color: var(--ink-soft); font-size: 14px; }

  .leaflet-container { font-family: var(--font); }
  .leaflet-control-attribution { font-size: 11px; background: rgba(255,255,255,.82) !important; }
</style>
</head>
<body>
<div id="map"></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<script>
const PACKED = __STANDS__;     // the stands, packed by to_page_payload in build_map.py
const SPECIES = __SPECIES__;   // one entry per mushroom, in switcher order
const LABELS = __LABELS__;     // inventory code -> English, expanded here rather
                               // than repeated on every stand in the payload
const MID = __MID_THRESHOLD__;
const STORAGE_KEY = "karkkila-sienikartta-species";

// Each species' scores ride along as a compact array under its own key:
// [score, category index, then one ratio per scored factor].
const SCORE = 0, CATEGORY = 1;
const RATIO = { fertility: 2, development: 3, species: 4, light: 5, soil: 6, terrain: 7 };
const CATEGORIES = __CATEGORIES__;

// An ordered ramp rather than three unrelated hues: the colour cools and
// darkens as the category improves (chanterelle gold -> yellow-green -> deep
// pine), so which green outranks which is readable straight off the map. The
// previous palette paired bright lime with dark green and said nothing about
// their order. Per category: `fill` paints the polygon and legend swatch,
// `line` its outline, `ink` carries text on the popup's `tint` background.
// Shared by both species: the ramp means "probability", and giving each
// mushroom its own hues would make the two maps harder to compare, not easier.
const PALETTE = {
  excellent: { fill: "#15653a", line: "#0e4527", ink: "#0f4a2a", tint: "#e6f1ea" },
  high:      { fill: "#5aa84a", line: "#3d7c31", ink: "#2f6b28", tint: "#ecf5e8" },
  medium:    { fill: "#e0a33c", line: "#a8741d", ink: "#8a5a11", tint: "#fbf2e1" },
};
const FALLBACK = { fill: "#9aa39c", line: "#6f7872", ink: "#3c4340", tint: "#f0f1ef" };
const pal = (category) => PALETTE[category] || FALLBACK;
const CATEGORY_LABELS = { excellent: "Excellent", high: "High", medium: "Moderate" };
// opacity climbs with the ranking too, so the ordering survives even where the
// basemap underneath is busy
const FILL_OPACITY = { excellent: 0.72, high: 0.58, medium: 0.45 };

// One ring from Google's encoded polyline format (latitude first, 1e-5
// degrees), back into GeoJSON's [lon, lat] pairs.
function decodeRing(encoded) {
  const ring = [];
  let i = 0, lat = 0, lon = 0;
  while (i < encoded.length) {
    const deltas = [0, 0].map(() => {
      let shift = 0, value = 0, chunk;
      do {
        chunk = encoded.charCodeAt(i++) - 63;
        value |= (chunk & 0x1f) << shift;
        shift += 5;
      } while (chunk >= 0x20);
      return value & 1 ? ~(value >> 1) : value >> 1;
    });
    lat += deltas[0];
    lon += deltas[1];
    ring.push([lon / 1e5, lat / 1e5]);
  }
  return ring;
}

// Unpacked once into the plain GeoJSON that everything below works with: the
// packing only exists to keep the download small.
const STANDS = {
  type: "FeatureCollection",
  features: PACKED.rows.map((row, i) => ({
    type: "Feature",
    properties: Object.fromEntries(PACKED.fields.map((name, j) => [name, row[j]])),
    geometry: { type: "Polygon", coordinates: PACKED.rings[i].map(decodeRing) },
  })),
};

// Everything mapped is inside these few municipalities, so this is the area
// the map is ever useful in. Built from the stand centroids rather than from a
// layer, so it covers both species regardless of which one is showing.
const DATA_BOUNDS = L.latLngBounds(
  STANDS.features.map((f) => [f.properties.lat, f.properties.lon])
).pad(0.02);

const map = L.map('map', { preferCanvas: true, zoomControl: false });
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: '&copy; OpenStreetMap contributors',
  maxZoom: 19,
}).addTo(map);

function mixtureLabel(diversity) {
  for (const [threshold, label] of LABELS.mixtureBands) {
    if (diversity >= threshold) return label;
  }
  return LABELS.mixtureFallback;
}

// Where the stand sits in the landscape, from the elevation model: how high
// it stands relative to the ground within 500 m, and how steeply it falls.
// The slope half is only spelled out when there is a slope worth mentioning.
function landformLabel(tpi, slope) {
  if (tpi == null) return "?";
  const band = LABELS.landformBands.find(([threshold]) => tpi >= threshold);
  const landform = band ? band[1] : LABELS.landformFallback;
  const steep = slope == null ? null : LABELS.slopeBands.find(([threshold]) => slope >= threshold);
  return steep ? `${landform}, ${steep[1]}` : landform;
}

// A low light/shade ratio means one of two opposite things -- too few trees or
// the wrong kind of canopy -- and the label has to say which, or a nearly
// treeless stand reads as "dense, little light" on the kantarelli map and as
// "shady spruce" on the suppilovahvero one.
function lightLabel(cfg, ratio, stemcount) {
  const light = cfg.light;
  if (ratio >= cfg.green.light) return light.good;
  if (stemcount != null && stemcount < light.sparseStems) return light.sparse;
  return ratio >= MID ? light.mid : light.poor;
}

// ratio is this field's contribution to the score, 0-1 relative to its own
// max -- null means "not a scored factor", so no badge is shown. Thresholds
// come from the active species' own green/mid cutoffs, injected from the same
// Python constants the "Excellent" rule uses, so a category and its dots can
// never disagree.
function scoreBadge(ratio, good) {
  // an invisible dot rather than none at all, so an unscored row still lines
  // its text up with the scored ones above and below it
  if (ratio == null) return '<span class="score-badge blank"></span>';
  const tier = ratio >= good ? "good" : ratio >= MID ? "mid" : "poor";
  return `<span class="score-badge ${tier}" title="Effect on score: ${tier}"></span>`;
}

function popupRow(label, value, ratio, good) {
  return `<div class="popup-field">${scoreBadge(ratio, good)}<div>` +
    `<span class="popup-label">${label}</span>` +
    `<span class="popup-value">${value}</span></div></div>`;
}

function popupHtml(p, cfg) {
  const v = p[cfg.key];
  const category = CATEGORIES[v[CATEGORY]];
  const soil = `${LABELS.soil[p.st] || "?"} (${LABELS.drainage[p.ds] || "?"})`;
  const rows = [
    popupRow("Site type", LABELS.fertility[p.fc] || "?", v[RATIO.fertility], cfg.green.fertility),
    popupRow("Development class", LABELS.development[p.dc] || "?", v[RATIO.development], cfg.green.development),
    popupRow("Dominant tree", LABELS.species[p.ts] || "Other", v[RATIO.species], cfg.green.species),
    popupRow("Tree mix", mixtureLabel(p.div), p.div, cfg.green.mixture),
    popupRow(cfg.light.row, lightLabel(cfg, v[RATIO.light], p.stem), v[RATIO.light], cfg.green.light),
    popupRow("Soil", soil, v[RATIO.soil], cfg.green.soil),
    popupRow(cfg.terrain.row, landformLabel(p.tpi, p.slp), v[RATIO.terrain], cfg.green.terrain),
  ];
  // binary factor, and only for the species whose model uses it: green when
  // near an esker, red when not
  if (cfg.esker) {
    rows.push(popupRow("Location", p.esk ? cfg.esker[0] : cfg.esker[1], p.esk ? 1 : 0, 1));
  }

  const c = pal(category);
  return `<div class="popup-header" style="--cat:${c.fill};--cat-ink:${c.ink};--cat-tint:${c.tint}">` +
    `<span class="popup-cat">${CATEGORY_LABELS[category] || category}</span>` +
    `<span class="popup-score">${Math.round(v[SCORE])}<em>/100</em></span>` +
    `</div><div class="popup-body">` + rows.join("") +
    `<a class="gmaps-btn" target="_blank" rel="noopener" ` +
    `href="https://www.google.com/maps/dir/?api=1&destination=${p.lat},${p.lon}&travelmode=driving">` +
    `Navigate here &middot; Google Maps</a></div>`;
}

// One Leaflet layer per species, built on first use and kept afterwards, so
// flipping back and forth costs nothing. A stand the active species has no
// score for is simply not in its layer.
const layers = new Map();
function speciesLayer(cfg) {
  if (!layers.has(cfg.slug)) {
    const stands = L.geoJSON(STANDS, {
      filter: (feature) => feature.properties[cfg.key] != null,
      style: (feature) => {
        const category = CATEGORIES[feature.properties[cfg.key][CATEGORY]];
        const c = pal(category);
        return { color: c.line, weight: 1, opacity: 0.6,
                 fillColor: c.fill, fillOpacity: FILL_OPACITY[category] ?? 0.45 };
      },
    // One popup for the whole layer rather than one per stand: Leaflet hands
    // the clicked stand to the content function, so the popup is filled in as
    // it opens and tens of thousands of stands carry no popup of their own.
    // The switcher and the legend are fixed overlays Leaflet knows nothing
    // about, so autopan has to be told to keep the popup clear of both --
    // without this a popup near the top edge opens with its score hidden
    // behind the species buttons.
    }).bindPopup((stand) => popupHtml(stand.feature.properties, cfg), {
      minWidth: 258,
      autoPanPaddingTopLeft: L.point(12, 72),
      autoPanPaddingBottomRight: L.point(12, 150),
    });
    // Real laji.fi sighting flags for this species, where a report happens to
    // fall inside the mapped area
    const sightings = L.geoJSON(cfg.sightings, {
      pointToLayer: (feature, latlng) => L.marker(latlng, {
        icon: L.divIcon({ className: "", html: '<div class="sighting-flag">🚩</div>', iconSize: [20, 20], iconAnchor: [4, 18] }),
      }),
      onEachFeature: (feature, layer) => {
        const d = feature.properties.date || "date unknown";
        layer.bindPopup(
          `<div class="popup-note"><b>${cfg.name} sighting</b>` +
          `<span>Reported to laji.fi<br>${d}</span></div>`
        );
      },
    });
    layers.set(cfg.slug, { stands, sightings });
  }
  return layers.get(cfg.slug);
}

// --- legend ------------------------------------------------------------
const legend = document.createElement("div");
legend.className = "legend";
legend.innerHTML =
  '<button class="legend-close" aria-label="Hide legend">×</button>' +
  '<span class="legend-title"></span>' +
  '<div class="legend-row">' +
  `<span><span class="swatch" style="background:${PALETTE.excellent.fill}"></span>${CATEGORY_LABELS.excellent}</span>` +
  `<span><span class="swatch" style="background:${PALETTE.high.fill}"></span>${CATEGORY_LABELS.high}</span>` +
  `<span><span class="swatch" style="background:${PALETTE.medium.fill}"></span>${CATEGORY_LABELS.medium}</span>` +
  '<span>🚩 Reported find (laji.fi)</span>' +
  "</div><small></small>";
document.body.appendChild(legend);

// closing hides it for the rest of this page view; no reopen button, since a
// floating "reopen" button ended up sitting awkwardly over the map itself
legend.querySelector(".legend-close").addEventListener("click", () => {
  legend.hidden = true;
});

// --- species switcher --------------------------------------------------
const switcher = document.createElement("div");
switcher.className = "species-switch";
switcher.setAttribute("role", "tablist");
switcher.setAttribute("aria-label", "Choose a mushroom");
document.body.appendChild(switcher);

let active = null;

function selectSpecies(cfg, { fit = false } = {}) {
  if (active === cfg) return;
  if (active) {
    const previous = speciesLayer(active);
    map.removeLayer(previous.stands);
    map.removeLayer(previous.sightings);
  }
  map.closePopup();
  active = cfg;

  const layer = speciesLayer(cfg);
  layer.stands.addTo(map);
  layer.sightings.addTo(map);
  if (fit) map.fitBounds(layer.stands.getBounds());

  legend.querySelector(".legend-title").textContent = `${cfg.name} probability`;
  legend.querySelector("small").textContent =
    `${cfg.intro} An estimate from each forest stand's ecological attributes ` +
    "(site type, trees, soil) - not a record of where mushrooms have actually grown.";
  for (const button of switcher.children) {
    button.setAttribute("aria-selected", String(button.dataset.slug === cfg.slug));
  }
  try { localStorage.setItem(STORAGE_KEY, cfg.slug); } catch (e) { /* private mode */ }
}

for (const cfg of SPECIES) {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = cfg.name;
  button.dataset.slug = cfg.slug;
  button.setAttribute("role", "tab");
  button.setAttribute("aria-selected", "false");
  button.title = cfg.latin;
  button.addEventListener("click", () => selectSpecies(cfg));
  switcher.appendChild(button);
}

// remember the last choice: the same person walks back into the same forest
// looking for the same mushroom
let remembered = null;
try { remembered = localStorage.getItem(STORAGE_KEY); } catch (e) { /* private mode */ }
selectSpecies(SPECIES.find((s) => s.slug === remembered) || SPECIES[0], { fit: true });

// Live location: read the browser's geolocation and refresh a "you are here"
// dot every 30s. file:// and localhost both count as secure contexts, so
// this works when the map is opened straight from disk.
const LOCATION_REFRESH_MS = 30000;
let locationMarker = null;
let accuracyCircle = null;
let firstFix = true;

function updateLocation() {
  if (!("geolocation" in navigator)) {
    console.warn("Geolocation not supported by this browser.");
    return;
  }
  navigator.geolocation.getCurrentPosition(
    (pos) => {
      const latlng = [pos.coords.latitude, pos.coords.longitude];
      if (!locationMarker) {
        locationMarker = L.marker(latlng, {
          icon: L.divIcon({ className: "", html: '<div class="my-location-dot"></div>', iconSize: [16, 16] }),
          zIndexOffset: 1000,
        }).addTo(map).bindPopup("You are here");
        accuracyCircle = L.circle(latlng, { radius: pos.coords.accuracy, color: "#1a73e8", weight: 1, fillOpacity: 0.1 }).addTo(map);
      } else {
        locationMarker.setLatLng(latlng);
        accuracyCircle.setLatLng(latlng).setRadius(pos.coords.accuracy);
      }
      // Only follow the fix once it is actually inside the mapped area. A
      // fix from home, from another town, or a bad first read would otherwise
      // drag the view onto empty basemap with no stands on it at all -- there
      // is no data outside the mapped area, so there is nothing to look at
      // there. The whole area stays framed until then, and the first fix that
      // does land inside the map still centres on it, so driving in works as
      // before.
      if (firstFix && DATA_BOUNDS.contains(latlng)) {
        map.setView(latlng, 15);
        firstFix = false;
      }
    },
    (err) => console.warn("Location lookup failed:", err.message),
    { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
  );
}

updateLocation();
setInterval(updateLocation, LOCATION_REFRESH_MS);
</script>
</body>
</html>
"""


def render_html(geojson_dict: dict) -> str:
    payload = json.dumps(to_page_payload(geojson_dict), ensure_ascii=False, separators=(",", ":"))
    html = HTML_TEMPLATE.replace("__STANDS__", payload)
    html = html.replace("__SPECIES__", json.dumps(species_config(), ensure_ascii=False))
    html = html.replace("__LABELS__", json.dumps(label_config(), ensure_ascii=False))
    html = html.replace("__CATEGORIES__", json.dumps(MAPPED_CATEGORIES))
    return html.replace("__MID_THRESHOLD__", json.dumps(sp.MID_THRESHOLD))


def main() -> None:
    layers = load_layers()
    near_esker = compute_near_esker(layers[0])

    frames = {}
    for slug, profile in PROFILES.items():
        frames[slug] = build_species_frame(layers, near_esker, profile)
        counts = pd.crosstab(frames[slug]["category"], frames[slug]["municipality"], margins=True)
        print(f"\n=== {profile.name} ===")
        print(counts.to_string())

    OUTPUT_GEOJSON.parent.mkdir(parents=True, exist_ok=True)
    geojson_dict = to_geojson_dict(frames)
    OUTPUT_GEOJSON.write_text(json.dumps(geojson_dict, ensure_ascii=False), encoding="utf-8")
    OUTPUT_HTML.write_text(render_html(geojson_dict), encoding="utf-8")

    print(f"\nWrote {OUTPUT_GEOJSON} ({len(geojson_dict['features'])} stands, both species)")
    print(f"Wrote {OUTPUT_HTML} ({OUTPUT_HTML.stat().st_size / 1e6:.1f} MB)")
    for cfg in species_config():
        print(f"  {cfg['name']}: {len(cfg['sightings']['features'])} laji.fi sightings embedded as flags")


if __name__ == "__main__":
    main()
