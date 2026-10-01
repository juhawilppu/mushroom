"""Score the forest stands of the mapped area (see area.py) for mushroom
habitat suitability and render the result as a MapLibre map with a species
switcher: a small page, plus the stands as vector tiles.

The habitat heuristics themselves live in `scripts/species.py`, one profile
per mushroom; this module is the engine that applies a profile to the forest
inventory and draws the result. Kantarelli wants dry, light-flooded,
well-drained mineral soil near eskers; suppilovahvero wants damp, shady,
moss-floored spruce forest and is at home on peat -- so the two maps
disagree about most of the municipality, which is the point of having both.

Both species share one set of tiles: stand geometry is by far the largest
part of the data and is identical between them, so it is written once and
each species contributes only its own scores. Attributes are shipped as
inventory codes and turned into labels in the browser, in English or, through
translations.py, Finnish. As tiles, a phone downloads and draws only the part
of the map in view, however many municipalities the map grows to -- this
thing gets loaded over mobile data, in a forest.

Run scripts/download_data.py first. Needs tippecanoe (brew install
tippecanoe) to cut the tiles.
"""

import json
import shutil
import subprocess

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import mapping

import area
import species as sp
import topography as topo
from species import PROFILES, SpeciesProfile
from translations import FINNISH

OUTPUT_GEOJSON = area.ROOT / "output" / "scored_stands.geojson"
# The deployable site: the page, the vector tiles it reads, and the host's
# header rules. scripts/deploy.sh publishes it as it stands.
SITE_DIR = area.ROOT / "output" / "site"
OUTPUT_HTML = SITE_DIR / "index.html"
TILES_DIR = SITE_DIR / "tiles"
# Pages serves .pbf as application/octet-stream, which Cloudflare passes on
# uncompressed; declared as protobuf, the tiles go out brotli-compressed.
PAGES_HEADERS = "/tiles/*\n  Content-Type: application/x-protobuf\n"
TILES_INPUT = area.CACHE_DIR / "stands_for_tiles.geojsonl"
# Zoomed out, a stand is a few pixels across -- too small to tap -- so the
# tiles up to there carry only what the colours need ("overview"), and the
# full attributes the popup reads ("stands") start at DETAIL_MIN_ZOOM. That
# takes the z11-z12 tiles, the ones browsed most, to about a third the size.
OVERVIEW_LAYER, DETAIL_LAYER = "overview", "stands"
DETAIL_MIN_ZOOM = 13
# Below z7 the whole area is a few pixels across. Above z13 the browser scales
# z13 tiles up itself: a z13 tile is about 2.4 km wide here, held at 4096 units
# across, so stand outlines already simplified to 2 m lose nothing -- and
# stopping at 13 keeps the tile count well inside what Pages accepts.
TILE_MIN_ZOOM, TILE_MAX_ZOOM = 7, 13

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


def tile_properties(props: dict) -> dict:
    """A stand's properties as a vector tile can hold them: scalars only.

    Each species' [score, category, ...ratios] block becomes named fields --
    k_score, k_cat, k_fertility and so on -- and missing values are left out
    rather than written as null. The stand id stays behind in the GeoJSON
    export: nothing on the page reads it.
    """
    out = {name: value for name, value in props.items()
           if name != "id" and value is not None and not isinstance(value, list)}
    for profile in PROFILES.values():
        block = props.get(profile.map_key)
        if block is None:
            continue
        score, category, *ratios = block
        key = profile.map_key
        out[f"{key}_score"] = score
        out[f"{key}_cat"] = category
        out.update({f"{key}_{factor}": ratio for factor, ratio in zip(SCORED_FACTORS, ratios)})
    return out


def write_tiles(geojson_dict: dict) -> None:
    """Cut the mapped stands into vector tiles, one file per tile.

    Shipped inside the page, the stands of nine municipalities came to 16 MB
    and tens of thousands of polygons a phone had to re-project on every zoom.
    As tiles, a phone downloads and draws only the part of the map in view.
    They are written as a plain {z}/{x}/{y}.pbf directory rather than one
    PMTiles archive because Cloudflare Pages ignores HTTP range requests,
    which a PMTiles archive depends on. Tiles are left uncompressed for the
    same reason: the host compresses them on the way out.
    """
    TILES_INPUT.parent.mkdir(parents=True, exist_ok=True)
    with TILES_INPUT.open("w", encoding="utf-8") as out:
        for feature in geojson_dict["features"]:
            props = tile_properties(feature["properties"])
            # every stand goes in twice: colours only while zoomed out, the
            # lot once it is big enough to tap
            for layer, zooms, properties in (
                (OVERVIEW_LAYER, {"maxzoom": DETAIL_MIN_ZOOM - 1},
                 {k: v for k, v in props.items() if k.endswith("_cat")}),
                (DETAIL_LAYER, {"minzoom": DETAIL_MIN_ZOOM}, props),
            ):
                out.write(json.dumps({
                    "type": "Feature",
                    "tippecanoe": {"layer": layer, **zooms},
                    "geometry": feature["geometry"],
                    "properties": properties,
                }, ensure_ascii=False, separators=(",", ":")) + "\n")

    # tippecanoe adds to an existing directory, so a stand dropped since the
    # last build would otherwise live on in a stale tile
    shutil.rmtree(TILES_DIR, ignore_errors=True)
    subprocess.run([
        "tippecanoe", "--quiet", "--force",
        "--output-to-directory", str(TILES_DIR),
        "--minimum-zoom", str(TILE_MIN_ZOOM),
        "--maximum-zoom", str(TILE_MAX_ZOOM),
        "--no-tile-compression",
        # neighbouring stands share their borders, and simplifying each one
        # on its own would open slivers between them at low zoom
        "--detect-shared-borders",
        # tippecanoe's default keeps outlines within 1/8 of a screen pixel;
        # a whole pixel is still invisible and makes the zoomed-out tiles far
        # lighter. z13 stays exact, as the browser scales it up for the
        # closest zooms.
        "--simplification=8", "--simplify-only-low-zooms",
        "--drop-densest-as-needed",
        "--read-parallel",
        str(TILES_INPUT),
    ], check=True)


def data_bounds(geojson_dict: dict) -> list[float]:
    """[west, south, east, north] of the mapped stands' centroids."""
    lons = [f["properties"]["lon"] for f in geojson_dict["features"]]
    lats = [f["properties"]["lat"] for f in geojson_dict["features"]]
    return [min(lons), min(lats), max(lons), max(lats)]


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
            # whole phrases rather than put together in the browser: Finnish
            # inflects the name ("Kantarellin todennäköisyys")
            "legend": f"{profile.name} probability",
            "sighting": f"{profile.name} sighting",
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


def strings_in(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from strings_in(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from strings_in(value)


def untranslated() -> list[str]:
    """species.py wording the Finnish page would show in English, for want of
    an entry in translations.py. The page's own strings are translated right
    where the template uses them; these are the ones written somewhere else."""
    shown = set(strings_in(label_config()))
    for cfg in species_config():
        # identifiers, the Latin name and the sighting data aren't translated
        shown.update(strings_in({k: v for k, v in cfg.items()
                                 if k not in ("slug", "key", "latin", "sightings")}))
    return sorted(shown - FINNISH.keys())


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Mushroom map</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/maplibre-gl/5.24.0/maplibre-gl.css">
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
  .legend small { display: flow-root; margin-top: 11px; padding-top: 10px; border-top: 1px solid var(--rule);
                  font-size: 12px; line-height: 1.5; color: var(--ink-faint); }
  /* The language switch closes the note, floated right: it shares the note's
     last line when there is room, and takes a line of its own when there
     isn't, so it costs the legend little or no height on a phone. */
  .lang-switch { float: right; display: flex; gap: 2px; margin: -4px -4px -4px 10px; }
  .lang-switch button { appearance: none; border: none; padding: 4px 8px; border-radius: 999px;
                        background: none; font: inherit; font-size: 11px; font-weight: 700;
                        letter-spacing: .06em; color: var(--ink-faint); cursor: pointer;
                        transition: background .15s, color .15s; }
  .lang-switch button:hover { color: var(--ink); }
  .lang-switch button[aria-pressed="true"] { background: var(--forest); color: #fff; }
  .legend-close { position: absolute; top: 10px; right: 10px; width: 30px; height: 30px;
                  border: none; border-radius: 50%; background: #f1f3ef; color: var(--ink-faint);
                  font-size: 17px; line-height: 30px; text-align: center; cursor: pointer;
                  transition: background .15s, color .15s; }
  .legend-close:hover { background: #e4e8e2; color: var(--ink); }

  .my-location-dot { width: 16px; height: 16px; border-radius: 50%; background: #1a73e8;
                     border: 2px solid white; box-shadow: 0 0 0 2px rgba(26,115,232,.45), 0 1px 4px rgba(0,0,0,.3); }

  /* --- popup ----------------------------------------------------------- */
  .maplibregl-popup-content { padding: 0; border-radius: 14px; overflow: hidden;
                              box-shadow: 0 12px 36px rgba(23,35,28,.22);
                              font-size: 15px; line-height: 1.5; }
  .stand-popup .maplibregl-popup-content { min-width: 258px; }
  .maplibregl-popup-close-button { top: 10px; right: 8px; width: 26px; height: 26px;
                                   border-radius: 50%; font-size: 20px; line-height: 24px;
                                   color: var(--ink-faint); background: none; }
  .maplibregl-popup-close-button:hover { color: var(--ink); background: none; }
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
  /* .maplibregl-popup-content carries no padding of its own (the stand popup
     supplies its own), so the other popups bring a padded wrapper */
  .popup-note { padding: 14px 40px 14px 16px; }
  .popup-note b { display: block; margin-bottom: 3px; font-size: 12px; font-weight: 700;
                  letter-spacing: .07em; text-transform: uppercase; color: var(--forest); }
  .popup-note span { color: var(--ink-soft); font-size: 14px; }

  .maplibregl-map { font-family: var(--font); }
  .maplibregl-ctrl-attrib { font-size: 11px; }
</style>
</head>
<body>
<div id="map"></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/maplibre-gl/5.24.0/maplibre-gl.js"></script>
<script>
const SPECIES = __SPECIES__;   // one entry per mushroom, in switcher order
const LABELS = __LABELS__;     // inventory code -> English, expanded here rather
                               // than repeated on every stand in the tiles
const FINNISH = __FINNISH__;   // English -> Finnish, from translations.py
const MID = __MID_THRESHOLD__;
const CATEGORIES = __CATEGORIES__;
const TILES = __TILES__;       // the vector tiles written next to this page
const BOUNDS = __BOUNDS__;     // [west, south, east, north] of the mapped stands
const STORAGE_KEY = "karkkila-sienikartta-species";
const LANGUAGE_KEY = "karkkila-sienikartta-language";

// A language picked with the switch sticks. Until then, Finnish goes to
// anyone whose browser asks for it first, and to anyone on Finnish time:
// plenty of Finns run their phones in English.
function initialLanguage() {
  try {
    const picked = localStorage.getItem(LANGUAGE_KEY);
    if (picked === "fi" || picked === "en") return picked;
  } catch (e) { /* private mode */ }
  const preferred = (navigator.languages?.[0] || navigator.language || "").toLowerCase();
  const zone = Intl.DateTimeFormat().resolvedOptions().timeZone;
  return preferred.split("-")[0] === "fi" || zone === "Europe/Helsinki" ? "fi" : "en";
}

// The page is written in English and every string goes through t() on its
// way to the screen; one without a Finnish entry stays English.
let lang = initialLanguage();
const t = (text) => (lang === "fi" && FINNISH[text]) || text;

// laji.fi dates are ISO (2024-08-15); written out the way each language does
function formatDate(iso) {
  const day = new Date(`${iso}T12:00`);
  if (isNaN(day)) return iso;
  return day.toLocaleDateString(lang === "fi" ? "fi" : "en-GB", { day: "numeric", month: "long", year: "numeric" });
}

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

// A species' scores ride on each stand as flat fields -- k_score, k_cat,
// k_fertility and so on -- because a vector tile holds only plain values.
function scoresOf(p, cfg) {
  return {
    score: p[`${cfg.key}_score`],
    category: CATEGORIES[p[`${cfg.key}_cat`]],
    ratio: (factor) => p[`${cfg.key}_${factor}`],
  };
}

function mixtureLabel(diversity) {
  for (const [threshold, label] of LABELS.mixtureBands) {
    if (diversity >= threshold) return t(label);
  }
  return t(LABELS.mixtureFallback);
}

// Where the stand sits in the landscape, from the elevation model: how high
// it stands relative to the ground within 500 m, and how steeply it falls.
// The slope half is only spelled out when there is a slope worth mentioning.
function landformLabel(tpi, slope) {
  if (tpi == null) return "?";
  const band = LABELS.landformBands.find(([threshold]) => tpi >= threshold);
  const landform = t(band ? band[1] : LABELS.landformFallback);
  const steep = slope == null ? null : LABELS.slopeBands.find(([threshold]) => slope >= threshold);
  return steep ? `${landform}, ${t(steep[1])}` : landform;
}

// A low light/shade ratio means one of two opposite things -- too few trees or
// the wrong kind of canopy -- and the label has to say which, or a nearly
// treeless stand reads as "dense, little light" on the kantarelli map and as
// "shady spruce" on the suppilovahvero one.
function lightLabel(cfg, ratio, stemcount) {
  const light = cfg.light;
  if (ratio >= cfg.green.light) return t(light.good);
  if (stemcount != null && stemcount < light.sparseStems) return t(light.sparse);
  return t(ratio >= MID ? light.mid : light.poor);
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
  const title = { good: t("Effect on score: good"), mid: t("Effect on score: mid"),
                  poor: t("Effect on score: poor") }[tier];
  return `<span class="score-badge ${tier}" title="${title}"></span>`;
}

function popupRow(label, value, ratio, good) {
  return `<div class="popup-field">${scoreBadge(ratio, good)}<div>` +
    `<span class="popup-label">${label}</span>` +
    `<span class="popup-value">${value}</span></div></div>`;
}

function popupHtml(p, cfg) {
  const s = scoresOf(p, cfg);
  const soil = `${t(LABELS.soil[p.st] || "?")} (${t(LABELS.drainage[p.ds] || "?")})`;
  const rows = [
    popupRow(t("Site type"), t(LABELS.fertility[p.fc] || "?"), s.ratio("fertility"), cfg.green.fertility),
    popupRow(t("Development class"), t(LABELS.development[p.dc] || "?"), s.ratio("development"), cfg.green.development),
    popupRow(t("Dominant tree"), t(LABELS.species[p.ts] || "Other"), s.ratio("species"), cfg.green.species),
    popupRow(t("Tree mix"), mixtureLabel(p.div), p.div, cfg.green.mixture),
    popupRow(t(cfg.light.row), lightLabel(cfg, s.ratio("light"), p.stem), s.ratio("light"), cfg.green.light),
    popupRow(t("Soil"), soil, s.ratio("soil"), cfg.green.soil),
    popupRow(t(cfg.terrain.row), landformLabel(p.tpi, p.slp), s.ratio("terrain"), cfg.green.terrain),
  ];
  // binary factor, and only for the species whose model uses it: green when
  // near an esker, red when not
  if (cfg.esker) {
    rows.push(popupRow(t("Location"), t(p.esk ? cfg.esker[0] : cfg.esker[1]), p.esk ? 1 : 0, 1));
  }

  const c = pal(s.category);
  return `<div class="popup-header" style="--cat:${c.fill};--cat-ink:${c.ink};--cat-tint:${c.tint}">` +
    `<span class="popup-cat">${t(CATEGORY_LABELS[s.category] || s.category)}</span>` +
    `<span class="popup-score">${Math.round(s.score)}<em>/100</em></span>` +
    `</div><div class="popup-body">` + rows.join("") +
    `<a class="gmaps-btn" target="_blank" rel="noopener" ` +
    `href="https://www.google.com/maps/dir/?api=1&destination=${p.lat},${p.lon}&travelmode=driving">` +
    `${t("Navigate here · Google Maps")}</a></div>`;
}

function sightingHtml(p) {
  return `<div class="popup-note"><b>${t(active.sighting)}</b>` +
    `<span>${t("Reported to laji.fi")}<br>${p.date ? formatDate(p.date) : t("date unknown")}</span></div>`;
}

// Everything mapped is inside these few municipalities, so this is the area
// the map is ever useful in, padded a little so a fix right at its edge counts.
const PAD = 0.02;
const [WEST, SOUTH, EAST, NORTH] = BOUNDS;
const DATA_BOUNDS = new maplibregl.LngLatBounds(
  [WEST - PAD * (EAST - WEST), SOUTH - PAD * (NORTH - SOUTH)],
  [EAST + PAD * (EAST - WEST), NORTH + PAD * (NORTH - SOUTH)],
);

const map = new maplibregl.Map({
  container: "map",
  // framed clear of the switcher above and the legend below
  bounds: BOUNDS,
  fitBoundsOptions: { padding: { top: 70, bottom: 150, left: 20, right: 20 } },
  maxZoom: 19,
  dragRotate: false,
  pitchWithRotate: false,
  touchPitch: false,
  attributionControl: { compact: true },
  // MapLibre's own screen-reader labels. It reads these once, so after a
  // switch of language they follow on the next visit rather than at once.
  locale: {
    "Map.Title": t("Map"),
    "Marker.Title": t("Map marker"),
    "Popup.Close": t("Close popup"),
    "AttributionControl.ToggleAttribution": t("Toggle attribution"),
  },
  style: {
    version: 8,
    sources: {
      osm: {
        type: "raster",
        tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
        tileSize: 256,
        maxzoom: 19,
        attribution: "&copy; OpenStreetMap contributors",
      },
      stands: {
        type: "vector",
        // tiles are fetched from a web worker, which has no page to resolve
        // a relative URL against, so the tile directory is made absolute here
        tiles: [new URL("tiles/", location.href).href + "{z}/{x}/{y}.pbf"],
        minzoom: TILES.minzoom,
        maxzoom: TILES.maxzoom,
      },
    },
    layers: [{
      id: "osm", type: "raster", source: "osm",
      // OSM's own greens and browns compete with the score colours for
      // attention. Muting the tiles keeps every road, path and label legible
      // while letting the stands read as the data layer they are.
      paint: { "raster-saturation": -0.55, "raster-contrast": -0.06, "raster-brightness-min": 0.06 },
    }],
  },
});
map.touchZoomRotate.disableRotation();
map.keyboard.disableRotation();

// Zoomed out, the tiles carry each stand's colour only ("overview"); from
// TILES.detailMinZoom on they carry everything the popup shows ("stands").
const STAND_LAYERS = [TILES.overview, TILES.detail];

// A style expression picking a value by the active species' category
function byCategory(cfg, pick, fallback) {
  return ["match", ["get", `${cfg.key}_cat`],
          ...CATEGORIES.flatMap((category, i) => [i, pick(category)]), fallback];
}

// Recolour the stands for a species; a stand that species has nothing to say
// about has no category for it, and is left off the map.
function paintSpecies(cfg) {
  const scored = ["has", `${cfg.key}_cat`];
  for (const layer of STAND_LAYERS) {
    map.setFilter(`${layer}-fill`, scored);
    map.setFilter(`${layer}-line`, scored);
    map.setPaintProperty(`${layer}-fill`, "fill-color", byCategory(cfg, (c) => pal(c).fill, FALLBACK.fill));
    map.setPaintProperty(`${layer}-fill`, "fill-opacity", byCategory(cfg, (c) => FILL_OPACITY[c], 0.45));
    map.setPaintProperty(`${layer}-line`, "line-color", byCategory(cfg, (c) => pal(c).line, FALLBACK.line));
  }
  map.setFilter("sightings", ["==", ["get", "species"], cfg.slug]);
}

// Real laji.fi sighting flags, where a report happens to fall inside the
// mapped area: every species in one source, filtered to the active one
const SIGHTINGS = {
  type: "FeatureCollection",
  features: SPECIES.flatMap((cfg) => cfg.sightings.features.map((feature) => ({
    ...feature, properties: { ...feature.properties, species: cfg.slug },
  }))),
};

// The flag is the legend's own emoji drawn onto a canvas, at twice its size so
// it stays sharp on high-density screens
function addFlagImage() {
  const scale = 2, size = 24 * scale;
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext("2d");
  ctx.font = `${20 * scale}px sans-serif`;
  ctx.textBaseline = "bottom";
  ctx.shadowColor = "rgba(0,0,0,.5)";
  ctx.shadowBlur = 2 * scale;
  ctx.shadowOffsetY = scale;
  ctx.fillText("🚩", 0, size - 2 * scale);
  map.addImage("flag", ctx.getImageData(0, 0, size, size), { pixelRatio: scale });
}

let popup = null;
let popupContent = null;  // redraws the open popup, for a change of language

// MapLibre fits a popup inside the map's own edges, but the switcher and the
// legend sit on top of the map, so the view is nudged to keep the popup clear
// of both -- without this a popup near the top edge opens with its score
// hidden behind the species buttons.
function keepClear(element) {
  const box = element.getBoundingClientRect();
  const top = switcher.getBoundingClientRect().bottom + 12;
  const bottom = (legend.hidden ? window.innerHeight : legend.getBoundingClientRect().top) - 12;
  const right = window.innerWidth - 12;
  const dy = box.top < top ? box.top - top : Math.max(0, Math.min(box.bottom - bottom, box.top - top));
  const dx = box.left < 12 ? box.left - 12 : Math.max(0, Math.min(box.right - right, box.left - 12));
  if (dx || dy) map.panBy([dx, dy]);
}

function openPopup(lngLat, content, className) {
  popup?.remove();
  popupContent = content;
  popup = new maplibregl.Popup({ className, maxWidth: "300px", focusAfterOpen: false })
    .setLngLat(lngLat).setHTML(content()).addTo(map);
  keepClear(popup.getElement());
}

const CLICKABLE = ["sightings", `${TILES.detail}-fill`, `${TILES.overview}-fill`];

map.on("click", (e) => {
  // topmost first, so a flag wins over the stand it stands in
  const [hit] = map.queryRenderedFeatures(e.point, { layers: CLICKABLE });
  if (!hit) return;
  if (hit.layer.id === "sightings") {
    openPopup(hit.geometry.coordinates, () => sightingHtml(hit.properties));
  } else if (hit.layer.id === `${TILES.detail}-fill`) {
    openPopup(e.lngLat, () => popupHtml(hit.properties, active), "stand-popup");
  } else {
    // zoomed out, a stand is a few pixels across and its tile carries only
    // its colour: take the reader in to where a stand can be tapped
    map.easeTo({ center: e.lngLat, zoom: TILES.detailMinZoom + 0.5 });
  }
});
for (const layer of CLICKABLE) {
  map.on("mouseenter", layer, () => { map.getCanvas().style.cursor = "pointer"; });
  map.on("mouseleave", layer, () => { map.getCanvas().style.cursor = ""; });
}

// --- legend ------------------------------------------------------------
// Text set once and left in place carries its English in data-text (or
// data-label, for an aria-label), which translatePage() turns into the
// current language.
const legend = document.createElement("div");
legend.className = "legend";
legend.innerHTML =
  '<button class="legend-close" data-label="Hide legend">×</button>' +
  '<span class="legend-title"></span>' +
  '<div class="legend-row">' +
  `<span><span class="swatch" style="background:${PALETTE.excellent.fill}"></span><span data-text="${CATEGORY_LABELS.excellent}"></span></span>` +
  `<span><span class="swatch" style="background:${PALETTE.high.fill}"></span><span data-text="${CATEGORY_LABELS.high}"></span></span>` +
  `<span><span class="swatch" style="background:${PALETTE.medium.fill}"></span><span data-text="${CATEGORY_LABELS.medium}"></span></span>` +
  '<span>🚩<span data-text="Reported find (laji.fi)"></span></span>' +
  '</div><small><span class="legend-intro"></span> ' +
  `<span data-text="An estimate from each forest stand's ecological attributes (site type, trees, soil) ` +
  `- not a record of where mushrooms have actually grown."></span>` +
  // each language named in itself, the way language switches usually are
  '<span class="lang-switch" role="group" data-label="Language">' +
  '<button type="button" lang="fi" title="Suomi" aria-label="Suomi">FI</button>' +
  '<button type="button" lang="en" title="English" aria-label="English">EN</button>' +
  "</span></small>";
document.body.appendChild(legend);

// closing hides it for the rest of this page view; no reopen button, since a
// floating "reopen" button ended up sitting awkwardly over the map itself
legend.querySelector(".legend-close").addEventListener("click", () => {
  legend.hidden = true;
});

// --- language ----------------------------------------------------------
const languageSwitch = legend.querySelector(".lang-switch");

function translatePage() {
  document.documentElement.lang = lang;
  document.title = t("Mushroom map");
  for (const el of document.querySelectorAll("[data-text]")) el.textContent = t(el.dataset.text);
  for (const el of document.querySelectorAll("[data-label]")) el.setAttribute("aria-label", t(el.dataset.label));
  for (const button of languageSwitch.children) {
    button.setAttribute("aria-pressed", String(button.lang === lang));
  }
}

function setLanguage(code) {
  lang = code;
  translatePage();
  // popups are drawn from scratch rather than tagged, so the open one is
  // opened afresh: swapping its HTML in place would drop the label MapLibre
  // gives its close button only when a popup opens
  if (popup?.isOpen()) openPopup(popup.getLngLat(), popupContent, popup.options.className);
  locationMarker?.getPopup().setHTML(locationNote());
  try { localStorage.setItem(LANGUAGE_KEY, code); } catch (e) { /* private mode */ }
}

languageSwitch.addEventListener("click", (e) => {
  const button = e.target.closest("button");
  if (button && button.lang !== lang) setLanguage(button.lang);
});

// --- species switcher --------------------------------------------------
const switcher = document.createElement("div");
switcher.className = "species-switch";
switcher.setAttribute("role", "tablist");
switcher.dataset.label = "Choose a mushroom";
document.body.appendChild(switcher);

let active = null;
let mapReady = false;

function selectSpecies(cfg) {
  if (active === cfg) return;
  popup?.remove();
  active = cfg;
  if (mapReady) paintSpecies(cfg);

  legend.querySelector(".legend-title").dataset.text = cfg.legend;
  legend.querySelector(".legend-intro").dataset.text = cfg.intro;
  for (const button of switcher.children) {
    button.setAttribute("aria-selected", String(button.dataset.slug === cfg.slug));
  }
  translatePage();
  try { localStorage.setItem(STORAGE_KEY, cfg.slug); } catch (e) { /* private mode */ }
}

for (const cfg of SPECIES) {
  const button = document.createElement("button");
  button.type = "button";
  button.dataset.text = cfg.name;
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
selectSpecies(SPECIES.find((s) => s.slug === remembered) || SPECIES[0]);

// Live location: read the browser's geolocation and refresh a "you are here"
// dot every 30s. localhost counts as a secure context, so this also works
// when the map is served locally.
const LOCATION_REFRESH_MS = 30000;
let locationMarker = null;
let firstFix = true;
const locationNote = () => `<div class="popup-note"><span>${t("You are here")}</span></div>`;

// The accuracy radius as a polygon: MapLibre's own circles are sized in screen
// pixels, not metres on the ground
function accuracyCircle([lon, lat], radius, steps = 64) {
  const ring = [];
  for (let i = 0; i <= steps; i++) {
    const angle = (i / steps) * 2 * Math.PI;
    ring.push([
      lon + (radius * Math.cos(angle)) / (111320 * Math.cos((lat * Math.PI) / 180)),
      lat + (radius * Math.sin(angle)) / 110540,
    ]);
  }
  return { type: "Feature", properties: {}, geometry: { type: "Polygon", coordinates: [ring] } };
}

function updateLocation() {
  if (!("geolocation" in navigator)) {
    console.warn("Geolocation not supported by this browser.");
    return;
  }
  navigator.geolocation.getCurrentPosition(
    (pos) => {
      const lngLat = [pos.coords.longitude, pos.coords.latitude];
      if (!locationMarker) {
        const dot = document.createElement("div");
        dot.className = "my-location-dot";
        locationMarker = new maplibregl.Marker({ element: dot })
          .setLngLat(lngLat)
          .setPopup(new maplibregl.Popup({ offset: 12, closeButton: false })
            .setHTML(locationNote()))
          .addTo(map);
      } else {
        locationMarker.setLngLat(lngLat);
      }
      map.getSource("accuracy").setData(accuracyCircle(lngLat, pos.coords.accuracy));
      // Only follow the fix once it is actually inside the mapped area. A
      // fix from home, from another town, or a bad first read would otherwise
      // drag the view onto empty basemap with no stands on it at all -- there
      // is no data outside the mapped area, so there is nothing to look at
      // there. The whole area stays framed until then, and the first fix that
      // does land inside the map still centres on it, so driving in works as
      // before.
      if (firstFix && DATA_BOUNDS.contains(lngLat)) {
        map.jumpTo({ center: lngLat, zoom: 15 });
        firstFix = false;
      }
    },
    (err) => console.warn("Location lookup failed:", err.message),
    { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
  );
}

map.on("load", () => {
  for (const layer of STAND_LAYERS) {
    map.addLayer({ id: `${layer}-fill`, type: "fill", source: "stands", "source-layer": layer });
    map.addLayer({
      id: `${layer}-line`, type: "line", source: "stands", "source-layer": layer,
      // zoomed out, outlines around stands a few pixels wide would be all
      // there is to see, so they fade in as the stands grow
      paint: { "line-width": 1, "line-opacity": ["interpolate", ["linear"], ["zoom"], 9, 0.15, 13, 0.6] },
    });
  }
  map.addSource("accuracy", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
  map.addLayer({ id: "accuracy-fill", type: "fill", source: "accuracy",
                 paint: { "fill-color": "#1a73e8", "fill-opacity": 0.1 } });
  map.addLayer({ id: "accuracy-line", type: "line", source: "accuracy",
                 paint: { "line-color": "#1a73e8", "line-width": 1 } });
  addFlagImage();
  map.addSource("sightings", { type: "geojson", data: SIGHTINGS });
  map.addLayer({
    id: "sightings", type: "symbol", source: "sightings",
    layout: { "icon-image": "flag", "icon-anchor": "bottom-left", "icon-offset": [-4, 4],
              "icon-allow-overlap": true, "icon-ignore-placement": true },
  });

  mapReady = true;
  paintSpecies(active);
  updateLocation();
  setInterval(updateLocation, LOCATION_REFRESH_MS);
});
</script>
</body>
</html>
"""


def render_html(geojson_dict: dict) -> str:
    tiles = {"overview": OVERVIEW_LAYER, "detail": DETAIL_LAYER, "minzoom": TILE_MIN_ZOOM,
             "maxzoom": TILE_MAX_ZOOM, "detailMinZoom": DETAIL_MIN_ZOOM}
    html = HTML_TEMPLATE.replace("__TILES__", json.dumps(tiles))
    html = html.replace("__BOUNDS__", json.dumps(data_bounds(geojson_dict)))
    html = html.replace("__SPECIES__", json.dumps(species_config(), ensure_ascii=False))
    html = html.replace("__LABELS__", json.dumps(label_config(), ensure_ascii=False))
    html = html.replace("__FINNISH__", json.dumps(FINNISH, ensure_ascii=False))
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
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    write_tiles(geojson_dict)
    (SITE_DIR / "_headers").write_text(PAGES_HEADERS, encoding="utf-8")
    OUTPUT_HTML.write_text(render_html(geojson_dict), encoding="utf-8")

    tiles = list(TILES_DIR.rglob("*.pbf"))
    print(f"\nWrote {OUTPUT_GEOJSON} ({len(geojson_dict['features'])} stands, both species)")
    print(f"Wrote {len(tiles)} vector tiles to {TILES_DIR} "
          f"({sum(t.stat().st_size for t in tiles) / 1e6:.1f} MB)")
    print(f"Wrote {OUTPUT_HTML} ({OUTPUT_HTML.stat().st_size / 1e3:.0f} KB)")
    for cfg in species_config():
        print(f"  {cfg['name']}: {len(cfg['sightings']['features'])} laji.fi sightings embedded as flags")
    missing = untranslated()
    if missing:
        print(f"\nNo Finnish in scripts/translations.py for {len(missing)} strings, shown in English:")
        for text in missing:
            print(f"  {text!r}")


if __name__ == "__main__":
    main()
