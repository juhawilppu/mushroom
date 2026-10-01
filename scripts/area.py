"""The area the mushroom maps cover, and where each municipality's data lives.

The map is one continuous area built from several neighbouring
municipalities: Karkkila, Vihti south of it, Espoo south of that and
Kirkkonummi west of Espoo, which between them take in the whole of Nuuksio;
Helsinki, Vantaa and Sipoo running east from Espoo along the coast; and
Lohja and Siuntio to the west, between Karkkila and Kirkkonummi.
Every source is downloaded and cached per municipality, so adding one to
MUNICIPALITIES fetches only that one's data, but the stands are scored and
ranked as a single population: a stand's colour has to mean the same thing
on both sides of a municipal border, and Nuuksio is split between three
municipalities.
"""

from pathlib import Path

MUNICIPALITIES = ["Karkkila", "Vihti", "Espoo", "Kirkkonummi", "Helsinki", "Vantaa", "Sipoo",
                  "Lohja", "Siuntio"]

# "Karkkila, Vihti, ... and Siuntio", for log lines
AREA_NAME = (", ".join(MUNICIPALITIES[:-1]) + " and " + MUNICIPALITIES[-1]
             if len(MUNICIPALITIES) > 1 else MUNICIPALITIES[0])

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
CACHE_DIR = ROOT / "data" / "cache"


def gpkg_path(municipality: str) -> Path:
    return RAW_DIR / f"MV_{municipality}" / f"MV_{municipality}.gpkg"


def gtk_formations_path(municipality: str) -> Path:
    return CACHE_DIR / f"gtk_formations_{municipality}.geojson"


def dem_path(municipality: str) -> Path:
    return CACHE_DIR / f"dem10m_{municipality}.npz"


def terrain_path(municipality: str) -> Path:
    return CACHE_DIR / f"terrain_stands_{municipality}.npz"


def laji_sightings_path(municipality: str, slug: str) -> Path:
    return CACHE_DIR / f"laji_sightings_{municipality}_{slug}.json"
