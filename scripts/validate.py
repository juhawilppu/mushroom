"""Check whether a species' score actually ranks real sightings above random forest.

calibrate.py answers "which categories do sightings favour?", one attribute at
a time. This answers the question that matters once the weights are set: does
the finished score, all factors together, put real sighting locations near the
top of the map? Without it a model can look well calibrated attribute by
attribute and still rank no better than chance.

    AUC   probability that a random real sighting outscores a random stand in
          the mapped area. 0.5 = coin toss, 1.0 = perfect separation.
    lift  how much more likely a sighting is to be in the top slice of the
          map than chance would give. The slice is defined by a score cutoff,
          and scores TIE heavily -- the tables are discrete, so thousands of
          stands share a score and a nominal "top 10%" cutoff can select 19%
          of the map. Lift is therefore measured against the share actually
          selected, never against the nominal 15%: comparing a tie-heavy
          model to a tie-free one on the nominal figure flatters the tie-heavy
          one badly enough to reverse the conclusion.

Only the site factors are scored -- kasvupaikka, kehitysluokka, maapera,
valoisuus and maastonmuoto. The species and sekametsa terms are left out because the sighting
rows carry the WFS's PROPORTIONSPRUCE/PINE/OTHER while the mapped area's background
carries per-stratum basal areas, and scoring the two sides from different
inputs would flatter or punish the model for the wrong reason.

Both numbers are IN-SAMPLE: the weights were tuned against these same
sightings, so treat them as an upper bound, and as a way to compare model
versions against each other rather than as an absolute accuracy.

    python scripts/validate.py

Run scripts/calibrate.py first for each species, to build the sighting caches.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import build_map as bm
import calibrate
import species as sp
from species import PROFILES, SpeciesProfile

# The WFS rows the sighting cache is built from name the same attributes in
# upper case; everything else about scoring them is identical.
WFS_COLUMNS = {
    "fertilityclass": "FERTILITYCLASS",
    "developmentclass": "DEVELOPMENTCLASS",
    "soiltype": "SOILTYPE",
    "drainagestate": "DRAINAGESTATE",
    "stemcount": "STEMCOUNT",
    # terrain is read from the elevation model for both sides, so it is
    # already named the same way on sighting rows as on stands
    "tpi": "tpi_large",
    "slope": "slope",
}

TOP_SHARE = 0.15  # "Korkea" and above


def site_score(df: pd.DataFrame, profile: SpeciesProfile, columns=None) -> pd.Series:
    """The map's score, restricted to the site factors, limiting-factor
    penalty included -- that penalty is a large part of what the ranking does,
    so leaving it out would validate a model the map does not use."""
    points = bm.site_factor_points(df, profile, columns)
    budget = profile.factor_points
    ratios = {
        factor: (points[factor] / budget[factor]).clip(upper=1)
        for factor in ("fertility", "development", "soil", "light", "terrain")
    }
    weakest = pd.concat(
        [(ratio / profile.green_thresholds[factor]).clip(upper=1)
         for factor, ratio in ratios.items() if factor in sp.LIMITING_FACTORS],
        axis=1,
    ).min(axis=1).fillna(0)
    total = sum(points[factor] for factor in ratios)
    return total * (sp.LIMITING_FLOOR + (1 - sp.LIMITING_FLOOR) * weakest)


def auc(positive: np.ndarray, background: np.ndarray) -> float:
    """Mann-Whitney U as a probability, which is what AUC is."""
    ranks = pd.Series(np.concatenate([positive, background])).rank().to_numpy()
    n_pos, n_bg = len(positive), len(background)
    return (ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_bg)


def to_code(series: pd.Series) -> pd.Series:
    """WFS numeric codes -> the string-coded space the profiles are written in."""
    return series.dropna().astype(float).astype(int).astype(str).reindex(series.index)


def load_sightings(profile: SpeciesProfile) -> pd.DataFrame:
    path = calibrate.cache_path(profile)
    if not path.exists():
        raise SystemExit(
            f"No sighting cache at {path}.\n"
            f"Run: python scripts/calibrate.py --species {profile.slug}"
        )
    df = pd.DataFrame(json.loads(path.read_text()))
    for wfs in ("FERTILITYCLASS", "SOILTYPE", "DRAINAGESTATE", "MAINGROUP", "SUBGROUP"):
        df[wfs] = to_code(df[wfs])
    terrain = calibrate.sighting_terrain(profile, df)
    return pd.concat([df, terrain], axis=1)


def background(profile: SpeciesProfile) -> tuple[pd.DataFrame, pd.Series]:
    stand, growthplace, treestand, treestandsummary, _ = bm.load_layers()
    df = (stand.merge(growthplace, on="standid", how="left")
               .merge(treestand, on="standid", how="left")
               .merge(treestandsummary, on="treestandid", how="left"))
    return df, bm.is_excluded(df, profile)


def evaluate(profile: SpeciesProfile, sightings: pd.DataFrame,
             bg: pd.DataFrame, bg_excluded: pd.Series) -> dict:
    excluded = bm.is_excluded(sightings, profile, WFS_COLUMNS, "MAINGROUP", "SUBGROUP")
    positive = site_score(sightings, profile, WFS_COLUMNS)[~excluded].dropna().to_numpy()
    bg_scores = site_score(bg, profile)[~bg_excluded].dropna().to_numpy()
    cutoff = np.quantile(bg_scores, 1 - TOP_SHARE)
    selected = (bg_scores >= cutoff).mean()   # >= TOP_SHARE whenever scores tie
    caught = (positive >= cutoff).mean()
    return {
        "n": len(positive),
        "n_total": len(sightings),
        "excluded_share": excluded.mean(),
        "auc": auc(positive, bg_scores),
        "selected": selected,
        "caught": caught,
        "lift": caught / selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", choices=sorted(PROFILES),
                        help="validate only this species (default: all of them)")
    args = parser.parse_args()
    slugs = [args.species] if args.species else list(PROFILES)

    sightings = {slug: load_sightings(PROFILES[slug]) for slug in slugs}
    for slug in slugs:
        profile = PROFILES[slug]
        bg, bg_excluded = background(profile)
        r = evaluate(profile, sightings[slug], bg, bg_excluded)
        print(f"\n=== {profile.name} ===")
        print(f"  {r['n']} sightings scored (of {r['n_total']}; "
              f"{r['excluded_share']:.0%} sit on stands the model excludes outright)")
        print(f"  AUC:  {r['auc']:.3f}")
        print(f"  top slice holds {r['selected']:.0%} of stands and catches "
              f"{r['caught']:.0%} of sightings -> {r['lift']:.2f}x chance")

    if len(slugs) > 1:
        # Each species' sightings scored by BOTH models. If a profile is
        # genuinely species-specific, it should beat the other one on its own
        # species; if the numbers only reflect which model is better overall,
        # the profiles are telling the species apart on the strength of their
        # author's priors rather than on the evidence.
        print("\n=== cross-model AUC (rows: whose sightings, columns: which model) ===")
        width = max(len(PROFILES[s].name) for s in slugs) + 2
        header = "".join(f"{PROFILES[s].name:>{width}}" for s in slugs)
        print(f"{'':>{width}}{header}")
        for slug in slugs:
            cells = ""
            for model_slug in slugs:
                profile = PROFILES[model_slug]
                bg, bg_excluded = background(profile)
                cells += f"{evaluate(profile, sightings[slug], bg, bg_excluded)['auc']:>{width}.3f}"
            print(f"{PROFILES[slug].name:>{width}}{cells}")


if __name__ == "__main__":
    main()
