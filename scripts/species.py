"""Per-species habitat profiles for the mushroom maps.

Everything that differs between the mushrooms lives here: which stands are
habitat at all, how each forestry attribute scores, how many points each
factor is worth, and what the map calls things. `build_map.py` is a single
scoring engine driven by these profiles, so adding a species is a matter of
writing one more profile rather than forking the pipeline.

The two profiles are deliberately near-opposites in places -- kantarelli
wants dry, well-drained, light-flooded mineral soil near eskers, while
suppilovahvero wants damp, shady, moss-floored spruce forest and is happy on
peat -- which is exactly why the two maps are worth having side by side.

Point tables are calibrated against real laji.fi sightings; see
`scripts/calibrate.py` and the per-table comments.
"""

from dataclasses import dataclass, field

# --- shared code tables (metsätietostandardi), species-independent ---------

# UI labels are English. The Finnish forest site types keep their standard
# abbreviations (OMT, MT, ...), which is what Finnish sources call them too.
FERTILITY_LABELS = {
    "1": "Herb-rich forest", "2": "Herb-rich heath forest (OMT)", "3": "Mesic heath forest (MT)",
    "4": "Sub-xeric heath forest (VT)", "5": "Xeric heath forest (CT)", "6": "Barren heath forest",
    "7": "Rocky or sandy ground", "8": "Hilltop or fell forest",
}

DEVELOPMENT_LABELS = {
    "02": "Young thinning stand", "03": "Advanced thinning stand",
    "04": "Mature, regeneration-ready stand", "05": "Shelterwood stand",
    "ER": "Uneven-aged stand", "S0": "Seed-tree stand",
    "Y1": "Sapling stand with overstorey", "T2": "Sapling stand",
}

SOIL_LABELS = {
    "10": "Coarse mineral soil", "11": "Coarse till", "12": "Coarse sorted soil",
    "20": "Fine mineral soil", "21": "Fine till", "22": "Fine sorted soil",
    "23": "Silty soil", "24": "Clay",
    "30": "Stony coarse mineral soil", "31": "Stony coarse till",
    "32": "Stony coarse sorted soil", "40": "Stony fine mineral soil",
    "50": "Bedrock or boulders", "60": "Peat", "61": "Sedge peat", "62": "Sphagnum peat",
    "70": "Humus soil", "80": "Mud soil",
}

# Shown in brackets after the soil type ("Coarse mineral soil (undrained)"),
# so the "heath" half of each official name is left off as redundant.
DRAINAGE_LABELS = {
    "1": "undrained", "2": "turning boggy", "3": "ditched",
    "6": "undrained mire", "7": "freshly ditched mire", "8": "drained mire, changing",
    "9": "drained peatland forest",
}

TREESPECIES_LABELS = {
    "1": "Scots pine", "2": "Norway spruce", "3": "Silver birch", "4": "Downy birch",
    "5": "Aspen", "6": "Grey alder", "7": "Black alder",
    "29": "Broadleaf (unspecified)", "30": "Conifer (unspecified)",
}

# Gini-Simpson bands. A description of the stand itself, not of how good that
# stand is for a given mushroom, so both species share it.
MIXTURE_BANDS = [(0.55, "Well-mixed forest"), (0.3, "Somewhat mixed")]
MIXTURE_FALLBACK = "Nearly a single species"

EXCLUDED_DEVELOPMENT = {"A0", "T1"}  # Aukea, Taimikko alle 1.3 m -- no forest floor yet
MID_THRESHOLD = 0.3  # ratio below which a factor's badge turns red, for every species

# Factors the limiting-factor penalty is measured over. Liebig's law is about
# necessities -- a stand with no light, or no host tree, cannot be rescued by
# its other qualities. Landform is not one of those: it is a proxy for drainage
# and exposure, conditions the soil and fertility factors already speak to, so
# letting it veto an otherwise excellent stand punishes the same weakness twice
# and (measurably) dilutes the top of the map.
LIMITING_FACTORS = ("fertility", "development", "species", "mixture", "light", "soil")

# Share of the raw point total a stand keeps when one factor is at rock
# bottom. 1.0 would be a pure sum (full compensation between factors);
# lower values make the worst factor bite harder (Liebig's law of the minimum).
LIMITING_FLOOR = 0.7

ESKER_BUFFER_M = 150


@dataclass(frozen=True)
class LightProfile:
    """Canopy density response, as a piecewise-linear curve over stems/ha.

    Both species care about how much light reaches the floor; they just want
    opposite ends of it. The curve is a band rather than a slope either way,
    because a nearly treeless stand has no living mycorrhizal host no matter
    how the species feels about shade.
    """

    stemcount_knots: list[float]
    suitability_knots: list[float]
    row_label: str
    label_good: str
    label_mid: str   # off the preferred band, but not badly
    label_poor: str  # far off it, in the same direction as label_mid
    # Below this stem density the stand is simply too empty to host anything,
    # which is a different failure from "wrong kind of light" and has to be
    # said differently -- otherwise a nearly treeless stand reads as "dense".
    sparse_stems: float
    label_sparse: str


@dataclass(frozen=True)
class TerrainProfile:
    """Where in the landscape the stand sits, from the 10 m elevation model.

    The forest inventory cannot tell a dry crest from the damp hollow below it
    -- both can carry the same spruce on the same soil -- so this is the one
    factor with any chance of separating two mushrooms that want the same kind
    of forest for different reasons. Two piecewise-linear responses, on
    landform position (TPI) and on slope, combined by `tpi_weight`.
    """

    tpi_knots: list[float]
    tpi_suitability: list[float]
    slope_knots: list[float]
    slope_suitability: list[float]
    tpi_weight: float          # the rest of the weight goes to slope
    row_label: str
    # Landform wording is a description of the ground, not of how good it is,
    # so both species share LANDFORM_BANDS below and this only names the row.


# Landform description from TPI at the 500 m scale, in metres above or below
# the surrounding terrain. Shared by both species: it says what the ground is,
# not whether the mushroom likes it.
LANDFORM_BANDS = [
    (8, "Hilltop or ridge"),
    (3, "Upper slope"),
    (1, "Gentle rise"),
    (-1, "Level with its surroundings"),
    (-3, "Shallow dip"),
    (-8, "Hollow"),
]
LANDFORM_FALLBACK = "Deep hollow"
# Appended to the landform when the ground actually falls somewhere, from slope
# in degrees. Below the last threshold the landform wording says enough on its
# own and a redundant "flat" only makes the row longer.
SLOPE_BANDS = [(11, "steep"), (7, "sloping"), (4, "gently sloping")]


@dataclass(frozen=True)
class SpeciesProfile:
    slug: str
    map_key: str         # short property key this species' scores ship under
    name: str            # English name, as shown in the UI
    latin: str
    laji_target: str     # laji.fi search target (scientific name)
    intro: str           # one-line habitat summary for the legend

    fertility_points: dict[str, float]
    development_points: dict[str, float]
    soil_points: dict[str, float]
    soil_default: float
    drainage_multiplier: dict[str, float]
    drainage_default: float
    species_weight: dict[str, float]
    species_weight_default: float

    excluded_subgroup: set[str]
    excluded_fertility: set[str]

    mixture_points: float
    species_points: float
    light_points: float
    light: LightProfile
    terrain_points: float
    terrain: TerrainProfile

    green_thresholds: dict[str, float]
    esker_points: float = 0.0
    esker_row_labels: tuple[str, str] = ("", "")
    default_fertility_points: float = 6.0
    default_development_points: float = 8.0

    # factor -> point budget, so the "x/100" on the map stays honest
    @property
    def factor_points(self) -> dict[str, float]:
        return {
            "fertility": max(self.fertility_points.values()),
            "development": max(self.development_points.values()),
            "species": self.species_points,
            "mixture": self.mixture_points,
            "light": self.light_points,
            "soil": max(self.soil_points.values()),
            "terrain": self.terrain_points,
        }

    @property
    def max_raw_score(self) -> float:
        return sum(self.factor_points.values()) + self.esker_points

    @property
    def uses_esker(self) -> bool:
        return self.esker_points > 0


KANTARELLI = SpeciesProfile(
    slug="kantarelli",
    map_key="k",
    name="Chanterelle",
    latin="Cantharellus cibarius",
    laji_target="Cantharellus cibarius",
    intro="Fairly dry, well-lit heath forest on free-draining soil.",

    fertility_points={
        # Recalibrated against sightings placed to within 100 m (see the note
        # on COORDINATE_ACCURACY_MAX_M in calibrate.py -- the earlier 1 km cap
        # matched a quarter of records to whichever stand sat under a very
        # fuzzy centre point). On the clean set MT stays at prevalence (1.1x),
        # VT is clearly favoured (1.5x) and OMT is avoided relative to how
        # common it is (0.6x) -- so VT moves above OMT here, which is the
        # opposite of the ordering fertility class alone would suggest.
        "1": 17,  # Lehto - lush but often too dense/herby
        "2": 16,  # Lehtomainen kangas (OMT) - real sightings avoid it (0.6x)
        "3": 25,  # Tuore kangas (MT) - prime, and the bulk of all sightings
        "4": 21,  # Kuivahko kangas (VT) - genuinely favoured (1.5x)
        "5": 12,  # Kuiva kangas (CT) - enriched, but on ~2% of sightings
        "6": 4,   # Karukkokangas
        "7": 0,   # Kalliomaa ja hietikko
        "8": 0,   # Lakimetsä ja tunturi
    },
    development_points={
        # The strongest signal in either species' calibration, and it got
        # stronger on the precise-coordinate set: regeneration-ready stands
        # (04) carry 47% of sightings against 10% of the available forest, a
        # 4.5x enrichment, while 03 -- nearly two thirds of Karkkila -- sits
        # at 0.7x and young stands at 0.2x. The gap between 03 and 04 was
        # previously too narrow to separate them on the map; widening it is
        # the single change that most improves how the score ranks real
        # sightings, so 03 is scored for what it is: the default forest here,
        # not a destination.
        "02": 8,   # Nuori kasvatusmetsikkö - strongly avoided (0.2x)
        "03": 17,  # Varttunut kasvatusmetsikkö - common, under-represented (0.7x)
        "04": 25,  # Uudistuskypsä metsikkö - by far the best predictor (4.5x)
        "05": 12,  # Suojuspuumetsikkö
        "ER": 16,  # Eri-ikäisrakenteinen
        "S0": 5,   # Siemenpuumetsikkö - too open
        "Y1": 5,   # Ylispuustoinen taimikko
        "T2": 3,   # Taimikko yli 1.3 m - too young
    },
    soil_points={  # coarse/well-drained mineral soils score best
        "10": 15, "11": 15, "12": 15, "30": 14, "31": 14, "32": 14,
        "20": 9, "21": 9, "22": 9, "23": 8, "24": 7, "40": 8,
        "50": 8,
        "70": 7,  # Multamaa - organic-rich, holds moisture, middling for kantarelli
    },
    soil_default=7,
    drainage_multiplier={
        "1": 1.0,  # Ojittamaton kangas - natural
        "3": 0.8,  # Ojitettu kangas - ditched, altered hydrology
        "2": 0.5,  # Soistunut kangas - paludified
    },
    drainage_default=0.2,  # everything else here is peatland hydrology
    species_weight={
        # Calibrated against real laji.fi sightings (scripts/calibrate.py):
        # Mänty-dominant stands were 2x over-represented at real sighting
        # locations relative to their prevalence -- pine is a much stronger
        # chanterelle host here than a low weight would suggest, raised sharply.
        # Lehtipuu (unspecified broadleaf) was previously raised on the
        # assumption that unresolved broadleaf is mostly birch, but real
        # sightings are 10x LESS common there than prevalence would predict --
        # that assumption doesn't hold up against the data, so it's lowered
        # back down, below its original default even.
        "2": 1.0,   # Kuusi / Norway spruce - main host, matches real sightings closely
        "1": 0.75,  # Mänty / Scots pine - real sightings show this is a strong host too
        "3": 0.6,   # Rauduskoivu / silver birch (too few sightings to recalibrate)
        "4": 0.6,   # Hieskoivu / downy birch (too few sightings to recalibrate)
        "30": 0.4,  # Havupuu / unspecified conifer - no sighting data to calibrate against
        "29": 0.2,  # Lehtipuu / unspecified broadleaf - real sightings clearly avoid this
    },
    species_weight_default=0.1,
    excluded_subgroup={"2", "3", "4", "5"},  # Korpi, Räme, Neva, Letto - mire types
    excluded_fertility={"7", "8"},
    # the mixture half outweighs raw host quality, per specific request to
    # emphasize genuinely mixed forest over a monoculture of the "best" species
    mixture_points=15,
    species_points=10,
    light_points=10,
    light=LightProfile(
        # "valoisuus": repeatedly cited in foraging sources ("avoid dense dark
        # forest"), but not captured by development class alone -- a stand can
        # have high basal area from a few big old trees (open) or many small
        # crowded ones (dark) at the same age. Calibrated against real laji.fi
        # sightings: 74% fall in 300-800 stems/ha (vs 44% of background),
        # dropping off sharply above 800 and effectively absent below 300.
        stemcount_knots=[0, 150, 350, 800, 1600, 2400],
        suitability_knots=[0.15, 0.3, 1.0, 1.0, 0.3, 0.0],
        row_label="Light",
        label_good="Open, well-lit",
        label_mid="Fairly dense",
        label_poor="Dense, little light",
        sparse_stems=400,
        label_sparse="Very sparse trees",
    ),
    terrain_points=15,
    terrain=TerrainProfile(
        # Calibrated against sighting locations sampled from the 10 m
        # elevation model. The clearest terrain signal for either species is
        # what they AVOID: ground sitting 3 m or more below its surroundings
        # holds a quarter of Karkkila's forest but only an eighth of the
        # sightings (0.5x), which is a stronger depletion than any soil or
        # fertility class shows. Depressions here are wet, and wet is where
        # neither of these mushrooms fruits. Level to gently raised ground is
        # the sweet spot (1.6-1.7x); crests fall back slightly, being thin and
        # dry.
        tpi_knots=[-25, -8, -3, -1, 2, 8, 25],
        tpi_suitability=[0.25, 0.3, 0.55, 1.0, 1.0, 0.8, 0.7],
        # Slope barely moves kantarelli sightings at all (0.9-1.1x across the
        # range), so it is kept as a gentle preference rather than a real
        # constraint, and weighted low against TPI below.
        slope_knots=[0, 1, 3, 9, 14, 30],
        slope_suitability=[0.75, 0.9, 1.0, 1.0, 0.8, 0.6],
        tpi_weight=0.75,
        row_label="Landform",
    ),
    green_thresholds={
        "fertility": 0.65,
        "development": 0.65,
        "species": 0.65,
        "mixture": 0.55,   # Gini-Simpson tops out near 0.67 in practice
        "light": 0.75,     # roughly 325-1100 stems/ha, the empirically enriched band
        "soil": 0.65,
        "terrain": 0.7,     # level or gently raised ground, not a depression
    },
    esker_points=10,
    esker_row_labels=("Near an esker or ice-marginal formation", "Not near an esker"),
)


SUPPILOVAHVERO = SpeciesProfile(
    slug="suppilovahvero",
    map_key="s",
    name="Funnel chanterelle",
    latin="Craterellus tubaeformis",
    laji_target="Craterellus tubaeformis",
    intro="Mature, spruce-dominated heath forest; copes with damper and poorer ground than the chanterelle.",

    fertility_points={
        # Calibrated against 710 real Craterellus tubaeformis sightings from
        # laji.fi (scripts/calibrate.py). The folk description -- damp, rich,
        # mossy ground -- turns out not to survive contact with the data: MT
        # sits exactly at prevalence, OMT is clearly UNDER-represented (0.6x),
        # and the drier VT is over-represented (1.7x). The dry end (CT,
        # karukko) is enriched too, but on so few sightings that it only earns
        # a mild score rather than a top one.
        "1": 12,  # Lehto - herb layer crowds out the moss carpet
        "2": 15,  # Lehtomainen kangas (OMT) - under-represented relative to how common it is
        "3": 20,  # Tuore kangas (MT) - over half of all sightings
        "4": 17,  # Kuivahko kangas (VT) - genuinely favoured, not merely tolerated
        "5": 10,  # Kuiva kangas (CT) - enriched, but on thin evidence
        "6": 5,   # Karukkokangas
        "7": 0, "8": 0,
    },
    default_fertility_points=5,
    development_points={
        # The single strongest signal in the calibration, and the same one the
        # kantarelli model found: regeneration-ready stands (04) carry 45% of
        # sightings against 10% of the available forest, a 4.4x enrichment,
        # while young stands (02) and thickets are 5x AVOIDED. Whatever
        # suppilovahvero gets from a dense young spruce stand, it is not
        # fruiting bodies -- so this profile gives development class the
        # largest point budget of any factor.
        "02": 8,   # Nuori kasvatusmetsikkö - 0.2x, strongly avoided
        "03": 17,  # Varttunut kasvatusmetsikkö - common, but slightly under-represented
        "04": 25,  # Uudistuskypsä metsikkö - by far the best predictor
        "05": 12,  # Suojuspuumetsikkö
        "ER": 16,  # Eri-ikäisrakenteinen - has the old cohort that matters
        "S0": 5,   # Siemenpuumetsikkö
        "Y1": 5,   # Ylispuustoinen taimikko
        "T2": 4,   # Taimikko yli 1.3 m
    },
    default_development_points=6,
    soil_points={
        # Another prior the data overturned: fine-textured, moisture-holding
        # soils were expected to lead and instead came in at 0.6x, while plain
        # coarse mineral soil (1.4x) and stony ground (2.2x) are where the
        # sightings actually are. Rock -- which the kantarelli model treats as
        # a dead end -- is enriched 4.8x here, which fits the mossy,
        # boulder-strewn spruce forest this species is picked in. Peat sits
        # near prevalence, so it is neither the point nor a disqualifier.
        "10": 15, "11": 15, "12": 14, "30": 15, "31": 14, "32": 13,
        "50": 12,  # Kallio/kivikko - genuinely productive, unlike for kantarelli
        "60": 12, "61": 11, "62": 10, "70": 10, "80": 9,
        "20": 10, "21": 9, "22": 10, "23": 9, "24": 8, "40": 11,
    },
    soil_default=11,
    drainage_multiplier={
        # Here the damp-ground reputation does hold up: paludified kangas
        # (2.0x) and undrained mire (1.8x) are both enriched, and ditched
        # ground is the only state that is genuinely under-represented. So
        # unlike the kantarelli model, wetness is not penalised -- it is just
        # not the strong positive the fertility and soil tables were expected
        # to show.
        "1": 1.0,   # Ojittamaton kangas
        "2": 1.0,   # Soistunut kangas - paludified, 2x enriched
        "3": 0.85,  # Ojitettu kangas - the one state sightings avoid
        "6": 0.85,  # Ojittamaton suo
        "7": 0.9,   # Ojikko
        "8": 0.85,  # Muuttuma
        "9": 0.85,  # Turvekangas
    },
    drainage_default=0.85,
    species_weight={
        # Spruce dominates the sightings (66%, and a mean spruce proportion of
        # 0.49 at sighting locations), which is the one part of the textbook
        # description the data backs without reservation. Pine still comes out
        # over-represented against its own prevalence, the same way it did for
        # kantarelli, so it is scored as a real if secondary host rather than
        # the near-miss the literature implies.
        "2": 1.0,   # Kuusi - the host that matters
        "1": 0.6,   # Mänty - over-represented against prevalence
        "30": 0.8,  # Havupuu / unspecified conifer - mostly spruce here
        "3": 0.4,   # Rauduskoivu
        "4": 0.4,   # Hieskoivu
        "29": 0.15, # Lehtipuu / unspecified broadleaf - all but absent from sightings
    },
    species_weight_default=0.15,
    # Korpi (spruce mire) stays in, unlike in the kantarelli model: it is
    # forest with a spruce canopy and the drainage evidence above says damp
    # ground is fine. Only the treeless and pine-bog mire types go.
    excluded_subgroup={"3", "4", "5"},  # Räme, Neva, Letto
    excluded_fertility={"7", "8"},
    # a spruce monoculture is a perfectly good suppilovahvero forest, so
    # mixedness stays a mild positive at most -- the weight that the kantarelli
    # model puts on sekametsä goes to development class and host species here
    mixture_points=5,
    species_points=20,
    light_points=15,
    light=LightProfile(
        # Calibrated off stem density at sighting locations: 300-600 stems/ha
        # is enriched 1.9x and 600-900 sits at prevalence, with a steady
        # falloff above (0.4x at 1400-2200, 0.2x beyond). So this species does
        # NOT want the dark dense thicket it is usually described as haunting;
        # it wants roughly the same half-open canopy kantarelli does, just
        # holding on somewhat better into denser forest.
        stemcount_knots=[0, 150, 350, 900, 1600, 2600],
        suitability_knots=[0.1, 0.25, 1.0, 1.0, 0.3, 0.1],
        row_label="Light",
        label_good="Open, well-lit",
        label_mid="Fairly dense",
        label_poor="Dense, little light",
        sparse_stems=350,
        label_sparse="Very sparse trees",
    ),
    terrain_points=15,
    terrain=TerrainProfile(
        # Same avoidance of depressions as kantarelli (0.6x below -3 m), which
        # is the opposite of what the "damp hollows and ditch banks" reputation
        # predicts -- see the note on the species comparison in README. The
        # peak sits slightly further up the hillside than kantarelli's (3-8 m
        # above the surroundings, 1.4x).
        tpi_knots=[-25, -8, -3, 0, 4, 10, 25],
        tpi_suitability=[0.3, 0.35, 0.6, 0.85, 1.0, 1.0, 0.85],
        # This is the one terrain metric where the two species measurably
        # differ: suppilovahvero sightings avoid flat ground (0.6x below 2
        # degrees) and favour distinctly sloping ground (1.5x at 7-11
        # degrees), while kantarelli is indifferent to slope. The effect is
        # small -- a head-to-head AUC of 0.553 between the two species -- so
        # it is weighted as a real but modest preference.
        slope_knots=[0, 1, 3, 7, 12, 20, 30],
        slope_suitability=[0.5, 0.7, 0.95, 1.0, 1.0, 0.85, 0.7],
        tpi_weight=0.6,
        row_label="Landform",
    ),
    green_thresholds={
        "fertility": 0.7,   # MT, VT and OMT
        # Only uudistuskypsä (04) counts as green here. That is a hard line --
        # it puts a yellow dot on the varttunut kasvatusmetsikkö that most of
        # Karkkila consists of -- but 04 is the category real sightings pick
        # out 4.4x over prevalence while 03 sits at 0.7x, and without an esker
        # gate to play the role it plays on the kantarelli map, this is the
        # factor that keeps "Excellent" a short list worth walking to.
        "development": 0.72,
        "species": 0.7,     # spruce-dominated
        "mixture": 0.45,
        "light": 0.75,      # roughly 300-1200 stems/ha
        "soil": 0.75,       # mineral or stony ground in a natural drainage state
        "terrain": 0.7,     # off the valley floor, on ground with some fall to it
    },
)


PROFILES = {p.slug: p for p in (KANTARELLI, SUPPILOVAHVERO)}
