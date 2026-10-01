# Mushroom

A mushroom map of nine municipalities in southern Finland, showing
which patches of forest are most likely to grow chanterelles (*Cantharellus cibarius*) and funnel
chanterelles (*Craterellus tubaeformis*). Karkkila, Vihti, Lohja, Siuntio, Espoo,
Kirkkonummi, Helsinki, Vantaa and Sipoo form one continuous area, from
Karkkila and Lohja down through Vihti and Nuuksio to the coast, and along it
from Siuntio to Sipoo. It's built from Finland's open forest inventory data rather than
guesswork, and it's made to be used on your phone, in the woods. Each mushroom has its own map, and the buttons at
the top switch between them.

Karkkila is my home town. I've spent years walking its forests looking for
chanterelles and have never once come home with enough to actually cook, so
this is my fix for that. It's also why the map started with this one town
rather than all of Finland, before growing to take in Vihti, Lohja, Nuuksio
and the coast from Siuntio to Sipoo. And it seems to help: the chanterelle map found mushrooms on
its very first outing, which is how the funnel chanterelle earned a map of
its own.

In Finnish they're *kantarelli* and *suppilovahvero*, names you'll still see
in the code.

**Live map: [mushroom.juhawilppu.com](https://mushroom.juhawilppu.com)**

![Screenshot of the Karkkila mushroom map: a Chanterelle / Funnel chanterelle switcher above forest stands shaded by probability, with an open popup breaking one stand's score down factor by factor — a green, yellow or red dot per factor showing which ones earned the score and which held it back](docs/screenshot.jpg)

## What it does

Chanterelles are picky about where they grow. They favour mature spruce or
pine forest that isn't too dark or too wet, on well-drained soil, often near
an esker (a ridge of sand and gravel left behind by the last ice age). None
of that is guesswork: Finland's forest inventory records nearly all of it,
stand by stand, and the Geological Survey of Finland maps the eskers.

This project takes that data for each of the roughly 154,000 forest stands
in those nine municipalities and scores every stand against those habitat
preferences. The stands are then ranked against each other instead of
against a fixed cutoff. Most of this area is perfectly decent spruce forest,
so a fixed cutoff would call nearly all of it "good", and that doesn't tell
you where to go. The ranking spans the whole area rather than each
municipality on its own, so a colour means the same thing on both sides of a
municipal border, and Nuuksio is split between three municipalities.

On the map, a stand is:

- **Excellent** if every factor it's scored on looks right for the mushroom,
  with no exceptions
- **High** if it ranks in the top 15% of the rest
- **Moderate** if it ranks in the next 35%

The bottom half isn't drawn at all.

Espoo comes out well. Its forest is noticeably older than Karkkila's or
Vihti's: 37% of the stands it scores are mature and ready for felling,
against 10% in Karkkila, and that's the strongest single signal both models
have. Expect a lot of green in Espoo, Nuuksio included.

A blank patch doesn't always mean poor forest, though. The open inventory
doesn't cover every forest: only about half of Nuuksio National Park's area
has stands in it, so parts of the park are simply missing from the map.

Zoom in and tap any stand to see its score and the forest data behind it
(zoomed out, a tap takes you in close enough to pick one). A green,
yellow or red dot on each row shows which factors helped and which held it
back. On a phone the map also shows where you are, updated every 30 seconds,
and each popup has a button for directions there in Google Maps.

The map is in Finnish and English. It opens in Finnish if your browser's
first language is Finnish or your device is on Finnish time (plenty of Finns
run their phones in English), and in English otherwise. The FI / EN switch at
the foot of the legend changes it and remembers the choice.

## The two species

Each mushroom has its own habitat model in `scripts/species.py`.

The chanterelle wants fairly dry, half-open forest on well-drained mineral
soil, ideally near an esker. The funnel chanterelle is usually described as
its opposite, a creature of dank, shady spruce mire, but checking that
description against real sightings mostly disproved it. The sightings point
to mature stands ready for felling (4.6x over-represented), a canopy no
denser than the chanterelle's, coarse and even stony soil rather than fine,
damp soil, and *drier* forest types than expected. What held up was a
stronger preference for spruce (70% of sightings are in spruce-dominated
stands, against 62% for the chanterelle) and a real tolerance for boggy
ground. The funnel chanterelle map keeps spruce mires, which the chanterelle
model rules out.

The honest summary is that **these two mushrooms are hard to tell apart using
forest data**. Their sightings land in the same kinds of forest to within a
few percentage points on forest type, stand age, soil and tree density, and
a model tuned for one scores the other's sightings about as well as its own.
Where they really differ is in how much they like spruce, how well they cope
with wet ground, how they feel about slopes and, most usefully, *season*:
85% of funnel chanterelle sightings are from September to November and
almost none are from before August, while chanterelles peak in July and
August. If you're out in October, you're probably looking for funnel
chanterelles.

The comments next to each species' scoring tables record the evidence behind
every weight, including the places where the data contradicted the field
guides.

## Landform

The forest inventory describes a stand, but not where it sits. Two stands
with identical records can be a dry hilltop and the damp hollow below it. So
every stand is also scored on its landform, read from the national 10 m
elevation model: how high it sits relative to the ground around it, and how
steep it is. That was the biggest single improvement either model has had.
For both species, it lifted how densely sightings concentrate in the
best-scoring tenth of the map from about 2.5x what chance would give to about
4x.

The finding is blunt, and it's the same for both mushrooms: ground sitting
3 m or more below its surroundings makes up nearly a third of Karkkila's
forest but holds only about a fifth of the sightings (0.6x). Hollows here are
wet, and neither mushroom likes wet ground, which is the opposite of what the
funnel chanterelle's "damp hollows and ditch banks" reputation predicts.
Level or gently raised ground is the sweet spot. Hilltops do slightly worse,
their soil being thin and dry.

## Does it work?

`scripts/validate.py` scores real sighting locations with the map's own code
and checks whether they come out ahead of randomly picked forest:

| | AUC | top 15% of the map catches |
|---|---|---|
| Chanterelle | 0.753 | 48% of sightings — 3.2x chance |
| Funnel chanterelle | 0.767 | 47% of sightings — 3.2x chance |

AUC is the chance that a random sighting outscores a random patch of forest:
0.5 is a coin flip and 1.0 would be perfect. Both numbers are in-sample, since
the weights were tuned on these same sightings, so treat them as an upper
bound and as a way to compare versions of the model, not as its true
accuracy.

These figures were measured against Karkkila's forest, before the map grew
to cover the other eight municipalities; the script now measures against the whole mapped
area.

Two lessons are built into that script. Sighting coordinates are capped at
100 m accuracy: stands are only 1–3 ha, and a record that's only accurate to
a kilometre describes the forest someone walked through, not the one the
mushroom grew in. And lift is measured against the share of stands a cutoff
*actually* selects. Scores come from discrete tables, so thousands of stands
tie, and a nominal "top 10%" can select 19% of the map. Comparing models on
the nominal figure once made a genuine improvement look like a step
backwards.

This is a habitat model, not a mushroom detector. The weights are calibrated
against real sightings from laji.fi, but the map can only tell you where the
forest looks right, so treat it as "worth a look" rather than a promise. The
sightings are also opportunistic: they partly record where foragers happen
to walk, and the map inherits some of that bias.

And a reminder: in Finland, *jokamiehenoikeus* (everyman's right) lets anyone
pick mushrooms in almost any forest, whoever owns it. Just stay out of
people's yards, and check the rules if you're in a nature reserve.

## Data sources

- [Suomen metsäkeskus](https://www.metsakeskus.fi/fi/avoin-metsa-ja-luontotieto)
  (Finnish Forest Centre): open forest inventory data per stand, including
  tree species, development class, stem density, site type, soil and
  drainage.
- [GTK](https://www.gtk.fi/) (Geological Survey of Finland): glacial landform
  polygons, used to find eskers, via ArcGIS REST.
- [Maanmittauslaitos](https://www.maanmittauslaitos.fi/) (National Land
  Survey of Finland): the 10 m lidar-derived elevation model, used for
  landform. It's read with HTTP range requests from the openly mirrored
  nationwide file at funet, so only the parts covering the mapped
  municipalities are ever downloaded.
- [laji.fi](https://laji.fi/) (Finnish Biodiversity Information Facility):
  real sighting coordinates per species. They're used to calibrate the
  scoring weights and, for sightings inside the mapped area, as the flags on
  the map. Requires a free API token; see `scripts/calibrate.py`.

## Setup

Requires Python 3.11+ (the system Python on macOS is usually too old for
current geopandas) and [tippecanoe](https://github.com/felt/tippecanoe),
which cuts the map into vector tiles.

```
brew install tippecanoe
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```
python scripts/download_data.py   # downloads + caches source data under data/
python scripts/build_map.py       # scores stands for every species, writes output/site/
python -m http.server -d output/site   # then open http://localhost:8000
```

`output/site/` is the whole website: a small page (`index.html`), the stands
as vector tiles (`tiles/`, not kept in git), and a `_headers` file telling
Cloudflare Pages to serve the tiles compressed. The map is drawn on the GPU
with [MapLibre](https://maplibre.org/), and a phone downloads only the tiles
for the part of the map on screen, a few tens of kilobytes when zoomed in,
however many municipalities the map covers. Both species share the same
tiles: stand shapes are stored once, and the labels are filled in by the
browser from inventory codes. The page reads its tiles over HTTP, so opening
`index.html` straight from disk shows an empty map; serve the folder instead.

To recalibrate a species against real sightings, put a free
[laji.fi](https://laji.fi/) API token in `.env` as `LAJI_FI_TOKEN=...`, then
run `python scripts/calibrate.py --species suppilovahvero`. It prints how
often each category turns up at sightings compared with the mapped area's
forest as a whole. Use that to adjust the weights in that species' profile in
`scripts/species.py` by hand. (The code knows the species by their Finnish
names, `kantarelli` and `suppilovahvero`.)

To check whether a change to the weights actually helped, run
`python scripts/validate.py`.

To add a mushroom, write another `SpeciesProfile` in `scripts/species.py`
and add it to `PROFILES`. The scoring, the map, the switcher and the
sighting downloads all pick it up from there.

Everything on the page is written in English, in the code and in the label
tables in `scripts/species.py`. The Finnish lives in `scripts/translations.py`,
keyed by that English. Anything without an entry shows in English on the
Finnish page, and `build_map.py` lists any such text from `species.py` when it
writes the page, so a new mushroom or a reworded label needs a line there too.

To change which municipalities are mapped, edit `MUNICIPALITIES` in
`scripts/area.py`, then rerun both scripts. Each municipality's data is
downloaded and cached separately, so adding one only fetches that one.
