"""Finnish for the text the map shows.

The page and the label tables in species.py are written in English, and the
English doubles as the key here, the way gettext uses its msgids: the browser
looks each string up as it shows it and falls back to the English when there
is no entry. A label added or reworded in species.py therefore turns up in
English on the Finnish page until it gets a line here; build_map.py lists any
that are missing when it writes the page.

The forest inventory's own vocabulary follows the Finnish Forest Centre's
code lists (metsätietostandardi), which is what a forest owner's management
plan uses too.
"""

FINNISH = {
    # --- the page ------------------------------------------------------------
    "Mushroom map": "Sienikartta",
    "Choose a mushroom": "Valitse sieni",
    "Language": "Kieli",
    "Hide legend": "Piilota selite",
    "Excellent": "Erinomainen",
    "High": "Korkea",
    "Moderate": "Kohtalainen",
    "Reported find (laji.fi)": "Ilmoitettu löytö (laji.fi)",
    "An estimate from each forest stand's ecological attributes (site type, trees, soil) "
    "- not a record of where mushrooms have actually grown.":
        "Arvio perustuu kunkin metsäkuvion ominaisuuksiin (kasvupaikka, puusto, maaperä), "
        "ei tietoon siitä, missä sieniä on todella kasvanut.",
    "Site type": "Kasvupaikka",
    "Development class": "Kehitysluokka",
    "Dominant tree": "Pääpuulaji",
    "Other": "Muu",
    "Tree mix": "Puulajisekoitus",
    "Soil": "Maaperä",
    "Location": "Sijainti",
    "Effect on score: good": "Vaikutus pisteisiin: hyvä",
    "Effect on score: mid": "Vaikutus pisteisiin: kohtalainen",
    "Effect on score: poor": "Vaikutus pisteisiin: heikko",
    "Navigate here · Google Maps": "Reittiohjeet tänne · Google Maps",
    "Reported to laji.fi": "Ilmoitettu laji.fi:hin",
    "date unknown": "päivämäärä ei tiedossa",
    "You are here": "Olet tässä",
    # MapLibre's own screen-reader labels
    "Map": "Kartta",
    "Map marker": "Karttamerkki",
    "Close popup": "Sulje",
    "Toggle attribution": "Näytä tai piilota lähteet",

    # --- species (species.py profiles) ---------------------------------------
    "Chanterelle": "Kantarelli",
    "Chanterelle probability": "Kantarellin todennäköisyys",
    "Chanterelle sighting": "Kantarellihavainto",
    "Fairly dry, well-lit heath forest on free-draining soil.":
        "Melko kuiva, valoisa kangasmetsä hyvin vettä läpäisevällä maalla.",
    "Funnel chanterelle": "Suppilovahvero",
    "Funnel chanterelle probability": "Suppilovahveron todennäköisyys",
    "Funnel chanterelle sighting": "Suppilovahverohavainto",
    "Mature, spruce-dominated heath forest; copes with damper and poorer ground than the chanterelle.":
        "Vanha, kuusivaltainen kangasmetsä; sietää kosteampaa ja karumpaa maata kuin kantarelli.",
    "Light": "Valoisuus",
    "Open, well-lit": "Väljä, valoisa",
    "Fairly dense": "Melko tiheä",
    "Dense, little light": "Tiheä, vähän valoa",
    "Very sparse trees": "Hyvin harva puusto",
    "Landform": "Maastonmuoto",
    "Near an esker or ice-marginal formation": "Lähellä harjua tai reunamuodostumaa",
    "Not near an esker": "Ei harjun lähellä",

    # --- site type (FERTILITY_LABELS) ----------------------------------------
    "Herb-rich forest": "Lehto",
    "Herb-rich heath forest (OMT)": "Lehtomainen kangas (OMT)",
    "Mesic heath forest (MT)": "Tuore kangas (MT)",
    "Sub-xeric heath forest (VT)": "Kuivahko kangas (VT)",
    "Xeric heath forest (CT)": "Kuiva kangas (CT)",
    "Barren heath forest": "Karukkokangas",
    "Rocky or sandy ground": "Kalliomaa tai hietikko",
    "Hilltop or fell forest": "Lakimetsä tai tunturi",

    # --- development class (DEVELOPMENT_LABELS) ------------------------------
    "Young thinning stand": "Nuori kasvatusmetsikkö",
    "Advanced thinning stand": "Varttunut kasvatusmetsikkö",
    "Mature, regeneration-ready stand": "Uudistuskypsä metsikkö",
    "Shelterwood stand": "Suojuspuumetsikkö",
    "Uneven-aged stand": "Eri-ikäisrakenteinen metsikkö",
    "Seed-tree stand": "Siemenpuumetsikkö",
    "Sapling stand with overstorey": "Ylispuustoinen taimikko",
    "Sapling stand": "Taimikko",

    # --- soil (SOIL_LABELS) --------------------------------------------------
    "Coarse mineral soil": "Karkea kangasmaa",
    "Coarse till": "Karkea moreeni",
    "Coarse sorted soil": "Karkea lajittunut maa",
    "Fine mineral soil": "Hienojakoinen kangasmaa",
    "Fine till": "Hienoainesmoreeni",
    "Fine sorted soil": "Hienojakoinen lajittunut maa",
    "Silty soil": "Silttimaa",
    "Clay": "Savimaa",
    "Stony coarse mineral soil": "Kivinen karkea kangasmaa",
    "Stony coarse till": "Kivinen karkea moreeni",
    "Stony coarse sorted soil": "Kivinen karkea lajittunut maa",
    "Stony fine mineral soil": "Kivinen hienojakoinen kangasmaa",
    "Bedrock or boulders": "Kallio tai kivikko",
    "Peat": "Turvemaa",
    "Sedge peat": "Saraturve",
    "Sphagnum peat": "Rahkaturve",
    "Humus soil": "Multamaa",
    "Mud soil": "Liejumaa",

    # --- drainage, in brackets after the soil (DRAINAGE_LABELS) --------------
    "undrained": "ojittamaton",
    "turning boggy": "soistunut",
    "ditched": "ojitettu",
    "undrained mire": "ojittamaton suo",
    "freshly ditched mire": "ojikko",
    "drained mire, changing": "muuttuma",
    "drained peatland forest": "turvekangas",

    # --- dominant tree (TREESPECIES_LABELS) ----------------------------------
    "Scots pine": "Mänty",
    "Norway spruce": "Kuusi",
    "Silver birch": "Rauduskoivu",
    "Downy birch": "Hieskoivu",
    "Aspen": "Haapa",
    "Grey alder": "Harmaaleppä",
    "Black alder": "Tervaleppä",
    "Broadleaf (unspecified)": "Muu lehtipuu",
    "Conifer (unspecified)": "Muu havupuu",

    # --- tree mix (MIXTURE_BANDS) --------------------------------------------
    "Well-mixed forest": "Monipuolinen sekametsä",
    "Somewhat mixed": "Jonkin verran sekapuuta",
    "Nearly a single species": "Lähes yhden puulajin metsä",

    # --- landform and slope (LANDFORM_BANDS, SLOPE_BANDS) --------------------
    "Hilltop or ridge": "Mäen laki tai harjanne",
    "Upper slope": "Rinteen yläosa",
    "Gentle rise": "Loiva kohouma",
    "Level with its surroundings": "Ympäristön tasolla",
    "Shallow dip": "Matala painanne",
    "Hollow": "Notko",
    "Deep hollow": "Syvä notko",
    "steep": "jyrkkä",
    "sloping": "kalteva",
    "gently sloping": "loivasti kalteva",
}
