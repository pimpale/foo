import random

markers = [
    ".",
    "o",
    "v",
    "^",
    "<",
    ">",
    "1",
    "2",
    "3",
    "4",
    "8",
    "p",
    "P",
    "*",
    "h",
    "H",
    "+",
    "x",
    "X",
    "D",
    "d",
    "|",
    "_",
    4,
    5,
    6,
    7,  # Existing markers
    # Additional integer markers
    0,
    1,
    2,
    3,
    8,
    9,
    10,
    11,
    # Marker names for tick marks and carets
    "tickleft",
    "tickright",
    "tickup",
    "tickdown",
    "caretleft",
    "caretright",
    "caretup",
    "caretdown",
    "caretleftbase",
    "caretrightbase",
    "caretupbase",
    "caretdownbase",
    "$A$",
    "$B$",
    "$C$",
    # Custom markers using tuples (regular polygons, stars, and asterisks)
    (5, 0, 0),  # Regular pentagon
    (6, 0, 0),  # Regular hexagon
    (7, 0, 0),  # Regular heptagon
    (8, 0, 0),  # Regular octagon
    (5, 1, 0),  # 5-pointed star
    (6, 1, 0),  # 6-pointed star
    (7, 1, 0),  # 7-pointed star
    (5, 2, 0),  # 5-pointed asterisk
    (6, 2, 0),  # 6-pointed asterisk
]

random.seed(1)
markers_scrambled = markers[:20].copy()
random.shuffle(markers_scrambled)