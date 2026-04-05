#!/usr/bin/env python3
"""
Phonetic reconstruction and pansemitic form generation.
"""
import re
from reconstruction import Word

# ── Pansemitic form ───────────
_PROTO_TO_PANSEMITIC = [
    # Glottals and pharyngeals
    ("ɣ", "g"),
    ("ḥ", "x"),
    ("ḫ", "x"),
    # Interdentals
    ("ḏ", "z"),
    ("ṯ", "s"),
    ("ṯ̣", "ṣ"),
    # Laterals
    ("l", "l"),
    ("ś", "s"),
    ("ṣ́", "ṣ"),
    # emphatics
    ("ṭ", "ṭ"),
    ("q", "q"),
]



def construct_pansemitic_form(ancestor: Word):
    """Convert an ancestor Word into a pansemitic intermediate.

    Applies sound changes to produce a form with Arabic's 3-vowel system
    and Hebrew's consonant inventory.
    """
    if not ancestor or not ancestor.word:
        return None

    lang = ancestor.lang
    form = ancestor.word.lower()

    # Strip common affixes/decorations
    form = form.lstrip("*").rstrip("-")

    # For Semitic sources where f < p, ensure f→p
    if lang in ("sem-pro", "ar", "akk", "arc"):
        form = form.replace("f", "p")

    # Apply consonant compression (proto → pansemitic)
    for old, new in _PROTO_TO_PANSEMITIC:
        form = form.replace(old, new)

    # Normalize vowels: strip length, collapse e→i, o→a
    form = (form
            .replace("ā", "a").replace("ī", "i").replace("ū", "u")
            .replace("á", "a").replace("í", "i").replace("ú", "u")
            .replace("é", "e").replace("ó", "o")
            .replace("e", "i").replace("o", "a"))

    # Remove doubled vowels
    form = re.sub(r"([aiu])\1+", r"\1", form)
    form = form.strip()

    return form
