#!/usr/bin/env python3
"""Loss function for pansemitic triplets.

Measures how hard a pansemitic form is to comprehend for an Arabic or Hebrew
speaker.  The score is a weighted Levenshtein distance over IPA phoneme
sequences with a small set of linguistically-motivated edit operations:
vowel ins/del, vowel mutation, consonant ins/del, consonant mutation,
gemination/degemination.  Pharyngealization is a feature of consonant
mutation rather than its own op.
"""
from __future__ import annotations


# ── Phoneme feature tables ────────────────────────────────────
#
# Each base phoneme (no length/gemination, no pharyngealization modifier
# when stored separately) is assigned a feature vector with components in
# [0, 1].  Cost functions use feature distance to score mutations.
#
# Place  — 0 (labial) … 1 (glottal), monotone front-to-back.
# Manner — 0 (stop) … 1 (approximant), monotone on sonority/openness.
# Voicing, pharyngealized, lateral — boolean {0, 1}.

_CONSONANT_FEATURES: dict[str, dict[str, float]] = {
    # stops
    "p":  {"place": 0.00, "manner": 0.00, "voicing": 0, "pharyng": 0, "lateral": 0},
    "b":  {"place": 0.00, "manner": 0.00, "voicing": 1, "pharyng": 0, "lateral": 0},
    "t":  {"place": 0.30, "manner": 0.00, "voicing": 0, "pharyng": 0, "lateral": 0},
    "d":  {"place": 0.30, "manner": 0.00, "voicing": 1, "pharyng": 0, "lateral": 0},
    "k":  {"place": 0.70, "manner": 0.00, "voicing": 0, "pharyng": 0, "lateral": 0},
    "g":  {"place": 0.70, "manner": 0.00, "voicing": 1, "pharyng": 0, "lateral": 0},
    "q":  {"place": 0.80, "manner": 0.00, "voicing": 0, "pharyng": 0, "lateral": 0},
    "ʔ":  {"place": 1.00, "manner": 0.00, "voicing": 0, "pharyng": 0, "lateral": 0},
    "tˤ": {"place": 0.30, "manner": 0.00, "voicing": 0, "pharyng": 1, "lateral": 0},
    "dˤ": {"place": 0.30, "manner": 0.00, "voicing": 1, "pharyng": 1, "lateral": 0},
    # affricates
    "d͡ʒ": {"place": 0.45, "manner": 0.15, "voicing": 1, "pharyng": 0, "lateral": 0},
    "t͡ʃ": {"place": 0.45, "manner": 0.15, "voicing": 0, "pharyng": 0, "lateral": 0},
    "t͡s": {"place": 0.30, "manner": 0.15, "voicing": 0, "pharyng": 0, "lateral": 0},
    "d͡z": {"place": 0.30, "manner": 0.15, "voicing": 1, "pharyng": 0, "lateral": 0},
    # fricatives
    "f":  {"place": 0.00, "manner": 0.35, "voicing": 0, "pharyng": 0, "lateral": 0},
    "v":  {"place": 0.00, "manner": 0.35, "voicing": 1, "pharyng": 0, "lateral": 0},
    "θ":  {"place": 0.15, "manner": 0.35, "voicing": 0, "pharyng": 0, "lateral": 0},
    "ð":  {"place": 0.15, "manner": 0.35, "voicing": 1, "pharyng": 0, "lateral": 0},
    "s":  {"place": 0.30, "manner": 0.35, "voicing": 0, "pharyng": 0, "lateral": 0},
    "z":  {"place": 0.30, "manner": 0.35, "voicing": 1, "pharyng": 0, "lateral": 0},
    "sˤ": {"place": 0.30, "manner": 0.35, "voicing": 0, "pharyng": 1, "lateral": 0},
    "ðˤ": {"place": 0.15, "manner": 0.35, "voicing": 1, "pharyng": 1, "lateral": 0},
    "θˤ": {"place": 0.15, "manner": 0.35, "voicing": 0, "pharyng": 1, "lateral": 0},
    "ʃ":  {"place": 0.45, "manner": 0.35, "voicing": 0, "pharyng": 0, "lateral": 0},
    "ʒ":  {"place": 0.45, "manner": 0.35, "voicing": 1, "pharyng": 0, "lateral": 0},
    "x":  {"place": 0.70, "manner": 0.35, "voicing": 0, "pharyng": 0, "lateral": 0},
    "ɣ":  {"place": 0.70, "manner": 0.35, "voicing": 1, "pharyng": 0, "lateral": 0},
    "χ":  {"place": 0.80, "manner": 0.35, "voicing": 0, "pharyng": 0, "lateral": 0},
    "ħ":  {"place": 0.90, "manner": 0.35, "voicing": 0, "pharyng": 0, "lateral": 0},
    "ʕ":  {"place": 0.90, "manner": 0.35, "voicing": 1, "pharyng": 0, "lateral": 0},
    "h":  {"place": 1.00, "manner": 0.35, "voicing": 0, "pharyng": 0, "lateral": 0},
    "ɬ":  {"place": 0.30, "manner": 0.35, "voicing": 0, "pharyng": 0, "lateral": 1},
    "ɬˤ": {"place": 0.30, "manner": 0.35, "voicing": 0, "pharyng": 1, "lateral": 1},
    # nasals
    "m":  {"place": 0.00, "manner": 0.55, "voicing": 1, "pharyng": 0, "lateral": 0},
    "n":  {"place": 0.30, "manner": 0.55, "voicing": 1, "pharyng": 0, "lateral": 0},
    # liquids
    "l":  {"place": 0.30, "manner": 0.70, "voicing": 1, "pharyng": 0, "lateral": 1},
    "r":  {"place": 0.30, "manner": 0.85, "voicing": 1, "pharyng": 0, "lateral": 0},
    # approximants / glides
    "w":  {"place": 0.00, "manner": 1.00, "voicing": 1, "pharyng": 0, "lateral": 0},
    "j":  {"place": 0.55, "manner": 1.00, "voicing": 1, "pharyng": 0, "lateral": 0},
}

# Vowel features: height (high→low = 0→1), backness (front→back = 0→1), rounding.
_VOWEL_FEATURES: dict[str, dict[str, float]] = {
    "a": {"height": 1.00, "backness": 0.50, "rounded": 0},
    "e": {"height": 0.50, "backness": 0.00, "rounded": 0},
    "i": {"height": 0.00, "backness": 0.00, "rounded": 0},
    "o": {"height": 0.50, "backness": 1.00, "rounded": 1},
    "u": {"height": 0.00, "backness": 1.00, "rounded": 1},
}
