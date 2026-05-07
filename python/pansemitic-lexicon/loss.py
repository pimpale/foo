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

import math
from dataclasses import dataclass, replace
from typing import Callable, Self

# ── Phoneme feature tables ────────────────────────────────────
#
# Each base phoneme (no length/gemination, no pharyngealization modifier
# when stored separately) is assigned a feature vector.  Cost functions
# use feature distance to score mutations.
#
# Place  — 0 (labial) … 1 (glottal), monotone front-to-back.
# Manner — discrete enum, paired via _MANNER_MATRIX below.
# Voicing, pharyngealized, lateral, aspirated — boolean {0, 1}.

_MANNER_NASAL = 0
_MANNER_PLOSIVE = 1
_MANNER_AFFRICATE = 2
_MANNER_FRICATIVE = 3
_MANNER_TAP = 4
_MANNER_TRILL = 5
_MANNER_APPROXIMANT = 6

@dataclass(frozen=True)
class ConsonantFeatures:
    place: float
    manner: int
    voicing: int
    pharyng: int
    lateral: int
    aspirated: int
    rhotic: int
    long: int


def _elongate_in_place(features: dict) -> None:
    """Mutate a phoneme's features in place to represent gemination."""
    for k in list(features.keys()):
        if features[k].long == 0:
            features[k + "ː"] = replace(features[k], long=1)


_CONSONANT_FEATURES: dict[str, ConsonantFeatures] = {
    # plosives
    "p":   ConsonantFeatures(place=0.00, manner=_MANNER_PLOSIVE,     voicing=0, pharyng=0, lateral=0, aspirated=0, rhotic=0, long=0),
    "pʰ":  ConsonantFeatures(place=0.00, manner=_MANNER_PLOSIVE,     voicing=0, pharyng=0, lateral=0, aspirated=1, rhotic=0, long=0),
    "b":   ConsonantFeatures(place=0.00, manner=_MANNER_PLOSIVE,     voicing=1, pharyng=0, lateral=0, aspirated=0, rhotic=0, long=0),
    "t":   ConsonantFeatures(place=0.30, manner=_MANNER_PLOSIVE,     voicing=0, pharyng=0, lateral=0, aspirated=0, rhotic=0, long=0),
    "tʰ":  ConsonantFeatures(place=0.30, manner=_MANNER_PLOSIVE,     voicing=0, pharyng=0, lateral=0, aspirated=1, rhotic=0, long=0),
    "d":   ConsonantFeatures(place=0.30, manner=_MANNER_PLOSIVE,     voicing=1, pharyng=0, lateral=0, aspirated=0, rhotic=0, long=0),
    "k":   ConsonantFeatures(place=0.70, manner=_MANNER_PLOSIVE,     voicing=0, pharyng=0, lateral=0, aspirated=0, rhotic=0, long=0),
    "kʰ":  ConsonantFeatures(place=0.70, manner=_MANNER_PLOSIVE,     voicing=0, pharyng=0, lateral=0, aspirated=1, rhotic=0, long=0),
    "g":   ConsonantFeatures(place=0.70, manner=_MANNER_PLOSIVE,     voicing=1, pharyng=0, lateral=0, aspirated=0, rhotic=0, long=0),
    "ɡ":   ConsonantFeatures(place=0.70, manner=_MANNER_PLOSIVE,     voicing=1, pharyng=0, lateral=0, aspirated=0, rhotic=0, long=0),
    "q":   ConsonantFeatures(place=0.80, manner=_MANNER_PLOSIVE,     voicing=0, pharyng=0, lateral=0, aspirated=0, rhotic=0, long=0),
    "qˤ":  ConsonantFeatures(place=0.80, manner=_MANNER_PLOSIVE,     voicing=0, pharyng=1, lateral=0, aspirated=0, rhotic=0, long=0),
    "ʔ":   ConsonantFeatures(place=1.00, manner=_MANNER_PLOSIVE,     voicing=0, pharyng=0, lateral=0, aspirated=0, rhotic=0, long=0),
    "tˤ":  ConsonantFeatures(place=0.30, manner=_MANNER_PLOSIVE,     voicing=0, pharyng=1, lateral=0, aspirated=0, rhotic=0, long=0),
    "dˤ":  ConsonantFeatures(place=0.30, manner=_MANNER_PLOSIVE,     voicing=1, pharyng=1, lateral=0, aspirated=0, rhotic=0, long=0),
    # affricates
    "d͡ʒ":  ConsonantFeatures(place=0.45, manner=_MANNER_AFFRICATE,   voicing=1, pharyng=0, lateral=0, aspirated=0, rhotic=0, long=0),
    "t͡ʃ":  ConsonantFeatures(place=0.45, manner=_MANNER_AFFRICATE,   voicing=0, pharyng=0, lateral=0, aspirated=0, rhotic=0, long=0),
    "t͡s":  ConsonantFeatures(place=0.30, manner=_MANNER_AFFRICATE,   voicing=0, pharyng=0, lateral=0, aspirated=0, rhotic=0, long=0),
    "t͡sˤ": ConsonantFeatures(place=0.30, manner=_MANNER_AFFRICATE,   voicing=0, pharyng=1, lateral=0, aspirated=0, rhotic=0, long=0),
    "d͡z":  ConsonantFeatures(place=0.30, manner=_MANNER_AFFRICATE,   voicing=1, pharyng=0, lateral=0, aspirated=0, rhotic=0, long=0),
    # fricatives
    "f":   ConsonantFeatures(place=0.00, manner=_MANNER_FRICATIVE,   voicing=0, pharyng=0, lateral=0, aspirated=0, rhotic=0, long=0),
    "v":   ConsonantFeatures(place=0.00, manner=_MANNER_FRICATIVE,   voicing=1, pharyng=0, lateral=0, aspirated=0, rhotic=0, long=0),
    "β":   ConsonantFeatures(place=0.00, manner=_MANNER_FRICATIVE,   voicing=1, pharyng=0, lateral=0, aspirated=0, rhotic=0, long=0),
    "θ":   ConsonantFeatures(place=0.15, manner=_MANNER_FRICATIVE,   voicing=0, pharyng=0, lateral=0, aspirated=0, rhotic=0, long=0),
    "ð":   ConsonantFeatures(place=0.15, manner=_MANNER_FRICATIVE,   voicing=1, pharyng=0, lateral=0, aspirated=0, rhotic=0, long=0),
    "s":   ConsonantFeatures(place=0.30, manner=_MANNER_FRICATIVE,   voicing=0, pharyng=0, lateral=0, aspirated=0, rhotic=0, long=0),
    "z":   ConsonantFeatures(place=0.30, manner=_MANNER_FRICATIVE,   voicing=1, pharyng=0, lateral=0, aspirated=0, rhotic=0, long=0),
    "sˤ":  ConsonantFeatures(place=0.30, manner=_MANNER_FRICATIVE,   voicing=0, pharyng=1, lateral=0, aspirated=0, rhotic=0, long=0),
    "ðˤ":  ConsonantFeatures(place=0.15, manner=_MANNER_FRICATIVE,   voicing=1, pharyng=1, lateral=0, aspirated=0, rhotic=0, long=0),
    "θˤ":  ConsonantFeatures(place=0.15, manner=_MANNER_FRICATIVE,   voicing=0, pharyng=1, lateral=0, aspirated=0, rhotic=0, long=0),
    "ʃ":   ConsonantFeatures(place=0.45, manner=_MANNER_FRICATIVE,   voicing=0, pharyng=0, lateral=0, aspirated=0, rhotic=0, long=0),
    "ʒ":   ConsonantFeatures(place=0.45, manner=_MANNER_FRICATIVE,   voicing=1, pharyng=0, lateral=0, aspirated=0, rhotic=0, long=0),
    "x":   ConsonantFeatures(place=0.70, manner=_MANNER_FRICATIVE,   voicing=0, pharyng=0, lateral=0, aspirated=0, rhotic=0, long=0),
    "ɣ":   ConsonantFeatures(place=0.70, manner=_MANNER_FRICATIVE,   voicing=1, pharyng=0, lateral=0, aspirated=0, rhotic=0, long=0),
    "χ":   ConsonantFeatures(place=0.80, manner=_MANNER_FRICATIVE,   voicing=0, pharyng=0, lateral=0, aspirated=0, rhotic=0, long=0),
    "ħ":   ConsonantFeatures(place=0.90, manner=_MANNER_FRICATIVE,   voicing=0, pharyng=0, lateral=0, aspirated=0, rhotic=0, long=0),
    "ʕ":   ConsonantFeatures(place=0.90, manner=_MANNER_FRICATIVE,   voicing=1, pharyng=0, lateral=0, aspirated=0, rhotic=0, long=0),
    "h":   ConsonantFeatures(place=1.00, manner=_MANNER_FRICATIVE,   voicing=0, pharyng=0, lateral=0, aspirated=0, rhotic=0, long=0),
    # nasals
    "m":   ConsonantFeatures(place=0.00, manner=_MANNER_NASAL,       voicing=1, pharyng=0, lateral=0, aspirated=0, rhotic=0, long=0),
    "n":   ConsonantFeatures(place=0.30, manner=_MANNER_NASAL,       voicing=1, pharyng=0, lateral=0, aspirated=0, rhotic=0, long=0),
    # laterals
    "l":   ConsonantFeatures(place=0.30, manner=_MANNER_APPROXIMANT, voicing=1, pharyng=0, lateral=1, aspirated=0, rhotic=0, long=0),
    "lˤ":  ConsonantFeatures(place=0.30, manner=_MANNER_APPROXIMANT, voicing=1, pharyng=1, lateral=1, aspirated=0, rhotic=0, long=0),
    "ɫ":   ConsonantFeatures(place=0.30, manner=_MANNER_APPROXIMANT, voicing=1, pharyng=1, lateral=1, aspirated=0, rhotic=0, long=0),
    "ɬ":   ConsonantFeatures(place=0.30, manner=_MANNER_FRICATIVE,   voicing=0, pharyng=0, lateral=1, aspirated=0, rhotic=0, long=0),
    "ɬˤ":  ConsonantFeatures(place=0.30, manner=_MANNER_FRICATIVE,   voicing=0, pharyng=1, lateral=1, aspirated=0, rhotic=0, long=0),
    # rhotics
    "r":   ConsonantFeatures(place=0.30, manner=_MANNER_TRILL,       voicing=1, pharyng=0, lateral=0, aspirated=0, rhotic=1, long=0),
    "rˤ":  ConsonantFeatures(place=0.30, manner=_MANNER_TRILL,       voicing=1, pharyng=1, lateral=0, aspirated=0, rhotic=1, long=0),
    "ɾ":   ConsonantFeatures(place=0.30, manner=_MANNER_TAP,         voicing=1, pharyng=0, lateral=0, aspirated=0, rhotic=1, long=0),
    "ɹ":   ConsonantFeatures(place=0.30, manner=_MANNER_APPROXIMANT, voicing=1, pharyng=0, lateral=0, aspirated=0, rhotic=1, long=0),
    "ʀ":   ConsonantFeatures(place=0.80, manner=_MANNER_TRILL,       voicing=1, pharyng=0, lateral=0, aspirated=0, rhotic=1, long=0),
    "ʁ":   ConsonantFeatures(place=0.80, manner=_MANNER_FRICATIVE,   voicing=1, pharyng=0, lateral=0, aspirated=0, rhotic=1, long=0),
    # glides / approximants
    "w":   ConsonantFeatures(place=0.00, manner=_MANNER_APPROXIMANT, voicing=1, pharyng=0, lateral=0, aspirated=0, rhotic=0, long=0),
    "j":   ConsonantFeatures(place=0.55, manner=_MANNER_APPROXIMANT, voicing=1, pharyng=0, lateral=0, aspirated=0, rhotic=0, long=0),
}

_elongate_in_place(_CONSONANT_FEATURES)

# Manner-to-manner cost matrix.  Indexed by the _MANNER_* constants.
# Tuned to roughly track perceptual / diachronic distance for Semitic
# (e.g. plosive↔fricative cheap because of begadkefat lenition;
# tap↔trill cheap because allophonic).
#
# Only the upper triangle is filled; _symmetrize_in_place mirrors it
# below the diagonal so each pair has one source of truth to tweak.
_MANNER_MATRIX: list[list[float]] = [
    # nas  plo  aff  fri  tap  tri  apx
    [0.0, 0.6, 0.8, 0.7, 1.0, 1.0, 0.7],  # nasal
    [0.0, 0.0, 0.3, 0.4, 0.9, 0.9, 0.7],  # plosive
    [0.0, 0.0, 0.0, 0.3, 0.9, 0.9, 0.8],  # affricate
    [0.0, 0.0, 0.0, 0.0, 0.7, 0.7, 0.5],  # fricative
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.4],  # tap
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],  # trill
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # approximant
]


def _symmetrize_in_place(matrix: list[list[float]]) -> None:
    n = len(matrix)
    for i in range(n):
        for j in range(i):
            matrix[i][j] = matrix[j][i]


_symmetrize_in_place(_MANNER_MATRIX)


# Vowel features: height (high→low = 0→1), backness (front→back = 0→1), rounding.
@dataclass(frozen=True)
class VowelFeatures:
    height: float
    backness: float
    rounded: int
    long: int


_VOWEL_FEATURES: dict[str, VowelFeatures] = {
    "a": VowelFeatures(height=1.00, backness=0.50, rounded=0, long=0),
    "e": VowelFeatures(height=0.50, backness=0.00, rounded=0, long=0),
    "ɛ": VowelFeatures(height=0.70, backness=0.00, rounded=0, long=0),
    "i": VowelFeatures(height=0.00, backness=0.00, rounded=0, long=0),
    "y": VowelFeatures(height=0.00, backness=0.00, rounded=1, long=0),
    "o": VowelFeatures(height=0.50, backness=1.00, rounded=1, long=0),
    "ɔ": VowelFeatures(height=0.70, backness=1.00, rounded=1, long=0),
    "u": VowelFeatures(height=0.00, backness=1.00, rounded=1, long=0),
    "æ": VowelFeatures(height=0.90, backness=0.00, rounded=0, long=0),
    "ɒ": VowelFeatures(height=1.00, backness=1.00, rounded=1, long=0),
    "ə": VowelFeatures(height=0.50, backness=0.50, rounded=0, long=0),
}

_elongate_in_place(_VOWEL_FEATURES)

VOWELS = set(_VOWEL_FEATURES.keys())
IPA_MODIFIERS = set("ːˤʰʲʷˠʼ̃")
IPA_CONSONANT_DIGRAPHS = (
    "d͡ʒ", "t͡ʃ", "t͡s", "d͡z",
    "t͡ɬ", "d͡ɮ", "k͡x", "ɡ͡ɣ",
)



# ── Operation weights ─────────────────────────────────────────
# All features contribute to mutation cost via weighted Manhattan distance.
# Tune these and re-run to see how the lexicon-wide loss shifts.

# Consonant feature weights.  Manner has no weight knob: its cost lives
# entirely inside _MANNER_MATRIX.
W_PLACE = 1.0
W_VOICING = 0.5
W_PHARYNG = 0.5
W_LATERAL = 0.6
W_ASPIRATED = 0.2

# Vowel feature weights
W_HEIGHT = 0.4
W_BACKNESS = 0.35
W_ROUNDED = 0.15

# Length / gemination — the same physical feature (ː).  A single op in the
# user's list: "gemination / degemination" for consonants, subsumed into
# vowel mutation for vowels.
W_LENGTH_CONS = 0.3     # C vs Cː
W_LENGTH_VOWEL = 0.3    # V vs Vː

# Caps on mutation cost so distant pairs don't dominate the sum.
C_CONS_MUT_MAX = 3.0
C_VOWEL_MUT_MAX = 2.0
# Rhotic-to-rhotic alternation (e.g. alveolar r ↔ uvular ʀ/ʁ) is common
# allophonic variation, so cap their mutation cost more aggressively.
C_RHOTIC_ALLOPHONE_MAX = 0.3

# Insert/delete — vowels cheap, consonants high (option b from plan).
C_VOWEL_INSDEL = 0.8
C_CONS_INSDEL = 1.5
C_CHEAP_CONS_INSDEL = 0.75   # for a small set of "cheap" consonants (e.g. glottal stop, pharyngeals)

# Cross-type (consonant ↔ vowel): should never happen in a sensible
# alignment; guard with a very high cost.
C_CROSS_TYPE = 5.0

# Metathesis (adjacent-phoneme swap, e.g. Hebrew hitpaʕel t-infix).
# Strict form: only fires for exact swaps `(X, Y) ↔ (Y, X)`.
C_METATHESIS = 0.8


# ── Phoneme classes ──────────────────────────────────────────

@dataclass(frozen=True)
class Phoneme:
    """An IPA phoneme token, parsed into a base symbol plus length."""
    tok: str

    @staticmethod
    def of_token(tok: str) -> "Phoneme":
        if tok in _VOWEL_FEATURES:
            return Vowel(tok)
        if tok in _CONSONANT_FEATURES:
            return Consonant(tok)
        raise ValueError(f"Unknown phoneme: {tok!r}")
    
    @staticmethod
    def parse(ipa: str) -> "list[Phoneme]":
        """Split an IPA string into phoneme tokens.

        A token is a base (single letter or tie-bar digraph like d͡ʒ) plus any
        trailing modifier letters (ˤ, ː).  Non-letter / non-modifier characters
        (whitespace, punctuation, reconstruction markers) are skipped.
        """
        phonemes: list[Phoneme] = []
        i = 0
        n = len(ipa)
        while i < n:
            c = ipa[i]
            if not c.isalpha() and c not in "ʔʕ":
                i += 1
                continue
            # Tie-bar digraph: base + U+0361 + base
            if i + 2 < n and ipa[i + 1] == '͡':
                two = ipa[i] + ipa[i + 1] + ipa[i + 2]
                if two in IPA_CONSONANT_DIGRAPHS:
                    phon = two
                    i += 3
                    while i < n and ipa[i] in IPA_MODIFIERS:
                        phon += ipa[i]
                        i += 1
                    phonemes.append(Phoneme.of_token(phon))
                    continue
            phon = c
            i += 1
            while i < n and ipa[i] in IPA_MODIFIERS:
                phon += ipa[i]
                i += 1
            
            phonemes.append(Phoneme.of_token(phon))
        return phonemes

    def cost(self, other: "Phoneme") -> float:
        raise NotImplementedError

    def insdel_cost(self) -> float:
        raise NotImplementedError


@dataclass(frozen=True)
class Consonant(Phoneme):
    def __post_init__(self) -> None:
        if self.tok not in _CONSONANT_FEATURES:
            raise ValueError(f"Unknown consonant: {self.tok!r}")

    def features(self) -> ConsonantFeatures:
        return _CONSONANT_FEATURES[self.tok]

    def insdel_cost(self) -> float:
        if self.features().place == 1.0:
            cost = C_CHEAP_CONS_INSDEL
            if self.features().long:
                cost += W_LENGTH_CONS
            return cost 
        else:
            return C_CONS_INSDEL

    def cost(self, other: Phoneme) -> float:
        if not isinstance(other, Consonant):
            return C_CROSS_TYPE
        af = self.features()
        bf = other.features()
        d = (
            W_PLACE * abs(af.place - bf.place)
            + _MANNER_MATRIX[af.manner][bf.manner]
            + W_VOICING * abs(af.voicing - bf.voicing)
            + W_PHARYNG * abs(af.pharyng - bf.pharyng)
            + W_LATERAL * abs(af.lateral - bf.lateral)
            + W_ASPIRATED * abs(af.aspirated - bf.aspirated)
            + W_LENGTH_CONS * abs(af.long - bf.long)
        )
        if af.rhotic and bf.rhotic:
            d = min(d, C_RHOTIC_ALLOPHONE_MAX)
        return min(d, C_CONS_MUT_MAX)


@dataclass(frozen=True)
class Vowel(Phoneme):
    def __post_init__(self) -> None:
        if self.tok not in _VOWEL_FEATURES:
            raise ValueError(f"Unknown vowel: {self.tok!r}")

    def features(self) -> VowelFeatures:
        return _VOWEL_FEATURES[self.tok]

    def insdel_cost(self) -> float:
        return C_VOWEL_INSDEL

    def cost(self, other: Phoneme) -> float:
        if not isinstance(other, Vowel):
            return C_CROSS_TYPE
        af = self.features()
        bf = other.features()
        d = (
            W_HEIGHT * abs(af.height - bf.height)
            + W_BACKNESS * abs(af.backness - bf.backness)
            + W_ROUNDED * abs(af.rounded - bf.rounded)
            + W_LENGTH_VOWEL * abs(af.long - bf.long)
        )
        return min(d, C_VOWEL_MUT_MAX)


# ── Alignment ────────────────────────────────────────────────

# A rule's apply function takes the windows it would consume from each side
# and returns the cost of applying it, or None if the rule doesn't match.
RuleApply = Callable[[tuple[Phoneme, ...], tuple[Phoneme, ...]], "float | None"]


@dataclass(frozen=True)
class Rule:
    """An edit operation.

    `consume_a` / `consume_b` are how many phonemes the rule consumes from
    each sequence.  `apply` is called with the two windows and returns the
    cost of applying the rule, or None if it doesn't match those windows.
    """
    name: str
    consume_a: int
    consume_b: int
    apply: RuleApply


def _delete_apply(a_win: tuple[Phoneme, ...], _b_win: tuple[Phoneme, ...]) -> float:
    return a_win[0].insdel_cost()


def _insert_apply(_a_win: tuple[Phoneme, ...], b_win: tuple[Phoneme, ...]) -> float:
    return b_win[0].insdel_cost()


def _substitute_apply(a_win: tuple[Phoneme, ...], b_win: tuple[Phoneme, ...]) -> float:
    return a_win[0].cost(b_win[0])


def _metathesis_apply(a_win: tuple[Phoneme, ...], b_win: tuple[Phoneme, ...]) -> float | None:
    if len(a_win) != 2 or len(b_win) != 2:
        return None
    if a_win[0] == b_win[1] and a_win[1] == b_win[0] and a_win[0] != a_win[1]:
        return C_METATHESIS
    return None




# Default rule list.  Order matters for tie-breaking: the first rule whose
# total cost equals the running minimum at a cell wins.  Append custom
# rules (e.g. diphthong→glide as a 2:1 rule) — earlier entries win ties.
RULES: list[Rule] = [
    Rule(name="delete", consume_a=1, consume_b=0, apply=_delete_apply),
    Rule(name="insert", consume_a=0, consume_b=1, apply=_insert_apply),
    Rule(name="substitute", consume_a=1, consume_b=1, apply=_substitute_apply),
    Rule(name="metathesis", consume_a=2, consume_b=2, apply=_metathesis_apply),
]


@dataclass(frozen=True)
class SelectedRule:
    """A rule applied at a specific point in the alignment, with its cost."""
    rule: Rule
    a_phonemes: tuple[Phoneme, ...]
    b_phonemes: tuple[Phoneme, ...]
    cost: float


def trace_cost(
    a: list[Phoneme],
    b: list[Phoneme],
    rules: list[Rule] | None = None,
) -> list[SelectedRule]:
    """Optimal edit script aligning `a` to `b`, as a list of applied rules.

    DP iterates every rule at every cell: for each rule, looks back
    `(consume_a, consume_b)` cells and adds the rule's cost.  Earlier
    rules in the list win cost ties.  Boundary handling falls out of the
    rules themselves — at `(0, j)` only INSERT fits, at `(i, 0)` only
    DELETE fits.
    """
    if rules is None:
        rules = RULES
    if not any(r.consume_a > 0 or r.consume_b > 0 for r in rules):
        raise ValueError("Rule set must contain at least one progressing rule")

    n, m = len(a), len(b)
    INF = math.inf
    dp: list[list[float]] = [[INF] * (m + 1) for _ in range(n + 1)]
    back: list[list[Rule | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0

    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            for rule in rules:
                if rule.consume_a > i or rule.consume_b > j:
                    continue
                pi, pj = i - rule.consume_a, j - rule.consume_b
                prev = dp[pi][pj]
                if prev == INF:
                    continue
                a_win = tuple(a[pi:i])
                b_win = tuple(b[pj:j])
                c = rule.apply(a_win, b_win)
                if c is None:
                    continue
                total = prev + c
                if total < dp[i][j]:
                    dp[i][j] = total
                    back[i][j] = rule

    if dp[n][m] == INF:
        raise ValueError(
            "No alignment found; rule set cannot bridge these sequences"
        )

    trace: list[SelectedRule] = []
    i, j = n, m
    while i > 0 or j > 0:
        rule = back[i][j]
        assert rule is not None
        pi, pj = i - rule.consume_a, j - rule.consume_b
        a_win = tuple(a[pi:i])
        b_win = tuple(b[pj:j])
        c = rule.apply(a_win, b_win)
        assert c is not None
        trace.append(SelectedRule(rule, a_win, b_win, c))
        i, j = pi, pj
    trace.reverse()
    return trace


def align_cost(a: list[Phoneme], b: list[Phoneme]) -> float:
    """Weighted Levenshtein distance between two phoneme sequences."""
    return sum(sr.cost for sr in trace_cost(a, b))


def ipa_distance_with_trace(a: str, b: str) -> tuple[float, list[SelectedRule]]:
    """Same as `ipa_distance`, but also returns the optimal edit script."""
    at = Phoneme.parse(a)
    bt = Phoneme.parse(b)
    if not at and not bt:
        return 0.0, []
    trace = trace_cost(at, bt)
    raw = sum(sr.cost for sr in trace)
    return raw / max(len(at), len(bt)), trace


def ipa_distance(a: str, b: str) -> float:
    """Weighted edit distance between two IPA strings.  Normalized by the
    longer token sequence so long-vs-short comparisons stay on a shared
    scale."""
    return ipa_distance_with_trace(a, b)[0]


# ── Triplet loss ──────────────────────────────────────────────

@dataclass(frozen=True)
class LossBreakdown:
    arabic: float
    hebrew: float
    joint: float


def triplet_loss_breakdown(
    pansemitic_ipa: str,
    ar_ipa: str,
    he_ipa: str,
) -> LossBreakdown:
    """Return per-language and joint comprehension cost for a triplet."""
    arabic = ipa_distance(pansemitic_ipa, ar_ipa)
    hebrew = ipa_distance(pansemitic_ipa, he_ipa)
    return LossBreakdown(
        arabic=arabic,
        hebrew=hebrew,
        joint=arabic + hebrew,
    )

def triplet_loss(pansemitic_ipa: str, ar_ipa: str, he_ipa: str) -> float:
    """Pansemitic-form comprehension cost, summed over both speakers.

    Returns `distance(pan, ar) + distance(pan, he)`.  Each component is
    length-normalized; the sum sits in roughly [0, 4] with 0 = perfect
    match on both sides.
    """
    return triplet_loss_breakdown(pansemitic_ipa, ar_ipa, he_ipa).joint


# ── Scholar-pansemitic → IPA (inverse of _PANSEMITIC_IPA_TO_SCHOLAR) ──
# The CSV stores pansemitic in scholar notation.  To compute loss from a
# CSV row we need to push it back to IPA.  Order: j → d͡ʒ before y → j, so
# the IPA j we produce from scholar y isn't re-tagged as jīm.
_PAN_SCHOLAR_TO_IPA: list[tuple[str, str]] = [
    ("ṣ", "sˤ"),
    ("ṭ", "tˤ"),
    ("š", "ʃ"),
    ("ž", "ʒ"),
    ("j", "d͡ʒ"),
    ("y", "j"),
]


def pansemitic_scholar_to_ipa(form: str) -> str:
    out = form
    for src, dst in _PAN_SCHOLAR_TO_IPA:
        out = out.replace(src, dst)
    return out


# ── CLI driver ────────────────────────────────────────────────

def _serialize_step(sr: SelectedRule) -> dict:
    return {
        "rule": sr.rule.name,
        "a": [p.tok for p in sr.a_phonemes],
        "b": [p.tok for p in sr.b_phonemes],
        "cost": round(sr.cost, 4),
    }


def _main() -> None:
    import json
    import statistics
    from pathlib import Path

    in_path = Path("cognates2.json")
    out_path = Path("loss.json")
    losses: list[float] = []
    entries: list[dict] = []
    skipped = 0
    with open(in_path, encoding="utf-8") as f:
        cognates = json.load(f)
    for row in cognates:
        ar = row.get("arabic") or {}
        he = row.get("hebrew") or {}
        ar_ipa = ar.get("ipa") or ""
        he_ipa = he.get("ipa") or ""
        pan_scholar = row.get("pansemitic_form") or ""
        if not (ar_ipa and he_ipa and pan_scholar):
            skipped += 1
            continue
        pan_ipa = pansemitic_scholar_to_ipa(pan_scholar)
        ar_dist, ar_trace = ipa_distance_with_trace(pan_ipa, ar_ipa)
        he_dist, he_trace = ipa_distance_with_trace(pan_ipa, he_ipa)
        joint = ar_dist + he_dist
        losses.append(joint)
        entries.append({
            "arabic": ar.get("canonical") or "",
            "arabic_romanization": ar.get("roman") or "",
            "arabic_ipa": ar_ipa,
            "hebrew": he.get("canonical") or "",
            "hebrew_romanization": he.get("roman") or "",
            "hebrew_ipa": he_ipa,
            "pansemitic": pan_scholar,
            "pansemitic_ipa": pan_ipa,
            "arabic_distance": round(ar_dist, 4),
            "hebrew_distance": round(he_dist, 4),
            "joint": round(joint, 4),
            "arabic_trace": [_serialize_step(s) for s in ar_trace],
            "hebrew_trace": [_serialize_step(s) for s in he_trace],
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(entries)} entries to {out_path}")

    if not losses:
        print("No scorable triplets.")
        return

    print(f"Scored {len(losses)} triplets ({skipped} skipped for missing fields)")
    print(f"  mean   {statistics.mean(losses):.4f}")
    print(f"  median {statistics.median(losses):.4f}")
    print(f"  stdev  {statistics.stdev(losses):.4f}")
    print(f"  min    {min(losses):.4f}")
    print(f"  max    {max(losses):.4f}")
    # Quick histogram bucket
    buckets = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
    counts = [0] * (len(buckets) - 1)
    for L in losses:
        for i in range(len(buckets) - 1):
            if buckets[i] <= L < buckets[i + 1]:
                counts[i] += 1
                break
    print("  distribution:")
    for i, c in enumerate(counts):
        print(f"    [{buckets[i]:.2f}, {buckets[i+1]:.2f})  {c}")


if __name__ == "__main__":
    _main()
