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

from dataclasses import dataclass

from reconstruction import _tokenize_phonemes


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


# ── Operation weights ─────────────────────────────────────────
# All features contribute to mutation cost via weighted Manhattan distance.
# Tune these and re-run to see how the lexicon-wide loss shifts.

# Consonant feature weights
W_PLACE = 1.0
W_MANNER = 1.0
W_VOICING = 0.5
W_PHARYNG = 0.8
W_LATERAL = 0.6

# Vowel feature weights
W_HEIGHT = 1.0
W_BACKNESS = 0.8
W_ROUNDED = 0.4

# Length / gemination — the same physical feature (ː).  A single op in the
# user's list: "gemination / degemination" for consonants, subsumed into
# vowel mutation for vowels.
W_LENGTH_CONS = 0.3     # C vs Cː
W_LENGTH_VOWEL = 0.3    # V vs Vː

# Caps on mutation cost so distant pairs don't dominate the sum.
C_CONS_MUT_MAX = 1.5
C_VOWEL_MUT_MAX = 0.8

# Insert/delete — vowels cheap, consonants high (option b from plan).
C_VOWEL_INSDEL = 0.5
C_CONS_INSDEL = 1.5

# Cross-type (consonant ↔ vowel): should never happen in a sensible
# alignment; guard with a very high cost.
C_CROSS_TYPE = 2.0


# ── Cost functions ────────────────────────────────────────────

def _parse_token(tok: str) -> tuple[str, bool]:
    """Split a phoneme token into (base, is_long_or_geminate)."""
    if tok.endswith("ː"):
        return tok[:-1], True
    return tok, False


def _is_vowel_base(base: str) -> bool:
    return base in _VOWEL_FEATURES


def _vowel_cost(a: str, b: str, a_long: bool, b_long: bool) -> float:
    af = _VOWEL_FEATURES.get(a)
    bf = _VOWEL_FEATURES.get(b)
    if af is None or bf is None:
        # Unknown vowel (shouldn't happen after pansemitic/AR/HE normalization):
        # treat as max mutation unless bases literally match.
        base_match = a == b
        length_cost = W_LENGTH_VOWEL * (0 if a_long == b_long else 1)
        return length_cost if base_match else C_VOWEL_MUT_MAX
    d = (
        W_HEIGHT * abs(af["height"] - bf["height"])
        + W_BACKNESS * abs(af["backness"] - bf["backness"])
        + W_ROUNDED * abs(af["rounded"] - bf["rounded"])
        + W_LENGTH_VOWEL * (0 if a_long == b_long else 1)
    )
    return min(d, C_VOWEL_MUT_MAX)


def _cons_cost(a: str, b: str, a_long: bool, b_long: bool) -> float:
    af = _CONSONANT_FEATURES.get(a)
    bf = _CONSONANT_FEATURES.get(b)
    if af is None or bf is None:
        base_match = a == b
        length_cost = W_LENGTH_CONS * (0 if a_long == b_long else 1)
        return length_cost if base_match else C_CONS_MUT_MAX
    d = (
        W_PLACE * abs(af["place"] - bf["place"])
        + W_MANNER * abs(af["manner"] - bf["manner"])
        + W_VOICING * abs(af["voicing"] - bf["voicing"])
        + W_PHARYNG * abs(af["pharyng"] - bf["pharyng"])
        + W_LATERAL * abs(af["lateral"] - bf["lateral"])
        + W_LENGTH_CONS * (0 if a_long == b_long else 1)
    )
    return min(d, C_CONS_MUT_MAX)


def sub_cost(a: str, b: str) -> float:
    """Cost of substituting phoneme token `a` with `b`.  0 iff identical."""
    if a == b:
        return 0.0
    a_base, a_long = _parse_token(a)
    b_base, b_long = _parse_token(b)
    a_is_vowel = _is_vowel_base(a_base)
    b_is_vowel = _is_vowel_base(b_base)
    if a_is_vowel != b_is_vowel:
        return C_CROSS_TYPE
    if a_is_vowel:
        return _vowel_cost(a_base, b_base, a_long, b_long)
    return _cons_cost(a_base, b_base, a_long, b_long)


def insdel_cost(tok: str) -> float:
    """Cost of inserting or deleting phoneme token `tok`."""
    base, _ = _parse_token(tok)
    return C_VOWEL_INSDEL if _is_vowel_base(base) else C_CONS_INSDEL


# ── Alignment ────────────────────────────────────────────────

def align_cost(a_tokens: list[str], b_tokens: list[str]) -> float:
    """Weighted Levenshtein distance between two phoneme-token sequences.

    Standard three-op DP (insert from b, delete from a, substitute) with
    per-token costs from `sub_cost` / `insdel_cost`.  Gemination is folded
    into consonant substitution via the length feature — `Cː ↔ C` costs
    `W_LENGTH_CONS`, which is the gemination/degemination op.
    """
    n, m = len(a_tokens), len(b_tokens)
    if n == 0:
        return sum(insdel_cost(t) for t in b_tokens)
    if m == 0:
        return sum(insdel_cost(t) for t in a_tokens)

    # Row-by-row DP, keeping two rows to stay O(m) memory.
    prev = [0.0] * (m + 1)
    for j in range(1, m + 1):
        prev[j] = prev[j - 1] + insdel_cost(b_tokens[j - 1])

    curr = [0.0] * (m + 1)
    for i in range(1, n + 1):
        a_tok = a_tokens[i - 1]
        curr[0] = prev[0] + insdel_cost(a_tok)
        for j in range(1, m + 1):
            b_tok = b_tokens[j - 1]
            delete_a = prev[j] + insdel_cost(a_tok)
            insert_b = curr[j - 1] + insdel_cost(b_tok)
            substitute = prev[j - 1] + sub_cost(a_tok, b_tok)
            curr[j] = min(delete_a, insert_b, substitute)
        prev, curr = curr, prev

    return prev[m]


def ipa_distance(a: str, b: str) -> float:
    """Weighted edit distance between two IPA strings.  Normalized by the
    longer token sequence so long-vs-short comparisons stay on a shared
    scale."""
    at = _tokenize_phonemes(a)
    bt = _tokenize_phonemes(b)
    if not at and not bt:
        return 0.0
    raw = align_cost(at, bt)
    return raw / max(len(at), len(bt))


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
# CSV row we need to push it back to IPA.  Order: y → j before j → d͡ʒ so
# the jīm we produce isn't re-tagged as palatal.
_PAN_SCHOLAR_TO_IPA: list[tuple[str, str]] = [
    ("ṣ", "sˤ"),
    ("ṭ", "tˤ"),
    ("š", "ʃ"),
    ("ž", "ʒ"),
    ("y", "j"),
    ("j", "d͡ʒ"),
]


def pansemitic_scholar_to_ipa(form: str) -> str:
    out = form
    for src, dst in _PAN_SCHOLAR_TO_IPA:
        out = out.replace(src, dst)
    return out


# ── CLI driver ────────────────────────────────────────────────

def _main() -> None:
    import csv
    import statistics
    from pathlib import Path
    from reconstruction import ArabicWord, HebrewWord

    csv_path = Path("cognates2.csv")
    losses: list[float] = []
    skipped = 0
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ar_roman = row.get("arabic_romanization") or ""
            he_roman = row.get("hebrew_romanization") or ""
            pan_scholar = row.get("pansemitic") or ""
            if not (ar_roman and he_roman and pan_scholar):
                skipped += 1
                continue
            ar_ipa = ArabicWord.from_romanization(ar_roman).to_ipa()
            he_ipa = HebrewWord.from_romanization(he_roman).to_ipa()
            pan_ipa = pansemitic_scholar_to_ipa(pan_scholar)
            losses.append(triplet_loss(pan_ipa, ar_ipa, he_ipa))

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
