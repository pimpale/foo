#!/usr/bin/env python3
"""
Phonetic reconstruction and pansemitic form generation.

reconstruct_ancestor:
    Given a cognate pair, returns the best ancestor form as "lang:word".
    Prefers shared borrowing sources, then proto-semitic roots, then
    reconstructs from the Arabic form using known sound correspondences.

construct_pansemitic_form:
    Converts an ancestor form into a "pansemitic" intermediate form:
    - Vowels from Arabic (a, i, u — no length distinction)
    - Consonant inventory compressed to Hebrew's phonology
"""

import re

# ── Reconstruction from Arabic romanization ──────────────────────
#
# Arabic is generally more conservative consonantally than Hebrew,
# but has a few innovations we need to undo:
#
#   Arabic f  ← Proto-Semitic *p  (Hebrew kept p)
#   Arabic ṯ  ← PS *ṯ           (Hebrew merged → š/sh)
#   Arabic ḏ  ← PS *ḏ           (Hebrew merged → z)
#   Arabic ɣ  ← PS *ɣ           (Hebrew merged → ʿ)
#   Arabic ḵ  ← PS *ḫ           (Hebrew merged → ḥ)
#   Arabic w- ← PS *w-          (Hebrew shifted → y- word-initially)

# When reconstructing from Arabic, reverse Arabic-specific changes:
_AR_TO_PROTO = [
    # Arabic f → *p (Arabic p > f shift)
    ("f", "p"),
]

# Multi-char replacements applied first (order matters)
_AR_TO_PROTO_MULTI = [
    # No multi-char Arabic innovations to reverse currently;
    # Arabic kept the archaic consonants (ṯ, ḏ, etc.)
]


def _strip_vowel_length(text):
    """Remove vowel length markers: ā→a, ī→i, ū→u."""
    return (text
            .replace("ā", "a")
            .replace("ī", "i")
            .replace("ū", "u")
            .replace("á", "a")
            .replace("í", "i")
            .replace("ú", "u")
            .replace("ó", "o")
            .replace("é", "e"))


def _normalize_vowels_to_aiu(text):
    """Collapse vowels to {a, i, u} — Arabic's 3-vowel system.
    e/o are not native to proto-Semitic; map them back."""
    text = _strip_vowel_length(text)
    # e → i, o → u (these are Canaanite/Hebrew innovations)
    text = text.replace("e", "i")
    text = text.replace("o", "a")
    return text


def _reconstruct_from_arabic(ar_roman, he_roman):
    """Reconstruct a proto-form from Arabic romanization,
    cross-referencing Hebrew where Arabic innovated."""
    if not ar_roman:
        return None

    form = ar_roman.lower()

    # Strip leading glottal stop (ʔ) — often just orthographic in Arabic
    form = form.lstrip("ʔ")

    # Word-initial w/y: if Arabic has w- and Hebrew has y-,
    # the proto-form is *w- (Hebrew w > y)
    # Keep Arabic w- (it's archaic). Nothing to change.

    # Reverse Arabic f → *p
    form = form.replace("f", "p")

    # Strip vowel length
    form = _strip_vowel_length(form)

    # Remove word-final case endings if present
    form = re.sub(r"[aiu]n$", "", form)

    # Remove common affixes: al- prefix
    form = re.sub(r"^al-", "", form)

    return form


# ── Pansemitic form: Hebrew consonants + Arabic vowels ───────────
#
# Hebrew compressed Proto-Semitic's consonant inventory. Pansemitic
# adopts those mergers so the result is pronounceable for Hebrew speakers:
#
#   *ṯ (Arabic ṯ, theta) → sh   (Hebrew š)
#   *ḏ (Arabic ḏ, edh)   → z    (Hebrew z)
#   *ɣ (Arabic ḡ/ɣ, gh)  → ʿ    (Hebrew ayin)
#   *ḫ (Arabic ḵ, kh)    → ḥ    (Hebrew ḥet)
#   *ṣ́ (Arabic ḍ/ḍ̣)     → ts   (Hebrew tsade)
#   *ṱ (Arabic ẓ)        → ts   (Hebrew tsade)
#
# Vowels: Arabic's {a, i, u} with no length distinction.

_PROTO_TO_PANSEMITIC = [
    # Multi-char first
    ("ṯṯ", "shsh"),
    ("ḏḏ", "zz"),
    ("ḵḵ", "ḥḥ"),
    # Single-char
    ("ṯ", "sh"),
    ("ḏ", "z"),
    ("ɣ", "ʿ"),
    ("ḡ", "ʿ"),
    ("ḵ", "ḥ"),
    ("ẓ", "ts"),
    ("ḍ", "ts"),
]


def reconstruct_ancestor(ar_roman, he_roman, shared_sources=None):
    """Return the best ancestor form as 'lang:word'.

    Priority:
      1. Shared etymology source — LCA from the borrowing/inheritance graph
         (already sorted by the caller, first entry is the best)
      2. Reconstruction from Arabic romanization
    """
    # 1. Shared etymology source — pick the first (best) one
    if shared_sources:
        # shared_sources are strings like "akk:abum", "sem-pro:*ʔab-"
        return shared_sources[0]

    # 2. Reconstruct from Arabic
    form = _reconstruct_from_arabic(ar_roman, he_roman)
    if form:
        return f"recon:{form}"

    return None


def construct_pansemitic_form(ancestor_str):
    """Convert an ancestor form into a pansemitic intermediate.

    Takes a 'lang:word' string and applies sound changes to produce
    a form with Arabic's 3-vowel system and Hebrew's consonant inventory.
    """
    if not ancestor_str:
        return None

    # Split lang:word
    if ":" in ancestor_str:
        lang, word = ancestor_str.split(":", 1)
    else:
        return None

    word = word.lower()

    # Strip common affixes/decorations
    word = word.lstrip("*").rstrip("-")

    # For Akkadian and other ancestors, some specific adjustments
    # Akkadian kept p (no f shift), had š, etc.

    # Step 1: Normalize to proto-level consonants based on source language
    if lang == "sem-pro":
        # Already proto — just apply pansemitic changes
        form = word
    elif lang == "recon":
        # Already reconstructed from Arabic with f→p etc.
        form = word
    elif lang == "akk":
        # Akkadian: š→sh, kept p, gemination preserved
        form = word.replace("š", "sh")
        # Akkadian ss/ss → simplify
    elif lang in ("la", "grc", "fr", "en", "it", "es", "pt"):
        # European loanwords — keep as-is, just normalize vowels
        form = word
    elif lang == "arc":
        # Aramaic — very close to proto-Semitic
        form = word
    elif lang in ("fa", "pal"):
        # Persian — keep as-is
        form = word
    elif lang == "egy":
        # Egyptian — keep as-is
        form = word
    else:
        # Default: treat like proto
        form = word

    # Step 2: Apply consonant compression (proto → pansemitic)
    for old, new in _PROTO_TO_PANSEMITIC:
        form = form.replace(old, new)

    # Also: Arabic f should already be p from reconstruction,
    # but if source language had f, reverse it
    # (Only for Semitic sources where f < p)
    if lang in ("sem-pro", "recon", "akk", "arc"):
        form = form.replace("f", "p")

    # Step 3: Normalize vowels to a/i/u (no length, no e/o)
    form = _normalize_vowels_to_aiu(form)

    # Step 4: Clean up
    # Remove doubled vowels
    form = re.sub(r"([aiu])\1+", r"\1", form)
    # Strip any remaining diacritical marks on consonants that we missed
    # but keep ʔ (glottal stop), ʿ (ayin), ḥ (het)
    form = form.strip()

    return form
