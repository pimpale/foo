#!/usr/bin/env python3
"""
Semitic Cognate Finder

For each of the 1000 most common English words, finds Arabic-Hebrew cognate
pairs by cross-referencing Wiktionary (kaikki) etymology data.

Two matching layers:
  Layer 1: Explicit cognate references in Arabic/Hebrew etymology templates
  Layer 2: Shared Proto-Semitic root matching
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path("data")
COMMON_WORDS_FILE = Path("1000-most-common-words.txt")
OUTPUT_FILE = Path("cognates.json")

ARABIC_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670\u0640]")
HEBREW_DIACRITICS = re.compile(r"[\u0591-\u05BD\u05BF-\u05C7]")


def normalize_arabic(text: str) -> str:
    return ARABIC_DIACRITICS.sub("", text).strip()


def normalize_hebrew(text: str) -> str:
    return HEBREW_DIACRITICS.sub("", text).strip()


def build_semitic_index(filepath, target_lang, normalize_self, normalize_target):
    """
    Single pass through an Arabic or Hebrew kaikki file. Extracts:
      cognates:     normalized_self_word -> {normalized_target_words}
      sem_roots:    normalized_self_word -> {Proto-Semitic roots}
      display_form: normalized_self_word -> first-seen original spelling
    """
    cognates = defaultdict(set)
    sem_roots = defaultdict(set)
    display_form = {}

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            word = entry.get("word", "")
            if not word:
                continue

            norm = normalize_self(word)
            if norm not in display_form:
                display_form[norm] = word

            for tmpl in entry.get("etymology_templates", []):
                args = tmpl.get("args", {})
                name = tmpl.get("name", "")

                # Layer 1 — explicit cognate reference to the other Semitic language
                if name == "cog" and args.get("1") == target_lang:
                    cog_word = args.get("2", "")
                    if cog_word:
                        cognates[norm].add(normalize_target(cog_word))

                # Layer 2 — any template that traces to a Proto-Semitic form
                if args.get("2") == "sem-pro":
                    root = args.get("3", "")
                    if root:
                        sem_roots[norm].add(root)

    return cognates, sem_roots, display_form


def _collect_translation(t, word, translations):
    """Add a single translation record to the accumulator if it's ar/he."""
    t_word = t.get("word", "")
    if not t_word:
        return
    lang = t.get("lang_code", "")
    roman = t.get("roman", "")

    if lang == "ar":
        norm = normalize_arabic(t_word)
        if norm not in translations[word]["ar"]:
            translations[word]["ar"][norm] = {"word": t_word, "roman": roman}
    elif lang == "he":
        norm = normalize_hebrew(t_word)
        if norm not in translations[word]["he"]:
            translations[word]["he"][norm] = {"word": t_word, "roman": roman}


def build_english_translations(filepath, common_words):
    """
    Single pass through the (large) English kaikki file.
    Collects every Arabic and Hebrew translation for each common word,
    aggregated across all entries/senses/POS for that word.

    Translations live in two places:
      - entry["translations"]           (top-level)
      - entry["senses"][n]["translations"]  (per-sense)
    """
    translations = defaultdict(lambda: {"ar": {}, "he": {}})

    with open(filepath, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % 200_000 == 0:
                print(f"  ... {i:,} English entries scanned", file=sys.stderr)

            entry = json.loads(line)
            word = entry.get("word", "").lower()
            if word not in common_words:
                continue

            for t in entry.get("translations", []):
                _collect_translation(t, word, translations)

            for sense in entry.get("senses", []):
                for t in sense.get("translations", []):
                    _collect_translation(t, word, translations)

    return translations


def match_pair(ar_norm, he_norm, ar2he, he2ar, ar_roots, he_roots):
    """Check a single (Arabic, Hebrew) pair against both layers."""
    layers = []
    shared = []

    if he_norm in ar2he.get(ar_norm, set()):
        layers.append("direct_cognate_ar→he")
    if ar_norm in he2ar.get(he_norm, set()):
        layers.append("direct_cognate_he→ar")

    common_roots = ar_roots.get(ar_norm, set()) & he_roots.get(he_norm, set())
    if common_roots:
        layers.append("shared_proto_semitic_root")
        shared = sorted(common_roots)

    return layers, shared


def main():
    print("Loading common words …")
    common_words = []
    with open(COMMON_WORDS_FILE) as f:
        for line in f:
            w = line.strip()
            if w:
                common_words.append(w.lower())
    common_set = set(common_words)
    print(f"  {len(common_words)} words")

    # ── Arabic index ──────────────────────────────────────────────
    print("\nIndexing Arabic dictionary …")
    ar2he_cog, ar_sem, ar_disp = build_semitic_index(
        DATA_DIR / "kaikki.org-dictionary-Arabic.jsonl",
        target_lang="he",
        normalize_self=normalize_arabic,
        normalize_target=normalize_hebrew,
    )
    print(f"  {len(ar2he_cog)} words with Hebrew cognate refs")
    print(f"  {len(ar_sem)} words with Proto-Semitic roots")

    # ── Hebrew index ──────────────────────────────────────────────
    print("\nIndexing Hebrew dictionary …")
    he2ar_cog, he_sem, he_disp = build_semitic_index(
        DATA_DIR / "kaikki.org-dictionary-Hebrew.jsonl",
        target_lang="ar",
        normalize_self=normalize_hebrew,
        normalize_target=normalize_arabic,
    )
    print(f"  {len(he2ar_cog)} words with Arabic cognate refs")
    print(f"  {len(he_sem)} words with Proto-Semitic roots")

    # ── English translations ──────────────────────────────────────
    print("\nScanning English dictionary for translations (≈1.4M entries) …")
    en_trans = build_english_translations(
        DATA_DIR / "kaikki.org-dictionary-English.jsonl",
        common_set,
    )
    with_ar = sum(1 for v in en_trans.values() if v["ar"])
    with_he = sum(1 for v in en_trans.values() if v["he"])
    with_both = sum(1 for v in en_trans.values() if v["ar"] and v["he"])
    print(f"  {len(en_trans)} common words found in English kaikki")
    print(f"  {with_ar} with ≥1 Arabic translation")
    print(f"  {with_he} with ≥1 Hebrew translation")
    print(f"  {with_both} with both Arabic + Hebrew")

    # ── N×M cognate matching ──────────────────────────────────────
    print("\nMatching cognates …")
    results = {}

    for word in common_words:
        tr = en_trans.get(word)
        if not tr or not tr["ar"] or not tr["he"]:
            continue

        matches = []
        for ar_norm, ar_info in tr["ar"].items():
            for he_norm, he_info in tr["he"].items():
                layers, shared = match_pair(
                    ar_norm, he_norm, ar2he_cog, he2ar_cog, ar_sem, he_sem
                )
                if not layers:
                    continue
                matches.append(
                    {
                        "arabic": ar_info["word"],
                        "arabic_roman": ar_info["roman"],
                        "hebrew": he_info["word"],
                        "hebrew_roman": he_info["roman"],
                        "match_layers": layers,
                        **({"shared_proto_semitic_roots": shared} if shared else {}),
                    }
                )

        if matches:
            results[word] = {
                "english": word,
                "arabic_candidates": [
                    v["word"] for v in tr["ar"].values()
                ],
                "hebrew_candidates": [
                    v["word"] for v in tr["he"].values()
                ],
                "cognate_matches": matches,
            }

    print(f"\n  {len(results)} / {len(common_words)} words produced cognate matches")

    print(f"\nWriting {OUTPUT_FILE} …")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
