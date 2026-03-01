#!/usr/bin/env python3
"""
Semitic Cognate Finder

For each of the 1000 most common English words, finds Arabic-Hebrew cognate
pairs by cross-referencing Wiktionary (kaikki) etymology data.

Three matching layers:
  Layer 1: Explicit cognate references in Arabic/Hebrew etymology templates
  Layer 2: Shared Proto-Semitic root matching
  Layer 3: Shared borrowing source (both borrowed from the same word)
"""

import csv
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import orjson

DATA_DIR = Path("data")
ALL_WORDS_FILE = DATA_DIR / "kaikki.org-dictionary-all-words.jsonl"
COMMON_WORDS_FILE = Path("1000-most-common-words.txt")
OUTPUT_FILE = Path("cognates.json")
CSV_FILE = Path("cognates.csv")

ARABIC_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670\u0640]")
HEBREW_DIACRITICS = re.compile(r"[\u0591-\u05BD\u05BF-\u05C7]")
LATIN_TAIL = re.compile(r"\s*[a-zA-Z].*$")

ETYMON_SEM_PRO = re.compile(r"sem-pro:([^<>\s]+)")
ETYMON_LANG_WORD = re.compile(r"([a-z]{2,}):([^<>\s:]+)")

BORROW_TEMPLATES = {"bor", "der", "lbor", "ubor", "slbor", "borrowed"}

# Byte-level pre-filters for the all-languages file
_AR = b'"ar"'
_HE = b'"he"'
_EN = b'"en"'


def normalize_arabic(text: str) -> str:
    return ARABIC_DIACRITICS.sub("", text).strip()


def normalize_hebrew(text: str) -> str:
    return HEBREW_DIACRITICS.sub("", text).strip()


def _clean_translation_word(text: str) -> str:
    cleaned = LATIN_TAIL.sub("", text).strip()
    return cleaned if cleaned else text.strip()


def _process_semitic_entry(entry, target_lang, normalize_self, normalize_target,
                           cognates, sem_roots, display_form, lemma_of,
                           borrow_sources, gloss_index, common_words):
    """Process a single Arabic or Hebrew kaikki entry."""
    word = entry.get("word", "")
    if not word:
        return

    norm = normalize_self(word)
    if norm not in display_form:
        display_form[norm] = word

    for tmpl in entry.get("etymology_templates", []):
        args = tmpl.get("args", {})
        name = tmpl.get("name", "")

        if name == "cog" and args.get("1") == target_lang:
            cog_word = args.get("2", "")
            if cog_word:
                cognates[norm].add(normalize_target(cog_word))

        if args.get("2") == "sem-pro":
            root = args.get("3", "")
            if root:
                sem_roots[norm].add(root)

        if name == "etymon":
            for v in args.values():
                if not isinstance(v, str):
                    continue
                for m in ETYMON_SEM_PRO.finditer(v):
                    sem_roots[norm].add(m.group(1))
                for m in ETYMON_LANG_WORD.finditer(v):
                    if m.group(1) == target_lang:
                        cognates[norm].add(normalize_target(m.group(2)))

        # Layer 3: track borrowing sources
        if name in BORROW_TEMPLATES:
            src_lang = args.get("2", "")
            src_word = args.get("3", "")
            if (src_lang and src_word
                    and src_word not in ("-", "?", "")
                    and len(src_word) > 1):
                borrow_sources[norm].add((src_lang, src_word.lower()))

    for sense in entry.get("senses", []):
        for fof in sense.get("form_of", []):
            base = fof.get("word", "")
            if base:
                lemma_of[norm].add(normalize_self(base))

    # Reverse gloss lookup
    if common_words:
        roman = ""
        for f in entry.get("forms", []):
            if "romanization" in f.get("tags", []):
                roman = f.get("form", "")
                break

        for sense in entry.get("senses", []):
            for gloss in sense.get("glosses", []):
                gloss_words = gloss.lower().split()
                if len(gloss_words) > 4:
                    continue
                gloss_tokens = set(re.findall(r"[a-z]+", gloss.lower()))
                for cw in gloss_tokens & common_words:
                    gloss_index[cw].add((norm, word, roman))


def _process_english_entry(entry, common_words, translations):
    """Process a single English kaikki entry for translations."""
    word = entry.get("word", "").lower()
    if word not in common_words:
        return

    for t in entry.get("translations", []):
        _collect_translation(t, word, translations)

    for sense in entry.get("senses", []):
        for t in sense.get("translations", []):
            _collect_translation(t, word, translations)


def _collect_translation(t, word, translations):
    t_word = _clean_translation_word(t.get("word", ""))
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


def build_all_indexes(filepath, common_words):
    """
    Single pass through the all-languages kaikki file.
    Builds Arabic indexes, Hebrew indexes, and English translations
    simultaneously, using byte-level pre-filtering to skip irrelevant lines.
    """
    # Arabic indexes
    ar2he_cog = defaultdict(set)
    ar_sem = defaultdict(set)
    ar_disp = {}
    ar_lemmas = defaultdict(set)
    ar_borrow = defaultdict(set)
    ar_gloss = defaultdict(set)

    # Hebrew indexes
    he2ar_cog = defaultdict(set)
    he_sem = defaultdict(set)
    he_disp = {}
    he_lemmas = defaultdict(set)
    he_borrow = defaultdict(set)
    he_gloss = defaultdict(set)

    # English translations
    en_trans = defaultdict(lambda: {"ar": {}, "he": {}})

    ar_count = 0
    he_count = 0
    en_count = 0

    with open(filepath, "rb") as f:
        for i, line in enumerate(f):
            if i % 1_000_000 == 0:
                print(f"  ... {i:,} lines scanned "
                      f"(ar:{ar_count:,} he:{he_count:,} en:{en_count:,})",
                      file=sys.stderr)

            if _AR in line:
                entry = orjson.loads(line)
                lc = entry.get("lang_code", "")
                if lc == "ar":
                    ar_count += 1
                    _process_semitic_entry(
                        entry, "he", normalize_arabic, normalize_hebrew,
                        ar2he_cog, ar_sem, ar_disp, ar_lemmas,
                        ar_borrow, ar_gloss, common_words,
                    )
                    continue
                elif lc == "en":
                    en_count += 1
                    _process_english_entry(entry, common_words, en_trans)
                    continue

            if _HE in line:
                entry = orjson.loads(line)
                lc = entry.get("lang_code", "")
                if lc == "he":
                    he_count += 1
                    _process_semitic_entry(
                        entry, "ar", normalize_hebrew, normalize_arabic,
                        he2ar_cog, he_sem, he_disp, he_lemmas,
                        he_borrow, he_gloss, common_words,
                    )
                    continue
                elif lc == "en":
                    en_count += 1
                    _process_english_entry(entry, common_words, en_trans)
                    continue

            if _EN in line:
                entry = orjson.loads(line)
                if entry.get("lang_code") == "en":
                    en_count += 1
                    _process_english_entry(entry, common_words, en_trans)

    print(f"  ... done: {i+1:,} lines total "
          f"(ar:{ar_count:,} he:{he_count:,} en:{en_count:,})")

    return (
        ar2he_cog, ar_sem, ar_disp, ar_lemmas, ar_borrow, ar_gloss,
        he2ar_cog, he_sem, he_disp, he_lemmas, he_borrow, he_gloss,
        en_trans,
    )


def _expand_with_lemmas(norm, lemma_map):
    forms = {norm}
    forms.update(lemma_map.get(norm, set()))
    return forms


def match_pair(ar_norm, he_norm, ar2he, he2ar, ar_roots, he_roots,
               ar_lemmas, he_lemmas, ar_borrow, he_borrow):
    """Check a single (Arabic, Hebrew) pair against all three layers."""
    layers = []
    shared_roots = []
    shared_sources = []

    ar_forms = _expand_with_lemmas(ar_norm, ar_lemmas)
    he_forms = _expand_with_lemmas(he_norm, he_lemmas)

    # Layer 1: direct cognate references
    for af in ar_forms:
        for hf in he_forms:
            if hf in ar2he.get(af, set()):
                layers.append("direct_cognate_ar→he")
            if af in he2ar.get(hf, set()):
                layers.append("direct_cognate_he→ar")
    layers = list(dict.fromkeys(layers))

    # Layer 2: shared Proto-Semitic root
    all_ar_roots = set()
    for af in ar_forms:
        all_ar_roots.update(ar_roots.get(af, set()))
    all_he_roots = set()
    for hf in he_forms:
        all_he_roots.update(he_roots.get(hf, set()))
    common_roots = all_ar_roots & all_he_roots
    if common_roots:
        layers.append("shared_proto_semitic_root")
        shared_roots = sorted(common_roots)

    # Layer 3: shared borrowing source
    all_ar_borrow = set()
    for af in ar_forms:
        all_ar_borrow.update(ar_borrow.get(af, set()))
    all_he_borrow = set()
    for hf in he_forms:
        all_he_borrow.update(he_borrow.get(hf, set()))
    common_borrow = all_ar_borrow & all_he_borrow
    if common_borrow:
        layers.append("shared_borrowing_source")
        shared_sources = sorted(f"{lang}:{word}" for lang, word in common_borrow)

    return layers, shared_roots, shared_sources


def main():
    t_total = time.monotonic()

    print("Loading common words …")
    common_words = []
    with open(COMMON_WORDS_FILE) as f:
        for line in f:
            w = line.strip()
            if w:
                common_words.append(w.lower())
    common_set = set(common_words)
    print(f"  {len(common_words)} words")

    # ── Single pass through all-languages file ────────────────────
    print(f"\nScanning {ALL_WORDS_FILE.name} …")
    t0 = time.monotonic()
    (
        ar2he_cog, ar_sem, ar_disp, ar_lemmas, ar_borrow, ar_gloss,
        he2ar_cog, he_sem, he_disp, he_lemmas, he_borrow, he_gloss,
        en_trans,
    ) = build_all_indexes(ALL_WORDS_FILE, common_set)
    scan_time = time.monotonic() - t0

    print(f"\n  Arabic:  {len(ar2he_cog)} cognate refs, {len(ar_sem)} sem roots, "
          f"{len(ar_lemmas)} lemma links, {len(ar_borrow)} borrow sources, "
          f"{len(ar_gloss)} gloss matches")
    print(f"  Hebrew:  {len(he2ar_cog)} cognate refs, {len(he_sem)} sem roots, "
          f"{len(he_lemmas)} lemma links, {len(he_borrow)} borrow sources, "
          f"{len(he_gloss)} gloss matches")

    # Merge gloss-based candidates into the translation pool
    for cw in common_set:
        if cw not in en_trans:
            en_trans[cw] = {"ar": {}, "he": {}}
        for norm, display, roman in ar_gloss.get(cw, set()):
            if norm not in en_trans[cw]["ar"]:
                en_trans[cw]["ar"][norm] = {"word": display, "roman": roman}
        for norm, display, roman in he_gloss.get(cw, set()):
            if norm not in en_trans[cw]["he"]:
                en_trans[cw]["he"][norm] = {"word": display, "roman": roman}

    with_ar = sum(1 for v in en_trans.values() if v["ar"])
    with_he = sum(1 for v in en_trans.values() if v["he"])
    with_both = sum(1 for v in en_trans.values() if v["ar"] and v["he"])
    print(f"  English: {len(en_trans)} words, {with_ar} with ar, "
          f"{with_he} with he, {with_both} with both")
    print(f"  ⏱ {scan_time:.1f}s")

    # ── N×M cognate matching ──────────────────────────────────────
    print("\nMatching cognates …")
    t0 = time.monotonic()
    results = {}
    words_checked = 0

    for word in common_words:
        tr = en_trans.get(word)
        if not tr or not tr["ar"] or not tr["he"]:
            continue

        words_checked += 1
        matches = []
        for ar_norm, ar_info in tr["ar"].items():
            for he_norm, he_info in tr["he"].items():
                layers, shared_roots, shared_sources = match_pair(
                    ar_norm, he_norm, ar2he_cog, he2ar_cog, ar_sem, he_sem,
                    ar_lemmas, he_lemmas, ar_borrow, he_borrow,
                )
                if not layers:
                    continue
                match_entry = {
                    "arabic": ar_info["word"],
                    "arabic_roman": ar_info["roman"],
                    "hebrew": he_info["word"],
                    "hebrew_roman": he_info["roman"],
                    "match_layers": layers,
                }
                if shared_roots:
                    match_entry["shared_proto_semitic_roots"] = shared_roots
                if shared_sources:
                    match_entry["shared_borrowing_sources"] = shared_sources
                matches.append(match_entry)

        if matches:
            results[word] = {
                "english": word,
                "arabic_candidates": [v["word"] for v in tr["ar"].values()],
                "hebrew_candidates": [v["word"] for v in tr["he"].values()],
                "cognate_matches": matches,
            }

    match_elapsed = time.monotonic() - t0
    avg_ms = (match_elapsed / words_checked * 1000) if words_checked else 0
    print(f"\n  {len(results)} / {len(common_words)} words produced cognate matches")
    print(f"  ⏱ {match_elapsed:.1f}s ({words_checked} words checked, {avg_ms:.2f}ms avg)")

    print(f"\nWriting {OUTPUT_FILE} …")
    with open(OUTPUT_FILE, "wb") as f:
        f.write(orjson.dumps(results, option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS))

    print(f"Writing {CSV_FILE} …")
    with open(CSV_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["english", "arabic_romanization", "hebrew_romanization"])
        for word in common_words:
            if word not in results:
                continue
            for m in results[word]["cognate_matches"]:
                writer.writerow([word, m["arabic_roman"], m["hebrew_roman"]])

    print(f"\nDone in {time.monotonic() - t_total:.1f}s total.")


if __name__ == "__main__":
    main()
