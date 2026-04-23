#!/usr/bin/env python3
"""
Direct Semitic Cognate Finder

Finds Arabic-Hebrew cognate pairs directly from Wiktionary etymology data,
without using English as a bridge language.

Three matching layers:
  Layer 1: Explicit cognate references in Arabic/Hebrew etymology templates
  Layer 2: Shared Proto-Semitic root matching
  Layer 3: Shared borrowing source — direct or via transitive chain through
           a global etymology graph (e.g. Arabic←French←Latin→Hebrew)
"""

from __future__ import annotations

import csv
import dataclasses
import os
from dataclasses import dataclass, field
from urllib.parse import quote
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import numpy as np
import orjson

from reconstruction import (
    SharedSource,
    reconstruct_ancestor,
    ReconstructionError,
    UnsupportedLanguageError,
    ConsonantMismatchError,
    MissingRomanizationError,
    EmptyAncestorError,
)

DATA_DIR = Path("data")
ALL_WORDS_FILE = DATA_DIR / "kaikki.org-dictionary-all-words.jsonl"
OUTPUT_FILE = Path("cognates2.json")
CSV_FILE = Path("cognates2.csv")
SENSES_FILE = Path("senses.json")
EMBEDDINGS_FILE = Path("embeddings.npy")
FALSE_POSITIVES_FILE = Path("false-positives.txt")

ARABIC_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670\u0640]")
HEBREW_DIACRITICS = re.compile(r"[\u0591-\u05BD\u05BF-\u05C7]")

ETYMON_SEM_PRO = re.compile(r"sem-pro:([^<>\s]+)")
ETYMON_LANG_WORD = re.compile(r"([a-z]{2,}(?:-[a-z]+)*):([^<>\s:]+)")
ETYMON_METADATA = re.compile(r"<[^>]*>")

ETYMOLOGY_TEMPLATES = {"bor", "der", "lbor", "ubor", "slbor", "borrowed",
                       "inh", "inh+", "bor+", "der+"}
ETYMON_RELATIONS = {":bor", ":der", ":inh", ":from", ":lbor"}


def _extract_canonical(entry: dict[str, Any]) -> str:
    """Return the canonical form from a kaikki entry, or the headword as fallback."""
    for fm in entry.get("forms", []):
        if "canonical" in fm.get("tags", []):
            return fm.get("form", entry.get("word", ""))
    return entry.get("word", "")


_AR = b'"ar"'
_HE = b'"he"'
_ETYM = b'"etymology_templates"'


@dataclass
class WordData:
    """Per-word accumulated data from kaikki entries."""
    canonical: str
    norm: str
    romanization: str = ""
    glosses: set[str] = field(default_factory=set)
    cognates: set[tuple[str, str]] = field(default_factory=set)  # (normalized, raw)
    lemma_of: set[str] = field(default_factory=set)
    borrow_sources: set[tuple[str, str]] = field(default_factory=set)


@dataclass
class CognatePair:
    """A matched Arabic-Hebrew cognate pair with evidence."""
    ar_canonical: str
    he_canonical: str
    layers: list[str] = field(default_factory=list)
    # (lang, word) → (ar_depth, he_depth) — depth 0 = direct source
    sources: dict[tuple[str, str], tuple[int, int]] = field(default_factory=dict)


@dataclass
class SenseMatch:
    """Best-matching sense pair between an Arabic and Hebrew word."""
    arabic_sense: str
    arabic_pos: str
    hebrew_sense: str
    hebrew_pos: str
    similarity: float

@dataclass
class LangEntry:
    canonical: str
    roman: str
    glosses: list[str]
    wiktionary: str

@dataclass
class CognateEntry:
    """Final output entry for a cognate pair."""
    arabic: LangEntry
    hebrew: LangEntry
    match_layers: list[str]
    shared_borrowing_sources: dict[str, tuple[int, int]] | None = None
    best_sense_match: SenseMatch | None = None
    ancestor: str | None = None
    pansemitic_form: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict, omitting None fields."""
        d = dataclasses.asdict(self)
        return {k: v for k, v in d.items() if v is not None}


def normalize_arabic(text: str) -> str:
    # return text.strip()
    return ARABIC_DIACRITICS.sub("", text).strip()


def normalize_hebrew(text: str) -> str:
    # return text.strip()
    return HEBREW_DIACRITICS.sub("", text).strip()


def _process_semitic_entry(
    entry: dict[str, Any],
    target_lang: str,
    normalize_self: Callable[[str], str],
    normalize_target: Callable[[str], str],
    words: dict[str, WordData],
    canonical: str,
) -> None:
    """Process a single Arabic or Hebrew kaikki entry."""
    word = entry.get("word", "")
    if not word:
        return

    norm = normalize_self(word)
    if canonical not in words:
        words[canonical] = WordData(canonical=canonical, norm=norm)
    wd = words[canonical]

    if not wd.romanization:
        for fm in entry.get("forms", []):
            if "romanization" in fm.get("tags", []):
                wd.romanization = fm.get("form", "")
                break

    for tmpl in entry.get("etymology_templates", []):
        args = tmpl.get("args", {})
        name = tmpl.get("name", "")

        if name == "cog" and args.get("1") == target_lang:
            cog_word = args.get("2", "")
            if cog_word:
                raw = cog_word
                # Also check arg3 for diacritized form
                arg3 = args.get("3", "")
                if arg3 and arg3 not in ("-", "?", ""):
                    raw = arg3
                wd.cognates.add((normalize_target(cog_word), raw))

        if name == "etymon":
            for v in args.values():
                if not isinstance(v, str):
                    continue
                for m in ETYMON_LANG_WORD.finditer(ETYMON_METADATA.sub("", v)):
                    if m.group(1) == target_lang:
                        raw = m.group(2)
                        wd.cognates.add((normalize_target(raw), raw))

        if name in ETYMOLOGY_TEMPLATES:
            src_lang = args.get("2", "")
            src_word = args.get("3", "")
            if (src_lang and src_word
                    and src_word not in ("-", "?", "")
                    and len(src_word) > 1):
                wd.borrow_sources.add((src_lang, src_word))

    for sense in entry.get("senses", []):
        for fof in sense.get("form_of", []):
            base = fof.get("word", "")
            if base:
                wd.lemma_of.add(normalize_self(base))
        for gloss in sense.get("glosses", []):
            wd.glosses.add(gloss)


def _extract_borrowing(entry, lang_code, borrow_graph, template_tr_index):
    """Extract etymology sources from any entry into the global graph.

    Handles both positional templates (bor, der, inh, …) and the newer
    etymon template which encodes lang:word<metadata> in its args.

    Also captures the ``tr`` (transliteration) arg from templates into
    *template_tr_index* keyed by ``(src_lang, src_word)``.
    """
    word = entry.get("word", "")
    if not word:
        return
    key = (lang_code, word)
    for tmpl in entry.get("etymology_templates", []):
        name = tmpl.get("name", "")
        args = tmpl.get("args", {})

        if name in ETYMOLOGY_TEMPLATES:
            src_lang = args.get("2", "")
            src_word = args.get("3", "")
            if (src_lang and src_word
                    and src_word not in ("-", "?", "")
                    and len(src_word) > 1):
                src_key = (src_lang, src_word)
                borrow_graph[key].add(src_key)
                tr = args.get("tr", "")
                if tr and src_key not in template_tr_index:
                    template_tr_index[src_key] = tr

        elif name == "etymon":
            # Values contain relationship markers (:bor, :inh, …) and
            # lang:word<metadata> pairs.  Extract lang:word from any arg
            # that follows a relationship marker.
            vals = list(args.values())
            for v in vals:
                if not isinstance(v, str):
                    continue
                if v in ETYMON_RELATIONS or v == ":inh":
                    pass  # next value(s) should be lang:word
                # Match lang:word (strip <…> metadata first)
                cleaned = ETYMON_METADATA.sub("", v)
                for m in ETYMON_LANG_WORD.finditer(cleaned):
                    src_lang = m.group(1)
                    src_word = m.group(2)
                    if (src_word not in ("-", "?", "", "*")
                            and len(src_word) > 1
                            and src_lang != lang_code):
                        borrow_graph[key].add((src_lang, src_word))


def _expand_borrow_transitive(borrow_map, borrow_graph, max_depth=10):
    """Expand borrow sources by walking the borrowing graph, tracking depth.

    Converts borrow_map in-place from {canonical: set((lang, word), ...)}
    to {canonical: dict{(lang, word): depth}}, where depth 0 = direct source.
    """
    expanded_count = 0
    for canonical in list(borrow_map):
        sources = borrow_map[canonical]
        # Convert flat set to depth dict — direct sources are depth 0
        depth_map: dict[tuple[str, str], int] = {s: 0 for s in sources}
        frontier = set(sources)
        visited = set()
        current_depth = 0
        for _ in range(max_depth):
            next_frontier = set()
            for node in frontier:
                if node in visited:
                    continue
                visited.add(node)
                parents = borrow_graph.get(node, set())
                for p in parents:
                    if p not in depth_map:
                        depth_map[p] = current_depth + 1
                    next_frontier.add(p)
            frontier = next_frontier - visited
            current_depth += 1
            if not frontier:
                break
        if len(depth_map) > len(sources):
            expanded_count += 1
        borrow_map[canonical] = depth_map
    return expanded_count


def _extract_kaikki_romanization(entry) -> str:
    """Return the romanization from a kaikki entry's forms, or ''."""
    for fm in entry.get("forms", []):
        if "romanization" in fm.get("tags", []):
            return fm.get("form", "")
    return ""


def _process_chunk(
    filepath_str: str, start_byte: int, end_byte: int,
) -> tuple[
    dict[str, WordData],
    dict[str, WordData],
    dict[tuple[str, str], set[tuple[str, str]]],
    dict[tuple[str, str], str],
    dict[tuple[str, str], str],
    tuple[int, int, int],
]:
    """Worker: process one byte-range of the JSONL file, return partial indexes."""
    ar_words: dict[str, WordData] = {}
    he_words: dict[str, WordData] = {}
    borrow_graph: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    template_tr_index: dict[tuple[str, str], str] = {}
    kaikki_roman_index: dict[tuple[str, str], str] = {}

    ar_count = he_count = line_count = 0

    with open(filepath_str, "rb") as f:
        if start_byte > 0:
            f.seek(start_byte)
            f.readline()

        while f.tell() < end_byte:
            line = f.readline()
            if not line:
                break

            line_count += 1

            if not (_AR in line or _HE in line or _ETYM in line):
                continue

            entry = orjson.loads(line)
            lc = entry.get("lang_code", "")

            if lc == "ar":
                ar_count += 1
                canonical = _extract_canonical(entry)
                _process_semitic_entry(
                    entry, "he", normalize_arabic, normalize_hebrew, ar_words,
                    canonical=canonical,
                )
            elif lc == "he":
                he_count += 1
                canonical = _extract_canonical(entry)
                _process_semitic_entry(
                    entry, "ar", normalize_hebrew, normalize_arabic, he_words,
                    canonical=canonical,
                )

            # Collect romanization for any entry (for kaikki lookup tier)
            word = entry.get("word", "")
            if lc and word:
                key = (lc, word)
                if key not in kaikki_roman_index:
                    roman = _extract_kaikki_romanization(entry)
                    if roman:
                        kaikki_roman_index[key] = roman

            _extract_borrowing(entry, lc, borrow_graph, template_tr_index)

    return (ar_words, he_words, dict(borrow_graph),
            template_tr_index, kaikki_roman_index,
            (ar_count, he_count, line_count))


def _merge_set_dicts(target: dict, source: dict) -> None:
    for key, values in source.items():
        if key in target:
            target[key] |= values
        else:
            target[key] = set(values)


def _merge_word_dicts(target: dict[str, WordData], source: dict[str, WordData]) -> None:
    """Merge WordData dicts, combining set fields for shared canonical keys."""
    for canonical, wd in source.items():
        if canonical not in target:
            target[canonical] = wd
        else:
            t = target[canonical]
            if not t.romanization:
                t.romanization = wd.romanization
            t.glosses |= wd.glosses
            t.cognates |= wd.cognates
            t.lemma_of |= wd.lemma_of
            t.borrow_sources |= wd.borrow_sources


def _merge_str_dicts(target: dict, source: dict) -> None:
    """Merge str-valued dicts, keeping the first value for each key."""
    for key, value in source.items():
        if key not in target:
            target[key] = value


def build_all_indexes(
    filepath: Path, num_workers: int | None = None,
) -> tuple[
    dict[str, WordData],
    dict[str, WordData],
    dict[tuple[str, str], set[tuple[str, str]]],
    dict[tuple[str, str], str],
    dict[tuple[str, str], str],
]:
    """
    Parallel scan of the all-languages kaikki file.
    Extracts Arabic, Hebrew, borrowing graph, template transliterations,
    and kaikki romanization data.
    """
    if num_workers is None:
        num_workers = os.cpu_count() or 1

    file_size = filepath.stat().st_size
    chunk_size = file_size // num_workers

    chunks = []
    for i in range(num_workers):
        start = i * chunk_size
        end = file_size if i == num_workers - 1 else (i + 1) * chunk_size
        chunks.append((str(filepath), start, end))

    print(f"  dispatching {num_workers} workers "
          f"(~{chunk_size // (1024*1024)} MB/chunk) …", file=sys.stderr)

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_process_chunk, *c): i
                   for i, c in enumerate(chunks)}
        partial_results: list[Any] = [None] * num_workers
        done_count = 0
        totals = [0, 0, 0]

        for future in as_completed(futures):
            idx = futures[future]
            partial_results[idx] = future.result()
            counts = partial_results[idx][-1]
            for j in range(len(totals)):
                totals[j] += counts[j]
            done_count += 1
            print(f"  ... {done_count}/{num_workers} chunks done",
                  file=sys.stderr)

    print("  merging indexes …", file=sys.stderr)
    ar_words: dict[str, WordData] = {}
    he_words: dict[str, WordData] = {}
    borrow_graph: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    template_tr_index: dict[tuple[str, str], str] = {}
    kaikki_roman_index: dict[tuple[str, str], str] = {}

    for result in partial_results:
        p_ar, p_he, p_graph, p_tr, p_roman, _ = result
        _merge_word_dicts(ar_words, p_ar)
        _merge_word_dicts(he_words, p_he)
        _merge_set_dicts(borrow_graph, p_graph)
        _merge_str_dicts(template_tr_index, p_tr)
        _merge_str_dicts(kaikki_roman_index, p_roman)

    ar_n, he_n, lines_n = totals
    print(f"  ... done: {lines_n:,} lines total "
          f"(ar:{ar_n:,} he:{he_n:,}"
          f" graph:{len(borrow_graph):,} nodes)"
          f"\n  ... {len(template_tr_index):,} template transliterations,"
          f" {len(kaikki_roman_index):,} kaikki romanizations")

    return (ar_words, he_words, borrow_graph, template_tr_index, kaikki_roman_index)


def _make_cognate_index(words: dict[str, WordData]) -> dict[str, set[tuple[str, str]]]:
    """Extract cognate refs from WordData, keyed by canonical form.

    Values are sets of (normalized_target, raw_target) tuples.
    """
    return {canonical: set(wd.cognates) for canonical, wd in words.items() if wd.cognates}


def _make_borrow_index(words: dict[str, WordData]) -> dict[str, dict[tuple[str, str], int]]:
    """Extract borrow sources from WordData, keyed by canonical form.

    Initially all direct sources have depth 0; _expand_borrow_transitive
    adds transitive ancestors with increasing depth.
    """
    return {canonical: {s: 0 for s in wd.borrow_sources}
            for canonical, wd in words.items() if wd.borrow_sources}


def _find_lcas(
    sources: dict[tuple[str, str], tuple[int, int]],
    borrow_graph: dict[tuple[str, str], set[tuple[str, str]]],
) -> list[tuple[str, str]]:
    """Find lowest common ancestors among shared borrowing sources.

    A common ancestor is an LCA if none of its descendants (children in the
    borrow graph) are also in the common ancestor set.  The borrow_graph maps
    child → parents, so we invert it to get parent → children for the
    descendant check.
    """
    if len(sources) <= 1:
        return list(sources.keys())

    source_set = set(sources.keys())

    # Build local parent→children map for just the sources we care about
    children_of: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    for node in source_set:
        for parent in borrow_graph.get(node, set()):
            if parent in source_set:
                children_of[parent].add(node)

    # LCAs: sources that have no descendant also in the source set
    lcas = [s for s in source_set if not children_of.get(s)]

    # Tiebreak: sort by combined depth (ar_depth + he_depth)
    lcas.sort(key=lambda s: sum(sources[s]))
    return lcas


def main():
    t_total = time.monotonic()

    false_pos: set[tuple[str, str]] = set()
    if FALSE_POSITIVES_FILE.exists():
        with open(FALSE_POSITIVES_FILE) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) == 2:
                    false_pos.add((parts[0], parts[1]))
        print(f"  {len(false_pos)} false positive pairs loaded")

    # ── Single pass through all-languages file ────────────────────
    print(f"\nScanning {ALL_WORDS_FILE.name} …")
    t0 = time.monotonic()
    ar_words, he_words, borrow_graph, template_tr_index, kaikki_roman_index = build_all_indexes(ALL_WORDS_FILE)
    scan_time = time.monotonic() - t0

    # Build standalone cognate/borrow indexes keyed by canonical form
    ar2he_cog = _make_cognate_index(ar_words)
    he2ar_cog = _make_cognate_index(he_words)
    ar_borrow = _make_borrow_index(ar_words)
    he_borrow = _make_borrow_index(he_words)

    # Build norm→canonical reverse indexes for fallback resolution
    ar_norm_to_canonicals: dict[str, list[str]] = defaultdict(list)
    for canonical, wd in ar_words.items():
        ar_norm_to_canonicals[wd.norm].append(canonical)
    he_norm_to_canonicals: dict[str, list[str]] = defaultdict(list)
    for canonical, wd in he_words.items():
        he_norm_to_canonicals[wd.norm].append(canonical)

    ar_lemma_count = sum(1 for wd in ar_words.values() if wd.lemma_of)
    he_lemma_count = sum(1 for wd in he_words.values() if wd.lemma_of)

    print(f"\n  Arabic:  {len(ar2he_cog)} cognate refs, "
          f"{ar_lemma_count} lemma links, {len(ar_borrow)} borrow/etym sources")
    print(f"  Hebrew:  {len(he2ar_cog)} cognate refs, "
          f"{he_lemma_count} lemma links, {len(he_borrow)} borrow/etym sources")
    print(f"  Borrow graph: {len(borrow_graph)} nodes")

    ar_expanded = _expand_borrow_transitive(ar_borrow, borrow_graph)
    he_expanded = _expand_borrow_transitive(he_borrow, borrow_graph)
    print(f"  Transitive expansion: {ar_expanded} ar words, {he_expanded} he words")
    print(f"  ⏱ {scan_time:.1f}s")

    # ── Direct cognate matching ──────────────────────────────────
    print("\nMatching cognates directly …")
    t0 = time.monotonic()

    _JUNK = {"", "-", "?"}
    pair_data: dict[tuple[str, str], CognatePair] = {}

    def _is_root_or_single(word: str) -> bool:
        """Roots have 2+ hyphens (regular or maqaf), single letters have 1 grapheme."""
        if len(word) <= 1:
            return True
        hyphens = word.count("-") + word.count("\u05be")
        return hyphens >= 2

    def _ensure_pair(ar_c: str, he_c: str) -> CognatePair | None:
        ar_wd = ar_words.get(ar_c)
        he_wd = he_words.get(he_c)
        ar_n = ar_wd.norm if ar_wd else ar_c
        he_n = he_wd.norm if he_wd else he_c
        if ar_n in _JUNK or he_n in _JUNK:
            return None
        if _is_root_or_single(ar_n) or _is_root_or_single(he_n):
            return None
        if (ar_n, he_n) in false_pos:
            return None
        key = (ar_c, he_c)
        if key not in pair_data:
            pair_data[key] = CognatePair(ar_canonical=ar_c, he_canonical=he_c)
        return pair_data[key]

    def _resolve_target(raw: str, norm: str, target_words: dict[str, WordData],
                        norm_to_canonicals: dict[str, list[str]]) -> list[str]:
        """3-tier cognate target resolution → list of canonical keys."""
        # Tier 1: direct match on raw form (may be diacritized canonical)
        if raw in target_words:
            return [raw]
        # Tier 2: normalized fallback — find all canonical forms sharing the norm
        candidates = norm_to_canonicals.get(norm, [])
        if candidates:
            return candidates
        return []

    # Layer 1: Direct cognate references (ar→he)
    for ar_canonical, cog_pairs in ar2he_cog.items():
        for he_norm, he_raw in cog_pairs:
            he_targets = _resolve_target(he_raw, he_norm, he_words, he_norm_to_canonicals)
            for he_c in he_targets:
                pair = _ensure_pair(ar_canonical, he_c)
                if pair and "direct_cognate_ar→he" not in pair.layers:
                    pair.layers.append("direct_cognate_ar→he")

    # Layer 1: Direct cognate references (he→ar)
    for he_canonical, cog_pairs in he2ar_cog.items():
        for ar_norm, ar_raw in cog_pairs:
            ar_targets = _resolve_target(ar_raw, ar_norm, ar_words, ar_norm_to_canonicals)
            for ar_c in ar_targets:
                pair = _ensure_pair(ar_c, he_canonical)
                if pair and "direct_cognate_he→ar" not in pair.layers:
                    pair.layers.append("direct_cognate_he→ar")

    # Layer 2: Shared etymology source (borrowing, inheritance, or derivation)
    # ar_borrow/he_borrow are now {canonical: {(lang,word): depth}}
    source_to_ar: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    source_to_he: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    for ar_canonical, depth_map in ar_borrow.items():
        for s, depth in depth_map.items():
            prev = source_to_ar[s].get(ar_canonical)
            if prev is None or depth < prev:
                source_to_ar[s][ar_canonical] = depth
    for he_canonical, depth_map in he_borrow.items():
        for s, depth in depth_map.items():
            prev = source_to_he[s].get(he_canonical)
            if prev is None or depth < prev:
                source_to_he[s][he_canonical] = depth

    shared_sources_set = set(source_to_ar.keys()) & set(source_to_he.keys())
    for source in shared_sources_set:
        for ar_c, ar_depth in source_to_ar[source].items():
            for he_c, he_depth in source_to_he[source].items():
                pair = _ensure_pair(ar_c, he_c)
                if not pair:
                    continue
                if "shared_borrowing_source" not in pair.layers:
                    pair.layers.append("shared_borrowing_source")
                prev = pair.sources.get(source)
                if prev is None or (ar_depth, he_depth) < prev:
                    pair.sources[source] = (ar_depth, he_depth)

    match_elapsed = time.monotonic() - t0
    print(f"\n  {len(pair_data)} cognate pairs found")
    print(f"  ⏱ {match_elapsed:.1f}s")

    # ── Load sense embeddings ───────────────────────────────────
    print("\nLoading sense embeddings …")
    with open(SENSES_FILE, "rb") as f:
        senses_data = orjson.loads(f.read())
    embeddings = np.load(EMBEDDINGS_FILE)
    print(f"  {embeddings.shape[0]} senses, dim={embeddings.shape[1]}")

    def _get_senses(lang: str, canonical: str, norm: str) -> list[dict]:
        """Look up senses by canonical form, falling back to norm."""
        lang_senses = senses_data.get(lang, {})
        senses = lang_senses.get(canonical, [])
        if not senses:
            senses = lang_senses.get(norm, [])
        return senses

    def _best_sense_pair(ar_canonical: str, ar_norm: str,
                         he_canonical: str, he_norm: str) -> SenseMatch | None:
        """Find the (ar_sense, he_sense) pair with highest dot product."""
        ar_senses = _get_senses("ar", ar_canonical, ar_norm)
        he_senses = _get_senses("he", he_canonical, he_norm)
        if not ar_senses or not he_senses:
            return None

        ar_idxs = [s["idx"] for s in ar_senses]
        he_idxs = [s["idx"] for s in he_senses]
        ar_emb = embeddings[ar_idxs]  # (A, D)
        he_emb = embeddings[he_idxs]  # (H, D)
        dots = ar_emb @ he_emb.T      # (A, H)

        best = np.unravel_index(dots.argmax(), dots.shape)
        return SenseMatch(
            arabic_sense=ar_senses[best[0]]["gloss"],
            arabic_pos=ar_senses[best[0]]["pos"],
            hebrew_sense=he_senses[best[1]]["gloss"],
            hebrew_pos=he_senses[best[1]]["pos"],
            similarity=round(float(dots[best]), 4),
        )

    # ── Build output ─────────────────────────────────────────────
    print(f"\nWriting {OUTPUT_FILE} …")
    _WIKT = "https://en.wiktionary.org/wiki/"
    skipped = 0
    unsupported_langs: dict[str, int] = {}
    consonant_mismatches = 0
    missing_romanizations = 0
    empty_ancestors = 0
    results: list[CognateEntry] = []
    for (ar_canonical, he_canonical), pair in sorted(pair_data.items()):
        ar_wd = ar_words.get(ar_canonical)
        he_wd = he_words.get(he_canonical)
        ar_norm = ar_wd.norm if ar_wd else ar_canonical
        he_norm = he_wd.norm if he_wd else he_canonical
        if not _get_senses("ar", ar_canonical, ar_norm) or \
           not _get_senses("he", he_canonical, he_norm):
            skipped += 1
            continue
        ar_roman = ar_wd.romanization if ar_wd else ""
        he_roman = he_wd.romanization if he_wd else ""

        entry = CognateEntry(
            arabic=LangEntry(
                canonical=ar_canonical,
                roman=ar_roman,
                glosses=sorted(ar_wd.glosses if ar_wd else []),
                wiktionary=_WIKT + quote(ar_norm) + "#Arabic",
            ),
            hebrew=LangEntry(
                canonical=he_canonical,
                roman=he_roman,
                glosses=sorted(he_wd.glosses if he_wd else []),
                wiktionary=_WIKT + quote(he_norm) + "#Hebrew",
            ),
            match_layers=pair.layers,
            shared_borrowing_sources=None,  # filled in below after LCA
            best_sense_match=_best_sense_pair(ar_canonical, ar_norm, he_canonical, he_norm),
        )
        lcas = _find_lcas(pair.sources, borrow_graph) if pair.sources else []
        if pair.sources:
            lca_set = set(lcas)
            rest = sorted(
                [s for s in pair.sources if s not in lca_set],
                key=lambda s: sum(pair.sources[s]),
            )
            all_sorted = lcas + rest
            entry.shared_borrowing_sources = {
                f"{s[0]}:{s[1]}": pair.sources[s] for s in all_sorted
            }
        lca_sources = [
            SharedSource(
                lang=s[0],
                word=s[1],
                romanization=SharedSource.resolve_romanization(
                    s[0], s[1],
                    template_tr=template_tr_index.get(s),
                    kaikki_roman_index=kaikki_roman_index,
                ),
            )
            for s in lcas
        ] or None
        try:
            ancestor = reconstruct_ancestor(
                entry.arabic.roman, entry.hebrew.roman,
                shared_sources=lca_sources,
            )
            entry.ancestor = str(ancestor)
            pansemitic = ancestor.to_protopansemitic()
            if pansemitic:
                entry.pansemitic_form = pansemitic
        except UnsupportedLanguageError as e:
            unsupported_langs[e.lang] = unsupported_langs.get(e.lang, 0) + 1
        except ConsonantMismatchError:
            consonant_mismatches += 1
        except MissingRomanizationError:
            missing_romanizations += 1
        except EmptyAncestorError:
            empty_ancestors += 1
        results.append(entry)

    with open(OUTPUT_FILE, "wb") as f:
        f.write(orjson.dumps(
            [e.to_dict() for e in results], option=orjson.OPT_INDENT_2,
        ))

    print(f"Writing {CSV_FILE} …")
    with open(CSV_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["arabic", "arabic_romanization", "hebrew", "hebrew_romanization", "pansemitic", "layers"])
        for entry in results:
            writer.writerow([
                entry.arabic.canonical,
                entry.arabic.roman,
                entry.hebrew.canonical,
                entry.hebrew.roman,
                entry.pansemitic_form or "",
                ";".join(entry.match_layers),
            ])

    pansemitic_count = sum(1 for e in results if e.pansemitic_form)
    total_failures = sum(unsupported_langs.values()) + consonant_mismatches + missing_romanizations + empty_ancestors
    print(f"\nDone in {time.monotonic() - t_total:.1f}s total.")
    print(f"  {len(results)} cognate pairs written ({skipped} skipped, no senses)")
    print(f"  {pansemitic_count} pansemitic forms generated")
    print(f"\n  Pansemitic failures ({total_failures} total):")
    print(f"    Consonant count mismatch:     {consonant_mismatches}")
    print(f"    Unsupported source language:   {sum(unsupported_langs.values())}")
    if unsupported_langs:
        for lang, count in sorted(unsupported_langs.items(), key=lambda x: -x[1]):
            print(f"      {lang:20s} {count}")
    print(f"    Missing romanization:          {missing_romanizations}")
    print(f"    Empty ancestor:                {empty_ancestors}")


if __name__ == "__main__":
    main()
