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
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Self

import numpy as np
import orjson

from loss import LossBreakdown, triplet_loss_breakdown
from kaikki import PartialSource, SharedSource, canonical_from_entry
from reconstruction import (
    PansemiticWord,
    reconstruct_ancestor,
    word_from_sharedsource,
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
GOOD_OUTPUT_FILE = Path("cognates_good.json")
GOOD_CSV_FILE = Path("cognates_good.csv")
GOOD_SIMILARITY_THRESHOLD = 0.83
ROMANIZATION_TIER_DIFFS_FILE = Path("romanization_tier_differences.csv")
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


_AR = b'"ar"'
_HE = b'"he"'
_ETYM = b'"etymology_templates"'
_IPA = b'"ipa"'


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

    def absorb(self, other: Self) -> None:
        """Fold another partial WordData for the same canonical into self."""
        if not self.romanization:
            self.romanization = other.romanization
        self.glosses |= other.glosses
        self.cognates |= other.cognates
        self.lemma_of |= other.lemma_of
        self.borrow_sources |= other.borrow_sources


@dataclass
class ScanCounts:
    """Per-chunk counts used for progress reporting."""
    ar: int = 0
    he: int = 0
    lines: int = 0

    def add(self, other: Self) -> None:
        self.ar += other.ar
        self.he += other.he
        self.lines += other.lines


@dataclass
class KaikkiScanResult:
    """All indexes built from one (or merged across many) kaikki scans."""
    ar_words: dict[str, WordData] = field(default_factory=dict)
    he_words: dict[str, WordData] = field(default_factory=dict)
    borrow_graph: dict[tuple[str, str], set[tuple[str, str]]] = field(
        default_factory=lambda: defaultdict(set)
    )
    template_tr_index: dict[tuple[str, str], str] = field(default_factory=dict)
    kaikki_partials: dict[tuple[str, str], PartialSource] = field(default_factory=dict)
    counts: ScanCounts = field(default_factory=ScanCounts)

    def merge(self, other: Self) -> None:
        self._merge_words(self.ar_words, other.ar_words)
        self._merge_words(self.he_words, other.he_words)
        self._merge_setvalued(self.borrow_graph, other.borrow_graph)
        self._merge_first_wins(self.template_tr_index, other.template_tr_index)
        self._merge_first_wins(self.kaikki_partials, other.kaikki_partials)
        self.counts.add(other.counts)

    @staticmethod
    def _merge_words(
        target: dict[str, WordData], source: dict[str, WordData],
    ) -> None:
        """Combine partial WordData dicts on shared canonical keys."""
        for canonical, wd in source.items():
            existing = target.get(canonical)
            if existing is None:
                target[canonical] = wd
            else:
                existing.absorb(wd)

    @staticmethod
    def _merge_setvalued(target: dict, source: dict) -> None:
        """Merge dicts whose values are sets — union per shared key."""
        for key, values in source.items():
            if key in target:
                target[key] |= values
            else:
                target[key] = set(values)

    @staticmethod
    def _merge_first_wins(target: dict, source: dict) -> None:
        """Merge plain-value dicts, keeping whichever value was inserted first."""
        for key, value in source.items():
            target.setdefault(key, value)


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
    ipa: str | None = None  # native kaikki IPA if available, else derived from roman

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
    loss: LossBreakdown | None = None
    pansemitic_failure: str | None = None  # populated iff pansemitic_form is None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict, recursively omitting None fields."""
        def _strip(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: _strip(v) for k, v in obj.items() if v is not None}
            if isinstance(obj, list):
                return [_strip(x) for x in obj]
            return obj
        d = dataclasses.asdict(self)
        return {k: _strip(v) for k, v in d.items() if v is not None}


@dataclass
class RomanizationTierObservation:
    lang: str
    word: str
    tier1: str | None
    tier2: str | None
    tier3: str | None
    selected: str | None
    uses: int = 0


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

    # Many Semitic entries have an unpointed ``word`` field but a pointed
    # canonical form in ``forms``. Other entries may cite either one in
    # etymology templates, so index both as aliases for the same node.
    node_keys = {(lang_code, word)}
    canonical = canonical_from_entry(entry)
    if canonical and canonical != word:
        node_keys.add((lang_code, canonical))

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
                for key in node_keys:
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
                        src_key = (src_lang, src_word)
                        for key in node_keys:
                            borrow_graph[key].add(src_key)


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


def _process_chunk(
    filepath_str: str, start_byte: int, end_byte: int,
) -> KaikkiScanResult:
    """Worker: process one byte-range of the JSONL file, return partial indexes."""
    result = KaikkiScanResult()

    with open(filepath_str, "rb") as f:
        if start_byte > 0:
            f.seek(start_byte)
            f.readline()

        while f.tell() < end_byte:
            line = f.readline()
            if not line:
                break

            result.counts.lines += 1

            if not (_AR in line or _HE in line or _ETYM in line or _IPA in line):
                continue

            entry = orjson.loads(line)
            lc = entry.get("lang_code", "")

            if lc == "ar":
                result.counts.ar += 1
                canonical = canonical_from_entry(entry)
                _process_semitic_entry(
                    entry, "he", normalize_arabic, normalize_hebrew, result.ar_words,
                    canonical=canonical,
                )
            elif lc == "he":
                result.counts.he += 1
                canonical = canonical_from_entry(entry)
                _process_semitic_entry(
                    entry, "ar", normalize_hebrew, normalize_arabic, result.he_words,
                    canonical=canonical,
                )

            # Build a PartialSource for any entry (feeds tier3 + IPA lookup).
            partial = PartialSource.from_kaikki_entry(entry)
            if partial is not None:
                key_aliases = {(partial.lang, partial.word)}
                raw_word = entry.get("word", "")
                if raw_word and raw_word != partial.word:
                    key_aliases.add((partial.lang, raw_word))
                for key in key_aliases:
                    result.kaikki_partials.setdefault(key, partial)

            _extract_borrowing(entry, lc, result.borrow_graph, result.template_tr_index)

    # defaultdict → plain dict for cross-process pickling cleanliness.
    result.borrow_graph = dict(result.borrow_graph)
    return result


def build_all_indexes(
    filepath: Path, num_workers: int | None = None,
) -> KaikkiScanResult:
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
        partial_results: list[KaikkiScanResult | None] = [None] * num_workers
        done_count = 0

        for future in as_completed(futures):
            idx = futures[future]
            partial_results[idx] = future.result()
            done_count += 1
            print(f"  ... {done_count}/{num_workers} chunks done",
                  file=sys.stderr)

    print("  merging indexes …", file=sys.stderr)
    merged = KaikkiScanResult()
    for partial in partial_results:
        assert partial is not None
        merged.merge(partial)

    c = merged.counts
    print(f"  ... done: {c.lines:,} lines total "
          f"(ar:{c.ar:,} he:{c.he:,}"
          f" graph:{len(merged.borrow_graph):,} nodes)"
          f"\n  ... {len(merged.template_tr_index):,} template transliterations,"
          f" {len(merged.kaikki_partials):,} kaikki partial sources")

    return merged


def _make_cognate_index(words: dict[str, WordData]) -> dict[str, set[tuple[str, str]]]:
    """Extract cognate refs from WordData, keyed by canonical form.

    Values are sets of (normalized_target, raw_target) tuples.
    """
    return {canonical: set(wd.cognates) for canonical, wd in words.items() if wd.cognates}


def _make_borrow_index(
    words: dict[str, WordData],
    lang: str,
    cross_lang_source_targets: set[tuple[str, str]],
) -> dict[str, dict[tuple[str, str], int]]:
    """Extract borrow sources from WordData, keyed by canonical form.

    Initially all direct sources have depth 0; _expand_borrow_transitive
    adds transitive ancestors with increasing depth.
    """
    out: dict[str, dict[tuple[str, str], int]] = {}
    for canonical, wd in words.items():
        sources = {s: 0 for s in wd.borrow_sources}
        self_nodes = {(lang, canonical)}
        if wd.norm and wd.norm != canonical:
            self_nodes.add((lang, wd.norm))

        # Treat the Arabic/Hebrew lexeme itself as a graph node so the
        # existing transitive-LCA pipeline can return a direct donor word
        # (e.g. Hebrew <- Arabic) instead of always climbing to a deeper
        # shared ancestor. Only seed these self-nodes when the lexeme is
        # actually participating in an etymology chain itself, or when the
        # opposite side explicitly cites it as a source.
        if wd.borrow_sources or (self_nodes & cross_lang_source_targets):
            for node in self_nodes:
                sources[node] = 0

        if sources:
            out[canonical] = sources
    return out


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

    # Sort by combined depth; break ties on (lang, word) for determinism.
    lcas.sort(key=lambda s: (sum(sources[s]), s[0], s[1]))
    return lcas


def _availability_key(obs: RomanizationTierObservation) -> str:
    labels = []
    if obs.tier1:
        labels.append("1")
    if obs.tier2:
        labels.append("2")
    if obs.tier3:
        labels.append("3")
    return "+".join(labels) if labels else "none"


def _tiers_differ(obs: RomanizationTierObservation) -> bool:
    available = [v for v in (obs.tier1, obs.tier2, obs.tier3) if v]
    return len(available) >= 2 and len(set(available)) > 1


def _print_loss_statistics(results: list[CognateEntry]) -> None:
    losses = [entry.loss for entry in results if entry.loss is not None]
    skipped = len(results) - len(losses)

    print("\n  Loss statistics:")
    if not losses:
        print(f"    No scorable triplets ({skipped} skipped, missing loss)")
        return

    joint = [loss.joint for loss in losses]
    arabic = [loss.arabic for loss in losses]
    hebrew = [loss.hebrew for loss in losses]

    print(f"    Scored triplets: {len(losses)} ({skipped} skipped, missing loss)")
    print(f"    Joint mean:      {statistics.mean(joint):.4f}")
    print(f"    Joint median:    {statistics.median(joint):.4f}")
    if len(joint) > 1:
        print(f"    Joint stdev:     {statistics.stdev(joint):.4f}")
    else:
        print("    Joint stdev:     n/a")
    print(f"    Joint min:       {min(joint):.4f}")
    print(f"    Joint max:       {max(joint):.4f}")
    print(f"    Arabic mean:     {statistics.mean(arabic):.4f}")
    print(f"    Hebrew mean:     {statistics.mean(hebrew):.4f}")

    buckets = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, float("inf")]
    counts = [0] * (len(buckets) - 1)
    for loss in joint:
        for i in range(len(buckets) - 1):
            if buckets[i] <= loss < buckets[i + 1]:
                counts[i] += 1
                break
    print("    Joint distribution:")
    for i, count in enumerate(counts):
        upper = (
            "inf" if buckets[i + 1] == float("inf")
            else f"{buckets[i + 1]:.2f}"
        )
        print(f"      [{buckets[i]:.2f}, {upper})  {count}")


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
    scan = build_all_indexes(ALL_WORDS_FILE)
    ar_words = scan.ar_words
    he_words = scan.he_words
    borrow_graph = scan.borrow_graph
    template_tr_index = scan.template_tr_index
    kaikki_partials = scan.kaikki_partials
    scan_time = time.monotonic() - t0

    # Build standalone cognate/borrow indexes keyed by canonical form
    ar2he_cog = _make_cognate_index(ar_words)
    he2ar_cog = _make_cognate_index(he_words)
    ar_source_targets = {
        source for wd in he_words.values() for source in wd.borrow_sources
        if source[0] == "ar"
    }
    he_source_targets = {
        source for wd in ar_words.values() for source in wd.borrow_sources
        if source[0] == "he"
    }
    ar_borrow = _make_borrow_index(ar_words, "ar", ar_source_targets)
    he_borrow = _make_borrow_index(he_words, "he", he_source_targets)

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

    def _surface_source(lang: str, canonical: str, norm: str, roman: str) -> SharedSource:
        """Build a SharedSource for the Arabic/Hebrew surface form itself."""
        partial = kaikki_partials.get((lang, canonical)) or kaikki_partials.get((lang, norm))
        return SharedSource(
            lang=lang,
            word=canonical,
            kaikki_romanization=roman or None,
            ipa=partial.ipa if partial else None,
        )

    def _surface_ipa(source: SharedSource) -> str | None:
        """Resolve a surface SharedSource to normalized IPA."""
        try:
            return word_from_sharedsource(source).to_ipa() or None
        except ReconstructionError:
            return None

    # ── Build output ─────────────────────────────────────────────
    print(f"\nWriting {OUTPUT_FILE} …")
    _WIKT = "https://en.wiktionary.org/wiki/"
    skipped = 0
    unsupported_langs: dict[str, int] = {}
    consonant_mismatches = 0
    missing_romanizations = 0
    empty_ancestors = 0
    romanization_tier_obs: dict[tuple[str, str], RomanizationTierObservation] = {}
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

        ar_surface = _surface_source("ar", ar_canonical, ar_norm, ar_roman)
        he_surface = _surface_source("he", he_canonical, he_norm, he_roman)
        ar_ipa = _surface_ipa(ar_surface)
        he_ipa = _surface_ipa(he_surface)

        entry = CognateEntry(
            arabic=LangEntry(
                canonical=ar_canonical,
                roman=ar_roman,
                glosses=sorted(ar_wd.glosses if ar_wd else []),
                wiktionary=_WIKT + quote(ar_norm) + "#Arabic",
                ipa=ar_ipa,
            ),
            hebrew=LangEntry(
                canonical=he_canonical,
                roman=he_roman,
                glosses=sorted(he_wd.glosses if he_wd else []),
                wiktionary=_WIKT + quote(he_norm) + "#Hebrew",
                ipa=he_ipa,
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
        lca_sources: list[SharedSource] = []
        for s in lcas:
            citation = SharedSource.from_citation(s[0], s[1], template_tr_index.get(s))
            src = SharedSource.resolve(
                lang=s[0], word=s[1],
                entry=kaikki_partials.get(s),
                citation=citation,
            )
            key = (s[0], s[1])
            if key not in romanization_tier_obs:
                romanization_tier_obs[key] = RomanizationTierObservation(
                    lang=s[0],
                    word=s[1],
                    tier1=src.tier1,
                    tier2=src.tier2,
                    tier3=src.tier3,
                    selected=src.romanization,
                    uses=1,
                )
            else:
                romanization_tier_obs[key].uses += 1
            lca_sources.append(src)
        try:
            ancestor_word = (
                word_from_sharedsource(lca_sources[0]) if lca_sources else None
            )
            ancestor = reconstruct_ancestor(
                entry.arabic.roman, entry.hebrew.roman,
                ancestor=ancestor_word,
            )
            entry.ancestor = str(ancestor)
            pansemitic_word = PansemiticWord.from_word(ancestor)
            pansemitic = pansemitic_word.to_protosemitic_convention()
            if pansemitic:
                entry.pansemitic_form = pansemitic
                if ar_ipa and he_ipa:
                    entry.loss = triplet_loss_breakdown(
                        pansemitic_word.to_ipa(),
                        ar_ipa,
                        he_ipa,
                    )
            else:
                entry.pansemitic_failure = "empty_pansemitic"
        except UnsupportedLanguageError as e:
            unsupported_langs[e.lang] = unsupported_langs.get(e.lang, 0) + 1
            entry.pansemitic_failure = f"unsupported_language:{e.lang}"
        except ConsonantMismatchError as e:
            consonant_mismatches += 1
            entry.pansemitic_failure = (
                f"consonant_mismatch:ar={e.ar_count},he={e.he_count}"
            )
        except MissingRomanizationError as e:
            missing_romanizations += 1
            entry.pansemitic_failure = f"missing_romanization:{e.missing}"
        except EmptyAncestorError:
            empty_ancestors += 1
            entry.pansemitic_failure = "empty_ancestor"
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

    good_results = [
        e for e in results
        if e.pansemitic_form
        and e.best_sense_match is not None
        and e.best_sense_match.similarity > GOOD_SIMILARITY_THRESHOLD
    ]
    print(f"Writing {GOOD_OUTPUT_FILE} ({len(good_results)} entries) …")
    with open(GOOD_OUTPUT_FILE, "wb") as f:
        f.write(orjson.dumps(
            [e.to_dict() for e in good_results], option=orjson.OPT_INDENT_2,
        ))
    print(f"Writing {GOOD_CSV_FILE} …")
    with open(GOOD_CSV_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["arabic", "arabic_romanization", "hebrew", "hebrew_romanization", "pansemitic", "meaning"])
        for entry in good_results:
            writer.writerow([
                entry.arabic.canonical,
                entry.arabic.roman,
                entry.hebrew.canonical,
                entry.hebrew.roman,
                entry.pansemitic_form or "",
                entry.best_sense_match.arabic_sense if entry.best_sense_match else "",
            ])

    tier1_available = sum(1 for obs in romanization_tier_obs.values() if obs.tier1)
    tier2_available = sum(1 for obs in romanization_tier_obs.values() if obs.tier2)
    tier3_available = sum(1 for obs in romanization_tier_obs.values() if obs.tier3)
    tier_availability_counts: dict[str, int] = defaultdict(int)
    for obs in romanization_tier_obs.values():
        tier_availability_counts[_availability_key(obs)] += 1
    tier_diff_rows = sorted(
        [obs for obs in romanization_tier_obs.values() if _tiers_differ(obs)],
        key=lambda obs: (obs.lang, obs.word),
    )

    pansemitic_count = sum(1 for e in results if e.pansemitic_form)
    total_failures = sum(unsupported_langs.values()) + consonant_mismatches + missing_romanizations + empty_ancestors
    print(f"\nDone in {time.monotonic() - t_total:.1f}s total.")
    print(f"  {len(results)} cognate pairs written ({skipped} skipped, no senses)")
    print(f"  {pansemitic_count} pansemitic forms generated")
    print("\n  Romanization tier inspection:")
    print(f"    {len(romanization_tier_obs)} unique shared-source words "
          f"({sum(obs.uses for obs in romanization_tier_obs.values())} total lookups)")
    print(f"    Tier 1 available:              {tier1_available}")
    print(f"    Tier 2 available:              {tier2_available}")
    print(f"    Tier 3 available:              {tier3_available}")
    print("    Availability combinations:")
    for combo, count in sorted(tier_availability_counts.items()):
        print(f"      {combo:4s} {count}")
    print(f"    Differing tier rows:           {len(tier_diff_rows)}")

    print(f"\n  Pansemitic failures ({total_failures} total):")
    print(f"    Consonant count mismatch:     {consonant_mismatches}")
    print(f"    Unsupported source language:   {sum(unsupported_langs.values())}")
    if unsupported_langs:
        for lang, count in sorted(unsupported_langs.items(), key=lambda x: -x[1]):
            print(f"      {lang:20s} {count}")
    print(f"    Missing romanization:          {missing_romanizations}")
    print(f"    Empty ancestor:                {empty_ancestors}")
    _print_loss_statistics(results)


if __name__ == "__main__":
    main()
