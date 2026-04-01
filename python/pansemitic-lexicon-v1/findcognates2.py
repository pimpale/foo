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

import csv
import os
from urllib.parse import quote
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import orjson

from phonetics import reconstruct_ancestor, construct_pansemitic_form

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
ETYMON_LANG_WORD = re.compile(r"([a-z]{2,}):([^<>\s:]+)")

BORROW_TEMPLATES = {"bor", "der", "lbor", "ubor", "slbor", "borrowed"}

_AR = b'"ar"'
_HE = b'"he"'
_ETYM = b'"etymology_templates"'


def normalize_arabic(text: str) -> str:
    return ARABIC_DIACRITICS.sub("", text).strip()


def normalize_hebrew(text: str) -> str:
    return HEBREW_DIACRITICS.sub("", text).strip()


def _process_semitic_entry(entry, target_lang, normalize_self, normalize_target,
                           cognates, sem_roots, lemma_of, borrow_sources,
                           display_form, roman_map, glosses):
    """Process a single Arabic or Hebrew kaikki entry."""
    word = entry.get("word", "")
    if not word:
        return

    norm = normalize_self(word)
    if norm not in display_form:
        display_form[norm] = word
    if norm not in roman_map:
        for fm in entry.get("forms", []):
            if "romanization" in fm.get("tags", []):
                roman_map[norm] = fm.get("form", "")
                break

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
        for gloss in sense.get("glosses", []):
            glosses[norm].add(gloss)


def _extract_borrowing(entry, lang_code, borrow_graph):
    """Extract borrowing sources from any entry into the global graph."""
    word = entry.get("word", "")
    if not word:
        return
    key = (lang_code, word.lower())
    for tmpl in entry.get("etymology_templates", []):
        if tmpl.get("name", "") not in BORROW_TEMPLATES:
            continue
        args = tmpl.get("args", {})
        src_lang = args.get("2", "")
        src_word = args.get("3", "")
        if (src_lang and src_word
                and src_word not in ("-", "?", "")
                and len(src_word) > 1):
            borrow_graph[key].add((src_lang, src_word.lower()))


def _expand_borrow_transitive(borrow_map, borrow_graph, max_depth=10):
    """Expand borrow sources in-place by walking the full borrowing graph."""
    expanded_count = 0
    for norm, sources in borrow_map.items():
        ancestors = set()
        frontier = set(sources)
        visited = set()
        for _ in range(max_depth):
            next_frontier = set()
            for node in frontier:
                if node in visited:
                    continue
                visited.add(node)
                parents = borrow_graph.get(node, set())
                ancestors.update(parents)
                next_frontier.update(parents)
            frontier = next_frontier - visited
            if not frontier:
                break
        new = ancestors - sources
        if new:
            sources.update(new)
            expanded_count += 1
    return expanded_count


def _process_chunk(filepath_str, start_byte, end_byte):
    """Worker: process one byte-range of the JSONL file, return partial indexes."""
    ar2he_cog = defaultdict(set)
    ar_sem = defaultdict(set)
    ar_lemmas = defaultdict(set)
    ar_borrow = defaultdict(set)
    ar_disp = {}
    ar_roman = {}
    ar_glosses = defaultdict(set)

    he2ar_cog = defaultdict(set)
    he_sem = defaultdict(set)
    he_lemmas = defaultdict(set)
    he_borrow = defaultdict(set)
    he_disp = {}
    he_roman = {}
    he_glosses = defaultdict(set)

    borrow_graph = defaultdict(set)

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
                _process_semitic_entry(
                    entry, "he", normalize_arabic, normalize_hebrew,
                    ar2he_cog, ar_sem, ar_lemmas, ar_borrow,
                    ar_disp, ar_roman, ar_glosses,
                )
            elif lc == "he":
                he_count += 1
                _process_semitic_entry(
                    entry, "ar", normalize_hebrew, normalize_arabic,
                    he2ar_cog, he_sem, he_lemmas, he_borrow,
                    he_disp, he_roman, he_glosses,
                )

            _extract_borrowing(entry, lc, borrow_graph)

    return (
        dict(ar2he_cog), dict(ar_sem), dict(ar_lemmas), dict(ar_borrow),
        ar_disp, ar_roman, dict(ar_glosses),
        dict(he2ar_cog), dict(he_sem), dict(he_lemmas), dict(he_borrow),
        he_disp, he_roman, dict(he_glosses),
        dict(borrow_graph),
        (ar_count, he_count, line_count),
    )


def _merge_set_dicts(target, source):
    for key, values in source.items():
        if key in target:
            target[key] |= values
        else:
            target[key] = set(values)


def build_all_indexes(filepath, num_workers=None):
    """
    Parallel scan of the all-languages kaikki file.
    Only extracts Arabic, Hebrew, and borrowing graph data.
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
        partial_results = [None] * num_workers
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
    ar2he_cog = defaultdict(set)
    ar_sem = defaultdict(set)
    ar_lemmas = defaultdict(set)
    ar_borrow = defaultdict(set)
    ar_disp = {}
    ar_roman = {}
    ar_glosses = defaultdict(set)
    he2ar_cog = defaultdict(set)
    he_sem = defaultdict(set)
    he_lemmas = defaultdict(set)
    he_borrow = defaultdict(set)
    he_disp = {}
    he_roman = {}
    he_glosses = defaultdict(set)
    borrow_graph = defaultdict(set)

    for result in partial_results:
        (p_ar2he, p_ar_sem, p_ar_lem, p_ar_bor, p_ar_disp, p_ar_rom, p_ar_gl,
         p_he2ar, p_he_sem, p_he_lem, p_he_bor, p_he_disp, p_he_rom, p_he_gl,
         p_graph, _) = result

        _merge_set_dicts(ar2he_cog, p_ar2he)
        _merge_set_dicts(ar_sem, p_ar_sem)
        _merge_set_dicts(ar_lemmas, p_ar_lem)
        _merge_set_dicts(ar_borrow, p_ar_bor)
        _merge_set_dicts(ar_glosses, p_ar_gl)
        for k, v in p_ar_disp.items():
            ar_disp.setdefault(k, v)
        for k, v in p_ar_rom.items():
            ar_roman.setdefault(k, v)
        _merge_set_dicts(he2ar_cog, p_he2ar)
        _merge_set_dicts(he_sem, p_he_sem)
        _merge_set_dicts(he_lemmas, p_he_lem)
        _merge_set_dicts(he_borrow, p_he_bor)
        _merge_set_dicts(he_glosses, p_he_gl)
        for k, v in p_he_disp.items():
            he_disp.setdefault(k, v)
        for k, v in p_he_rom.items():
            he_roman.setdefault(k, v)
        _merge_set_dicts(borrow_graph, p_graph)

    ar_n, he_n, lines_n = totals
    print(f"  ... done: {lines_n:,} lines total "
          f"(ar:{ar_n:,} he:{he_n:,}"
          f" graph:{len(borrow_graph):,} nodes)")

    return (
        ar2he_cog, ar_sem, ar_lemmas, ar_borrow, ar_disp, ar_roman, ar_glosses,
        he2ar_cog, he_sem, he_lemmas, he_borrow, he_disp, he_roman, he_glosses,
        borrow_graph,
    )


def _make_bidirectional(graph):
    """Add reverse edges so A→B also implies B→A."""
    reverse = defaultdict(set)
    for word, targets in graph.items():
        for t in targets:
            reverse[t].add(word)
    for word, targets in reverse.items():
        if word in graph:
            graph[word] |= targets
        else:
            graph[word] = targets


def main():
    t_total = time.monotonic()

    false_pos = set()
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
    (
        ar2he_cog, ar_sem, ar_lemmas, ar_borrow, ar_disp, ar_roman, ar_glosses,
        he2ar_cog, he_sem, he_lemmas, he_borrow, he_disp, he_roman, he_glosses,
        borrow_graph,
    ) = build_all_indexes(ALL_WORDS_FILE)
    scan_time = time.monotonic() - t0

    print(f"\n  Arabic:  {len(ar2he_cog)} cognate refs, {len(ar_sem)} sem roots, "
          f"{len(ar_lemmas)} lemma links, {len(ar_borrow)} borrow sources")
    print(f"  Hebrew:  {len(he2ar_cog)} cognate refs, {len(he_sem)} sem roots, "
          f"{len(he_lemmas)} lemma links, {len(he_borrow)} borrow sources")
    print(f"  Borrow graph: {len(borrow_graph)} nodes")

    _make_bidirectional(ar2he_cog)
    _make_bidirectional(he2ar_cog)
    print(f"  After bidirectional: ar cognate refs {len(ar2he_cog)}, "
          f"he cognate refs {len(he2ar_cog)}")

    ar_expanded = _expand_borrow_transitive(ar_borrow, borrow_graph)
    he_expanded = _expand_borrow_transitive(he_borrow, borrow_graph)
    print(f"  Transitive expansion: {ar_expanded} ar words, {he_expanded} he words")
    print(f"  ⏱ {scan_time:.1f}s")

    # ── Direct cognate matching ──────────────────────────────────
    print("\nMatching cognates directly …")
    t0 = time.monotonic()

    # Collect all cognate pairs from the three layers into a unified dict
    # Key: (ar_norm, he_norm) → {layers, shared_roots, shared_sources}
    _JUNK = {"", "-", "?"}
    pair_data = {}

    def _is_root_or_single(word):
        """Roots have 2+ hyphens (regular or maqaf), single letters have 1 grapheme."""
        if len(word) <= 1:
            return True
        hyphens = word.count("-") + word.count("\u05be")
        return hyphens >= 2

    def _ensure_pair(ar_n, he_n):
        if ar_n in _JUNK or he_n in _JUNK:
            return None
        if _is_root_or_single(ar_n) or _is_root_or_single(he_n):
            return None
        key = (ar_n, he_n)
        if key not in pair_data:
            pair_data[key] = {"layers": [], "roots": [], "sources": []}
        return pair_data[key]

    # Layer 1: Direct cognate references
    # ar2he_cog: ar_norm → {he_norms}
    for ar_norm, he_norms in ar2he_cog.items():
        for he_norm in he_norms:
            if (ar_norm, he_norm) in false_pos:
                continue
            # Also check via lemmas
            ar_forms = {ar_norm} | ar_lemmas.get(ar_norm, set())
            he_forms = {he_norm} | he_lemmas.get(he_norm, set())
            for af in ar_forms:
                for hf in he_forms:
                    if hf in ar2he_cog.get(af, set()):
                        d = _ensure_pair(ar_norm, he_norm)
                        if d and "direct_cognate_ar→he" not in d["layers"]:
                            d["layers"].append("direct_cognate_ar→he")
                        break
                else:
                    continue
                break

    # he2ar_cog: he_norm → {ar_norms}
    for he_norm, ar_norms in he2ar_cog.items():
        for ar_norm in ar_norms:
            if (ar_norm, he_norm) in false_pos:
                continue
            d = _ensure_pair(ar_norm, he_norm)
            if d and "direct_cognate_he→ar" not in d["layers"]:
                d["layers"].append("direct_cognate_he→ar")

    # Layer 2: Shared Proto-Semitic root
    # Invert: root → {ar_norms} and root → {he_norms}
    root_to_ar = defaultdict(set)
    root_to_he = defaultdict(set)
    for ar_norm, roots in ar_sem.items():
        for r in roots:
            root_to_ar[r].add(ar_norm)
    for he_norm, roots in he_sem.items():
        for r in roots:
            root_to_he[r].add(he_norm)

    shared_roots_set = set(root_to_ar.keys()) & set(root_to_he.keys())
    for root in shared_roots_set:
        for ar_norm in root_to_ar[root]:
            for he_norm in root_to_he[root]:
                if (ar_norm, he_norm) in false_pos:
                    continue
                d = _ensure_pair(ar_norm, he_norm)
                if not d:
                    continue
                if "shared_proto_semitic_root" not in d["layers"]:
                    d["layers"].append("shared_proto_semitic_root")
                if root not in d["roots"]:
                    d["roots"].append(root)

    # Layer 3: Shared borrowing source
    # Invert: source → {ar_norms} and source → {he_norms}
    source_to_ar = defaultdict(set)
    source_to_he = defaultdict(set)
    for ar_norm, sources in ar_borrow.items():
        for s in sources:
            source_to_ar[s].add(ar_norm)
    for he_norm, sources in he_borrow.items():
        for s in sources:
            source_to_he[s].add(he_norm)

    shared_sources_set = set(source_to_ar.keys()) & set(source_to_he.keys())
    for source in shared_sources_set:
        for ar_norm in source_to_ar[source]:
            for he_norm in source_to_he[source]:
                if (ar_norm, he_norm) in false_pos:
                    continue
                d = _ensure_pair(ar_norm, he_norm)
                if not d:
                    continue
                if "shared_borrowing_source" not in d["layers"]:
                    d["layers"].append("shared_borrowing_source")
                src_str = f"{source[0]}:{source[1]}"
                if src_str not in d["sources"]:
                    d["sources"].append(src_str)

    match_elapsed = time.monotonic() - t0
    print(f"\n  {len(pair_data)} cognate pairs found")
    print(f"  ⏱ {match_elapsed:.1f}s")

    # ── Load sense embeddings ───────────────────────────────────
    print("\nLoading sense embeddings …")
    with open(SENSES_FILE, "rb") as f:
        senses_data = orjson.loads(f.read())
    embeddings = np.load(EMBEDDINGS_FILE)
    print(f"  {embeddings.shape[0]} senses, dim={embeddings.shape[1]}")

    def _best_sense_pair(ar_n, he_n):
        """Find the (ar_sense, he_sense) pair with highest dot product."""
        ar_senses = senses_data.get("ar", {}).get(ar_n, [])
        he_senses = senses_data.get("he", {}).get(he_n, [])
        if not ar_senses or not he_senses:
            return None

        ar_idxs = [s["idx"] for s in ar_senses]
        he_idxs = [s["idx"] for s in he_senses]
        ar_emb = embeddings[ar_idxs]  # (A, D)
        he_emb = embeddings[he_idxs]  # (H, D)
        dots = ar_emb @ he_emb.T      # (A, H)

        best = np.unravel_index(dots.argmax(), dots.shape)
        return {
            "arabic_sense": ar_senses[best[0]]["gloss"],
            "arabic_pos": ar_senses[best[0]]["pos"],
            "hebrew_sense": he_senses[best[1]]["gloss"],
            "hebrew_pos": he_senses[best[1]]["pos"],
            "similarity": round(float(dots[best]), 4),
        }

    # ── Build output ─────────────────────────────────────────────
    print(f"\nWriting {OUTPUT_FILE} …")
    _WIKT = "https://en.wiktionary.org/wiki/"
    ar_senses_keys = set(senses_data.get("ar", {}).keys())
    he_senses_keys = set(senses_data.get("he", {}).keys())
    skipped = 0
    results = []
    for (ar_norm, he_norm), data in sorted(pair_data.items()):
        if ar_norm not in ar_senses_keys or he_norm not in he_senses_keys:
            skipped += 1
            continue
        ar_word = ar_disp.get(ar_norm, ar_norm)
        he_word = he_disp.get(he_norm, he_norm)
        entry = {
            "arabic": ar_word,
            "arabic_roman": ar_roman.get(ar_norm, ""),
            "arabic_glosses": sorted(ar_glosses.get(ar_norm, set())),
            "arabic_wiktionary": _WIKT + quote(ar_word) + "#Arabic",
            "hebrew": he_word,
            "hebrew_roman": he_roman.get(he_norm, ""),
            "hebrew_glosses": sorted(he_glosses.get(he_norm, set())),
            "hebrew_wiktionary": _WIKT + quote(he_word) + "#Hebrew",
            "match_layers": data["layers"],
        }
        if data["roots"]:
            entry["shared_proto_semitic_roots"] = sorted(data["roots"])
        if data["sources"]:
            entry["shared_borrowing_sources"] = sorted(data["sources"])
        best = _best_sense_pair(ar_norm, he_norm)
        if best:
            entry["best_sense_match"] = best
        ancestor = reconstruct_ancestor(
            entry["arabic_roman"], entry["hebrew_roman"],
            shared_roots=data["roots"] or None,
            shared_sources=data["sources"] or None,
        )
        if ancestor:
            entry["ancestor"] = ancestor
            pansemitic = construct_pansemitic_form(ancestor)
            if pansemitic:
                entry["pansemitic_form"] = pansemitic
        results.append(entry)

    with open(OUTPUT_FILE, "wb") as f:
        f.write(orjson.dumps(results, option=orjson.OPT_INDENT_2))

    print(f"Writing {CSV_FILE} …")
    with open(CSV_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["arabic", "arabic_romanization", "hebrew", "hebrew_romanization", "layers"])
        for entry in results:
            writer.writerow([
                entry["arabic"],
                entry["arabic_roman"],
                entry["hebrew"],
                entry["hebrew_roman"],
                ";".join(entry["match_layers"]),
            ])

    print(f"\nDone in {time.monotonic() - t_total:.1f}s total.")
    print(f"  {len(results)} cognate pairs written ({skipped} skipped, no senses)")


if __name__ == "__main__":
    main()
