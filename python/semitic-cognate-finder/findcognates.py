#!/usr/bin/env python3
"""
Semitic Cognate Finder

For each word in english-base-forms.txt, finds Arabic-Hebrew cognate pairs
by cross-referencing Wiktionary (kaikki) etymology data.

Three matching layers:
  Layer 1: Explicit cognate references in Arabic/Hebrew etymology templates
  Layer 2: Shared Proto-Semitic root matching
  Layer 3: Shared borrowing source — direct or via transitive chain through
           a global etymology graph (e.g. Arabic←French←Latin→Hebrew)
"""

import csv
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import orjson

DATA_DIR = Path("data")
ALL_WORDS_FILE = DATA_DIR / "kaikki.org-dictionary-all-words.jsonl"
WORDS_FILE = Path("Oxford-3000.txt")
OUTPUT_FILE = Path("cognates.json")
CSV_FILE = Path("cognates.csv")
FALSE_POSITIVES_FILE = Path("false-positives.txt")

ARABIC_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670\u0640]")
HEBREW_DIACRITICS = re.compile(r"[\u0591-\u05BD\u05BF-\u05C7]")
LATIN_TAIL = re.compile(r"\s*[a-zA-Z].*$")

ETYMON_SEM_PRO = re.compile(r"sem-pro:([^<>\s]+)")
ETYMON_LANG_WORD = re.compile(r"([a-z]{2,}):([^<>\s:]+)")

BORROW_TEMPLATES = {"bor", "der", "lbor", "ubor", "slbor", "borrowed"}

_AR = b'"ar"'
_HE = b'"he"'
_EN = b'"en"'
_ETYM = b'"etymology_templates"'


def normalize_arabic(text: str) -> str:
    return ARABIC_DIACRITICS.sub("", text).strip()


def normalize_hebrew(text: str) -> str:
    return HEBREW_DIACRITICS.sub("", text).strip()


def _clean_translation_word(text: str) -> str:
    cleaned = LATIN_TAIL.sub("", text).strip()
    return cleaned if cleaned else text.strip()


def _process_semitic_entry(entry, target_lang, normalize_self, normalize_target,
                           cognates, sem_roots, lemma_of, borrow_sources,
                           synonyms, display_form, roman_map):
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
        for syn in sense.get("synonyms", []):
            syn_word = syn.get("word", "")
            if syn_word:
                synonyms[norm].add(normalize_self(syn_word))


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


def _extract_gloss_candidates(entry, word_set, normalize_fn, gloss_candidates):
    """Parse glosses from an ar/he entry to find English word matches."""
    word = entry.get("word", "")
    if not word:
        return
    pos = entry.get("pos", "")
    if pos not in ("noun", "verb", "adj"):
        return
    norm = normalize_fn(word)
    for sense in entry.get("senses", []):
        for gloss in sense.get("glosses", []):
            for chunk in gloss.split(","):
                chunk = chunk.strip().lower()
                if pos == "verb":
                    if chunk.startswith("to "):
                        eng = chunk[3:].strip()
                        if eng in word_set:
                            gloss_candidates[eng].add(norm)
                else:
                    if chunk in word_set:
                        gloss_candidates[chunk].add(norm)


def _process_english_entry(entry, word_set, translations):
    """Process a single English kaikki entry for translations."""
    word = entry.get("word", "").lower()
    if word not in word_set:
        return

    translations.setdefault(word, {"ar": {}, "he": {}})
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


def _process_chunk(filepath_str, start_byte, end_byte, word_set):
    """Worker: process one byte-range of the JSONL file, return partial indexes."""
    ar2he_cog = defaultdict(set)
    ar_sem = defaultdict(set)
    ar_lemmas = defaultdict(set)
    ar_borrow = defaultdict(set)
    ar_synonyms = defaultdict(set)
    ar_disp = {}
    ar_roman = {}

    he2ar_cog = defaultdict(set)
    he_sem = defaultdict(set)
    he_lemmas = defaultdict(set)
    he_borrow = defaultdict(set)
    he_synonyms = defaultdict(set)
    he_disp = {}
    he_roman = {}

    borrow_graph = defaultdict(set)
    ar_gloss_cands = defaultdict(set)
    he_gloss_cands = defaultdict(set)
    en_trans = {}

    ar_count = he_count = en_count = line_count = 0

    with open(filepath_str, "rb") as f:
        if start_byte > 0:
            f.seek(start_byte)
            f.readline()

        while f.tell() < end_byte:
            line = f.readline()
            if not line:
                break

            line_count += 1

            if not (_AR in line or _HE in line or _EN in line
                    or _ETYM in line):
                continue

            entry = orjson.loads(line)
            lc = entry.get("lang_code", "")

            if lc == "ar":
                ar_count += 1
                _process_semitic_entry(
                    entry, "he", normalize_arabic, normalize_hebrew,
                    ar2he_cog, ar_sem, ar_lemmas, ar_borrow, ar_synonyms,
                    ar_disp, ar_roman,
                )
                _extract_gloss_candidates(
                    entry, word_set, normalize_arabic, ar_gloss_cands)
            elif lc == "he":
                he_count += 1
                _process_semitic_entry(
                    entry, "ar", normalize_hebrew, normalize_arabic,
                    he2ar_cog, he_sem, he_lemmas, he_borrow, he_synonyms,
                    he_disp, he_roman,
                )
                _extract_gloss_candidates(
                    entry, word_set, normalize_hebrew, he_gloss_cands)
            elif lc == "en":
                en_count += 1
                _process_english_entry(entry, word_set, en_trans)

            _extract_borrowing(entry, lc, borrow_graph)

    return (
        dict(ar2he_cog), dict(ar_sem), dict(ar_lemmas), dict(ar_borrow),
        dict(ar_synonyms), ar_disp, ar_roman,
        dict(he2ar_cog), dict(he_sem), dict(he_lemmas), dict(he_borrow),
        dict(he_synonyms), he_disp, he_roman,
        dict(borrow_graph), dict(ar_gloss_cands), dict(he_gloss_cands),
        en_trans,
        (ar_count, he_count, en_count, line_count),
    )


def _merge_set_dicts(target, source):
    for key, values in source.items():
        if key in target:
            target[key] |= values
        else:
            target[key] = set(values)


def _merge_translations(target, source):
    for word, trans in source.items():
        if word not in target:
            target[word] = trans
        else:
            for lang in ("ar", "he"):
                for norm, info in trans[lang].items():
                    if norm not in target[word][lang]:
                        target[word][lang][norm] = info


def build_all_indexes(filepath, word_set, num_workers=None):
    """
    Parallel scan of the all-languages kaikki file using ProcessPoolExecutor.
    Splits the file into byte-range chunks processed by separate workers,
    then merges the partial indexes.
    """
    if num_workers is None:
        num_workers = os.cpu_count() or 1

    file_size = filepath.stat().st_size
    chunk_size = file_size // num_workers

    chunks = []
    for i in range(num_workers):
        start = i * chunk_size
        end = file_size if i == num_workers - 1 else (i + 1) * chunk_size
        chunks.append((str(filepath), start, end, word_set))

    print(f"  dispatching {num_workers} workers "
          f"(~{chunk_size // (1024*1024)} MB/chunk) …", file=sys.stderr)

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_process_chunk, *c): i
                   for i, c in enumerate(chunks)}
        partial_results = [None] * num_workers
        done_count = 0
        totals = [0, 0, 0, 0]

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
    ar_synonyms = defaultdict(set)
    ar_disp = {}
    ar_roman = {}
    he2ar_cog = defaultdict(set)
    he_sem = defaultdict(set)
    he_lemmas = defaultdict(set)
    he_borrow = defaultdict(set)
    he_synonyms = defaultdict(set)
    he_disp = {}
    he_roman = {}
    borrow_graph = defaultdict(set)
    ar_gloss_cands = defaultdict(set)
    he_gloss_cands = defaultdict(set)
    en_trans = {}

    for result in partial_results:
        (p_ar2he, p_ar_sem, p_ar_lem, p_ar_bor, p_ar_syn, p_ar_disp, p_ar_rom,
         p_he2ar, p_he_sem, p_he_lem, p_he_bor, p_he_syn, p_he_disp, p_he_rom,
         p_graph, p_ar_gloss, p_he_gloss,
         p_en, _) = result

        _merge_set_dicts(ar2he_cog, p_ar2he)
        _merge_set_dicts(ar_sem, p_ar_sem)
        _merge_set_dicts(ar_lemmas, p_ar_lem)
        _merge_set_dicts(ar_borrow, p_ar_bor)
        _merge_set_dicts(ar_synonyms, p_ar_syn)
        for k, v in p_ar_disp.items():
            ar_disp.setdefault(k, v)
        for k, v in p_ar_rom.items():
            ar_roman.setdefault(k, v)
        _merge_set_dicts(he2ar_cog, p_he2ar)
        _merge_set_dicts(he_sem, p_he_sem)
        _merge_set_dicts(he_lemmas, p_he_lem)
        _merge_set_dicts(he_borrow, p_he_bor)
        _merge_set_dicts(he_synonyms, p_he_syn)
        for k, v in p_he_disp.items():
            he_disp.setdefault(k, v)
        for k, v in p_he_rom.items():
            he_roman.setdefault(k, v)
        _merge_set_dicts(borrow_graph, p_graph)
        _merge_set_dicts(ar_gloss_cands, p_ar_gloss)
        _merge_set_dicts(he_gloss_cands, p_he_gloss)
        _merge_translations(en_trans, p_en)

    ar_n, he_n, en_n, lines_n = totals
    print(f"  ... done: {lines_n:,} lines total "
          f"(ar:{ar_n:,} he:{he_n:,} en:{en_n:,}"
          f" graph:{len(borrow_graph):,} nodes)")

    return (
        ar2he_cog, ar_sem, ar_lemmas, ar_borrow, ar_synonyms, ar_disp, ar_roman,
        he2ar_cog, he_sem, he_lemmas, he_borrow, he_synonyms, he_disp, he_roman,
        borrow_graph, ar_gloss_cands, he_gloss_cands,
        en_trans,
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


def _collect(forms, index):
    out = set()
    for f in forms:
        out.update(index.get(f, set()))
    return out


def match_pair(ar_norm, he_norm, ar2he, he2ar, ar_roots, he_roots,
               ar_lemmas, he_lemmas, ar_borrow, he_borrow):
    """Check a single (Arabic, Hebrew) pair against all three layers."""
    layers = []
    shared_roots = []
    shared_sources = []

    ar_forms = {ar_norm}
    ar_forms.update(ar_lemmas.get(ar_norm, set()))
    he_forms = {he_norm}
    he_forms.update(he_lemmas.get(he_norm, set()))

    for af in ar_forms:
        for hf in he_forms:
            if hf in ar2he.get(af, set()):
                layers.append("direct_cognate_ar→he")
            if af in he2ar.get(hf, set()):
                layers.append("direct_cognate_he→ar")
    layers = list(dict.fromkeys(layers))

    common_roots = _collect(ar_forms, ar_roots) & _collect(he_forms, he_roots)
    if common_roots:
        layers.append("shared_proto_semitic_root")
        shared_roots = sorted(common_roots)

    common_borrow = _collect(ar_forms, ar_borrow) & _collect(he_forms, he_borrow)
    if common_borrow:
        layers.append("shared_borrowing_source")
        shared_sources = sorted(f"{lang}:{word}" for lang, word in common_borrow)

    return layers, shared_roots, shared_sources


def main():
    t_total = time.monotonic()

    print("Loading word list …")
    words = []
    with open(WORDS_FILE) as f:
        for line in f:
            w = line.strip()
            if w:
                words.append(w.lower())
    word_set = set(words)
    print(f"  {len(words)} words from {WORDS_FILE}")

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
        ar2he_cog, ar_sem, ar_lemmas, ar_borrow, ar_synonyms, ar_disp, ar_roman,
        he2ar_cog, he_sem, he_lemmas, he_borrow, he_synonyms, he_disp, he_roman,
        borrow_graph, ar_gloss_cands, he_gloss_cands,
        en_trans,
    ) = build_all_indexes(ALL_WORDS_FILE, word_set)
    scan_time = time.monotonic() - t0

    print(f"\n  Arabic:  {len(ar2he_cog)} cognate refs, {len(ar_sem)} sem roots, "
          f"{len(ar_lemmas)} lemma links, {len(ar_borrow)} borrow sources, "
          f"{len(ar_synonyms)} synonym links")
    print(f"  Hebrew:  {len(he2ar_cog)} cognate refs, {len(he_sem)} sem roots, "
          f"{len(he_lemmas)} lemma links, {len(he_borrow)} borrow sources, "
          f"{len(he_synonyms)} synonym links")
    print(f"  Borrow graph: {len(borrow_graph)} nodes")
    print(f"  Gloss candidates: {len(ar_gloss_cands)} ar, {len(he_gloss_cands)} he")

    _make_bidirectional(ar_synonyms)
    _make_bidirectional(he_synonyms)
    _make_bidirectional(ar2he_cog)
    _make_bidirectional(he2ar_cog)
    print(f"  After bidirectional: ar cognate refs {len(ar2he_cog)}, "
          f"he cognate refs {len(he2ar_cog)}, "
          f"ar synonyms {len(ar_synonyms)}, he synonyms {len(he_synonyms)}")

    ar_expanded = _expand_borrow_transitive(ar_borrow, borrow_graph)
    he_expanded = _expand_borrow_transitive(he_borrow, borrow_graph)
    print(f"  Transitive expansion: {ar_expanded} ar words, {he_expanded} he words")

    with_ar = sum(1 for v in en_trans.values() if v["ar"])
    with_he = sum(1 for v in en_trans.values() if v["he"])
    with_both = sum(1 for v in en_trans.values() if v["ar"] and v["he"])
    print(f"  English: {len(en_trans)} words, {with_ar} with ar, "
          f"{with_he} with he, {with_both} with both")
    print(f"  ⏱ {scan_time:.1f}s")

    # ── Inject gloss-derived candidates ─────────────────────────
    print("\nInjecting gloss-derived candidates …")
    gloss_added = 0
    for eng_word, ar_norms in ar_gloss_cands.items():
        if eng_word not in word_set:
            continue
        tr = en_trans.setdefault(eng_word, {"ar": {}, "he": {}})
        for norm in ar_norms:
            if norm not in tr["ar"] and norm in ar_disp:
                tr["ar"][norm] = {"word": ar_disp[norm], "roman": ar_roman.get(norm, "")}
                gloss_added += 1
    for eng_word, he_norms in he_gloss_cands.items():
        if eng_word not in word_set:
            continue
        tr = en_trans.setdefault(eng_word, {"ar": {}, "he": {}})
        for norm in he_norms:
            if norm not in tr["he"] and norm in he_disp:
                tr["he"][norm] = {"word": he_disp[norm], "roman": he_roman.get(norm, "")}
                gloss_added += 1
    print(f"  {gloss_added} gloss candidates added")

    # ── Expand candidates via synonyms ────────────────────────────
    print("\nExpanding candidates via synonyms …")
    syn_added = 0
    for word in words:
        tr = en_trans.get(word)
        if not tr:
            continue
        for ar_norm in list(tr["ar"]):
            for syn_norm in ar_synonyms.get(ar_norm, set()):
                if syn_norm not in tr["ar"] and syn_norm in ar_disp:
                    tr["ar"][syn_norm] = {
                        "word": ar_disp[syn_norm],
                        "roman": ar_roman.get(syn_norm, ""),
                    }
                    syn_added += 1
        for he_norm in list(tr["he"]):
            for syn_norm in he_synonyms.get(he_norm, set()):
                if syn_norm not in tr["he"] and syn_norm in he_disp:
                    tr["he"][syn_norm] = {
                        "word": he_disp[syn_norm],
                        "roman": he_roman.get(syn_norm, ""),
                    }
                    syn_added += 1
    print(f"  {syn_added} synonym candidates added")

    # ── N×M cognate matching ──────────────────────────────────────
    print("\nMatching cognates …")
    t0 = time.monotonic()
    results = {}
    words_checked = 0

    for word in words:
        tr = en_trans.get(word)
        if not tr or not tr["ar"] or not tr["he"]:
            continue

        words_checked += 1
        matches = []
        for ar_norm, ar_info in tr["ar"].items():
            for he_norm, he_info in tr["he"].items():
                if (ar_norm, he_norm) in false_pos:
                    continue
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
    print(f"\n  {len(results)} / {len(words)} words produced cognate matches")
    print(f"  ⏱ {match_elapsed:.1f}s ({words_checked} words checked, {avg_ms:.2f}ms avg)")

    print(f"\nWriting {OUTPUT_FILE} …")
    with open(OUTPUT_FILE, "wb") as f:
        f.write(orjson.dumps(results, option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS))

    print(f"Writing {CSV_FILE} …")
    with open(CSV_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["english", "arabic_romanization", "hebrew_romanization"])
        for word in words:
            if word not in results:
                continue
            for m in results[word]["cognate_matches"]:
                writer.writerow([word, m["arabic_roman"], m["hebrew_roman"]])

    print(f"\nDone in {time.monotonic() - t_total:.1f}s total.")


if __name__ == "__main__":
    main()
