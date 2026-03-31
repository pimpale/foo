#!/usr/bin/env python3
"""
Reduce the 1000 most common English words to their base/dictionary forms
using Wiktionary's own form_of links from the kaikki data.

E.g. "children" → "child", "began" → "begin", "feet" → "foot"

Outputs english-base-forms.txt with one word per line, deduplicated.
"""

import sys
import time
from collections import defaultdict
from pathlib import Path

import orjson

ALL_WORDS_FILE = Path("data/kaikki.org-dictionary-all-words.jsonl")
COMMON_WORDS_FILE = Path("1000-most-common-words.txt")
BASE_FORMS_FILE = Path("english-base-forms.txt")

_EN = b'"en"'


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

    print(f"\nScanning {ALL_WORDS_FILE.name} for English form_of links …")
    t0 = time.monotonic()
    # Track form_of bases and whether the word has any independent senses
    word_bases = defaultdict(set)
    word_has_own_sense = defaultdict(bool)
    en_count = 0

    with open(ALL_WORDS_FILE, "rb") as f:
        for i, line in enumerate(f):
            if i % 2_000_000 == 0:
                print(f"  ... {i:,} lines scanned ({en_count:,} en)",
                      file=sys.stderr)
            if _EN not in line:
                continue
            entry = orjson.loads(line)
            if entry.get("lang_code") != "en":
                continue
            en_count += 1

            word = entry.get("word", "").lower()
            if word not in common_set:
                continue

            for sense in entry.get("senses", []):
                if "form_of" in sense:
                    for fof in sense["form_of"]:
                        base = fof.get("word", "").lower()
                        if base and base != word and base in common_set:
                            word_bases[word].add(base)
                else:
                    word_has_own_sense[word] = True

    # Only replace words that have NO independent senses
    en_lemmas = {}
    for word, bases in word_bases.items():
        if not word_has_own_sense[word]:
            en_lemmas[word] = bases

    print(f"  done in {time.monotonic() - t0:.1f}s "
          f"({en_count:,} English entries scanned)")
    print(f"  {len(word_bases)} words with form_of links, "
          f"{len(en_lemmas)} are pure inflections (no independent senses)")

    for word, bases in sorted(en_lemmas.items()):
        print(f"    {word} → {sorted(bases)}")

    skipped = set(word_bases) - set(en_lemmas)
    if skipped:
        print(f"  Skipped (have independent senses): {sorted(skipped)}")

    # Build deduplicated base form list preserving original order
    base_form_list = []
    for word in common_words:
        lemmas = en_lemmas.get(word, set())
        if lemmas:
            base_form_list.append(sorted(lemmas)[0])
        else:
            base_form_list.append(word)
    base_form_list = list(dict.fromkeys(base_form_list))

    print(f"\n  {len(common_words)} original → {len(base_form_list)} base forms")

    with open(BASE_FORMS_FILE, "w") as f:
        for w in base_form_list:
            f.write(w + "\n")
    print(f"  Written to {BASE_FORMS_FILE}")


if __name__ == "__main__":
    main()
