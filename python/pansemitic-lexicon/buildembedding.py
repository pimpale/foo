#!/usr/bin/env python3
"""
Build per-sense embeddings for Arabic and Hebrew words using Qwen

For each sense of each Arabic/Hebrew word in the Wiktionary dump, creates an
embedding of the string: "pos | word | gloss1, gloss2, ..."

Outputs:
  - senses.json: metadata keyed by language → normalized_word → list of senses
                 each sense has an "idx" pointing into the embedding matrix
  - embeddings.npy: float32 matrix of shape (num_senses, 4096)
"""

import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import orjson
from sentence_transformers import SentenceTransformer

from reconstruction import ArabicWord, HebrewWord

DATA_DIR = Path("data")
ALL_WORDS_FILE = DATA_DIR / "kaikki.org-dictionary-all-words.jsonl"
SENSES_FILE = Path("senses.json")
EMBEDDINGS_FILE = Path("embeddings.npy")

_AR = b'"ar"'
_HE = b'"he"'

BATCH_SIZE = 1024


def _extract_canonical(entry):
    """Return the canonical form from a kaikki entry, or the headword as fallback."""
    for fm in entry.get("forms", []):
        if "canonical" in fm.get("tags", []):
            return fm.get("form", entry.get("word", ""))
    return entry.get("word", "")


def _extract_senses_chunk(filepath_str, start_byte, end_byte):
    """Worker: extract (lang, word, canonical, pos, glosses, romanization) from a chunk."""
    results = []

    with open(filepath_str, "rb") as f:
        if start_byte > 0:
            f.seek(start_byte)
            f.readline()

        while f.tell() < end_byte:
            line = f.readline()
            if not line:
                break

            if not (_AR in line or _HE in line):
                continue

            entry = orjson.loads(line)
            lang = entry.get("lang_code", "")
            if lang not in ("ar", "he"):
                continue

            word = entry.get("word", "")
            if not word:
                continue

            canonical = _extract_canonical(entry)
            pos = entry.get("pos", "")
            roman = ""
            for fm in entry.get("forms", []):
                if "romanization" in fm.get("tags", []):
                    roman = fm.get("form", "")
                    break

            for sense in entry.get("senses", []):
                glosses = sense.get("glosses", [])
                if not glosses:
                    continue
                gloss_text = ", ".join(glosses)
                results.append((lang, word, canonical, pos, gloss_text, roman))

    return results


def extract_all_senses(filepath, num_workers=None):
    """Parallel scan of the JSONL file to extract all Arabic/Hebrew senses."""
    if num_workers is None:
        num_workers = os.cpu_count() or 1

    file_size = filepath.stat().st_size
    chunk_size = file_size // num_workers

    chunks = []
    for i in range(num_workers):
        start = i * chunk_size
        end = file_size if i == num_workers - 1 else (i + 1) * chunk_size
        chunks.append((str(filepath), start, end))

    print(f"  dispatching {num_workers} workers …", file=sys.stderr)

    all_senses = []
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_extract_senses_chunk, *c): i
                   for i, c in enumerate(chunks)}
        done_count = 0
        for future in as_completed(futures):
            result = future.result()
            all_senses.extend(result)
            done_count += 1
            print(f"  ... {done_count}/{num_workers} chunks done", file=sys.stderr)

    return all_senses


def build_sense_records(raw_senses):
    """Deduplicate and organize senses, assign embedding indices.

    Keyed by canonical form (diacritized) so that entries like
    צַדִּיק (righteous) and צָדִּי״ק (letter name) get separate sense lists.
    """
    # key: (lang, canonical, pos, gloss_text) → first occurrence info
    seen = {}
    records = defaultdict(lambda: defaultdict(list))
    idx = 0

    for lang, word, canonical, pos, gloss_text, roman in raw_senses:
        norm = (ArabicWord if lang == "ar" else HebrewWord).normalize(word)
        key = (lang, canonical, pos, gloss_text)
        if key in seen:
            continue
        seen[key] = idx

        records[lang][canonical].append({
            "norm": norm,
            "pos": pos,
            "gloss": gloss_text,
            "roman": roman,
            "idx": idx,
        })
        idx += 1

    return records, idx


def build_embedding_texts(records):
    """Build the text strings to embed, ordered by idx."""
    count = sum(
        len(senses)
        for lang_records in records.values()
        for senses in lang_records.values()
    )
    texts = [""] * count
    for lang, lang_records in records.items():
        for norm, senses in lang_records.items():
            for s in senses:
                # e.g. "noun | تمساح | crocodile, alligator"
                texts[s["idx"]] = f"A {s['pos']} meaning '{s['gloss']}'"
    return texts


def embed_texts(model, texts, batch_size=BATCH_SIZE):
    """Embed all texts in batches, return normalized float32 numpy matrix."""
    all_embeddings = []
    total = len(texts)

    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        emb = model.encode(batch, normalize_embeddings=True)
        all_embeddings.append(emb.astype(np.float32))

        done = min(i + batch_size, total)
        print(f"  embedded {done}/{total} senses", file=sys.stderr)

    return np.concatenate(all_embeddings, axis=0)


def main():
    t_total = time.monotonic()

    print("Scanning for Arabic/Hebrew senses …")
    t0 = time.monotonic()
    raw_senses = extract_all_senses(ALL_WORDS_FILE)
    print(f"  {len(raw_senses)} raw senses in {time.monotonic() - t0:.1f}s")

    print("Building sense records …")
    records, num_senses = build_sense_records(raw_senses)
    ar_count = sum(len(v) for v in records.get("ar", {}).values())
    he_count = sum(len(v) for v in records.get("he", {}).values())
    print(f"  {num_senses} unique senses (ar: {ar_count}, he: {he_count})")

    print("Building embedding texts …")
    texts = build_embedding_texts(records)

    print("Loading model …")
    t0 = time.monotonic()
    model = SentenceTransformer("microsoft/harrier-oss-v1-0.6b")
    # model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
    # model = SentenceTransformer('nvidia/NV-Embed-v2', trust_remote_code=True)
    print(f"  loaded in {time.monotonic() - t0:.1f}s")

    print("Embedding senses …")
    t0 = time.monotonic()
    embeddings = embed_texts(model, texts)
    print(f"  done in {time.monotonic() - t0:.1f}s, shape: {embeddings.shape}")

    print(f"Writing {SENSES_FILE} …")
    # Convert defaultdicts to plain dicts for serialization
    out = {lang: dict(words) for lang, words in records.items()}
    with open(SENSES_FILE, "wb") as f:
        f.write(orjson.dumps(out, option=orjson.OPT_INDENT_2))

    print(f"Writing {EMBEDDINGS_FILE} …")
    np.save(EMBEDDINGS_FILE, embeddings)

    print(f"\nDone in {time.monotonic() - t_total:.1f}s total.")
    print(f"  {SENSES_FILE}: metadata with idx references")
    print(f"  {EMBEDDINGS_FILE}: {embeddings.shape} float32 matrix")


if __name__ == "__main__":
    main()
