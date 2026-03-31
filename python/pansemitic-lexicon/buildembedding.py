#!/usr/bin/env python3
"""
Build per-sense embeddings for Arabic and Hebrew words using NV-Embed-v2.

For each sense of each Arabic/Hebrew word in the Wiktionary dump, creates an
embedding of the string: "pos | word | gloss1, gloss2, ..."

Outputs:
  - senses.json: metadata keyed by language → normalized_word → list of senses
                 each sense has an "idx" pointing into the embedding matrix
  - embeddings.npy: float32 matrix of shape (num_senses, 4096)
"""

import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import orjson
import torch
import torch.nn.functional as F
from transformers import AutoModel

DATA_DIR = Path("data")
ALL_WORDS_FILE = DATA_DIR / "kaikki.org-dictionary-all-words.jsonl"
SENSES_FILE = Path("senses.json")
EMBEDDINGS_FILE = Path("embeddings.npy")

ARABIC_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670\u0640]")
HEBREW_DIACRITICS = re.compile(r"[\u0591-\u05BD\u05BF-\u05C7]")

_AR = b'"ar"'
_HE = b'"he"'

BATCH_SIZE = 64


def normalize_arabic(text: str) -> str:
    return ARABIC_DIACRITICS.sub("", text).strip()


def normalize_hebrew(text: str) -> str:
    return HEBREW_DIACRITICS.sub("", text).strip()


def _extract_senses_chunk(filepath_str, start_byte, end_byte):
    """Worker: extract (lang, word, pos, glosses, romanization) from a chunk."""
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
                results.append((lang, word, pos, gloss_text, roman))

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
    """Deduplicate and organize senses, assign embedding indices."""
    # key: (lang, norm, pos, gloss_text) → first occurrence info
    seen = {}
    records = defaultdict(lambda: defaultdict(list))
    idx = 0

    for lang, word, pos, gloss_text, roman in raw_senses:
        norm = normalize_arabic(word) if lang == "ar" else normalize_hebrew(word)
        key = (lang, norm, pos, gloss_text)
        if key in seen:
            continue
        seen[key] = idx

        records[lang][norm].append({
            "display": word,
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
                texts[s["idx"]] = f"{s['pos']} | {s['display']} | {s['gloss']}"
    return texts


def embed_texts(model, texts, batch_size=BATCH_SIZE):
    """Embed all texts in batches, return normalized float32 numpy matrix."""
    all_embeddings = []
    total = len(texts)

    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        with torch.no_grad():
            emb = model.encode(batch, instruction="", max_length=4096)
            emb = F.normalize(emb, p=2, dim=1)
        all_embeddings.append(emb.cpu().float().numpy())

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

    print("Loading NV-Embed-v2 model …")
    t0 = time.monotonic()
    model = AutoModel.from_pretrained("nvidia/NV-Embed-v2", trust_remote_code=True)
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
