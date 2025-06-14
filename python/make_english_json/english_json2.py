#%%
import json
import pathlib
from copy import deepcopy
from pathlib import Path
from typing import Dict, Set

# Load irregular noun/verb maps early so helper functions can reference them
irregular_nouns = json.loads(Path("irregular_noun_forms.json").read_text())
irregular_verbs = json.loads(Path("irregular_verb_forms.json").read_text())

# ---------------------------------------------------------------------------
# Helper functions for inflection (noun plural, verb forms)
# ---------------------------------------------------------------------------


def noun_to_noun_pl(singular: str) -> str:
    """Very small pluraliser – handles regular patterns used in TinyStories."""
    # irregular first
    if singular in irregular_nouns:
        return irregular_nouns[singular]["plural"]
    # regular patterns
    if singular.endswith(("s", "sh", "ch", "x")):
        return singular + "es"
    if singular.endswith("y") and len(singular) > 1 and singular[-2] not in "aeiou":
        return singular[:-1] + "ies"
    return singular + "s"


# Verb inflection helpers ----------------------------------------------------

# `irregular_verbs` already loaded above


def vb_to_vbd(verb: str) -> str:  # preterite
    if verb in irregular_verbs:
        return irregular_verbs[verb]["VBD"]
    if verb.endswith("e"):
        return verb + "d"
    if verb.endswith("y") and verb[-2] not in "aeiou":
        return verb[:-1] + "ied"
    return verb + "ed"


def vb_to_vbn(verb: str) -> str:  # past-participle
    if verb in irregular_verbs:
        return irregular_verbs[verb]["VBN"]
    return vb_to_vbd(verb)


def vb_to_vbg(verb: str) -> str:  # gerund
    if verb in irregular_verbs:
        return irregular_verbs[verb]["VBG"]
    if verb.endswith("ie"):
        return verb[:-2] + "ying"
    if verb.endswith("e") and verb != "be":  # be -> being (already ends with e)
        return verb[:-1] + "ing"
    return verb + "ing"


def vb_to_vbz(verb: str) -> str:  # 3rd-person-singular
    if verb in irregular_verbs:
        return irregular_verbs[verb]["VBZ"]
    if verb.endswith(("s", "sh", "ch", "x")):
        return verb + "es"
    if verb.endswith("y") and verb[-2] not in "aeiou":
        return verb[:-1] + "ies"
    return verb + "s"


# ---------------------------------------------------------------------------
# Resolve noun classes via graph reachability
# ---------------------------------------------------------------------------

NOUNS_DIR = Path("nouns")


def load_raw_noun_files() -> Dict[str, dict]:
    """Load every *.json in nouns/ and return mapping of filename -> json dict."""
    raw = {}
    for fp in NOUNS_DIR.glob("*.json"):
        raw[fp.name] = json.loads(fp.read_text())
    return raw


def resolve_nouns(raw: Dict[str, dict]) -> Dict[str, Set[str]]:
    """Resolve transitive inclusion of classes to produce full word sets."""

    cache: Dict[str, Set[str]] = {}

    def dfs(fname: str, stack: Set[str]) -> Set[str]:
        if fname in cache:
            return cache[fname]
        if fname in stack:
            # Cycle detected – ignore to prevent infinite recursion.
            return set()
        stack.add(fname)

        data = raw.get(fname, {})
        words = set(data.get("words", {}).keys())
        for inc in data.get("classes", []):
            # normalise inc to filename (ensure endswith .json)
            inc_file = inc if inc.endswith(".json") else inc + ".json"
            words |= dfs(inc_file, stack)
        stack.remove(fname)
        cache[fname] = words
        return words

    resolved: Dict[str, Set[str]] = {}
    for fname in raw:
        resolved[fname] = dfs(fname, set())
    return resolved


# ---------------------------------------------------------------------------
# Build english_json
# ---------------------------------------------------------------------------

indeclinable = json.loads(Path("indeclinable.json").read_text())
verbs = json.loads(Path("verbs.json").read_text())
prepositions = json.loads(Path("prepositions.json").read_text())

english_json: Dict[str, Dict[str, None]] = {}

# 1. Indeclinable categories (determiners, conjunctions, etc.)
for kind in indeclinable:
    english_json[kind] = deepcopy(indeclinable[kind])

# 2. Noun classes (from nouns/ folder)
raw_noun_data = load_raw_noun_files()
resolved_nouns = resolve_nouns(raw_noun_data)

for fname, words in resolved_nouns.items():
    class_name = Path(fname).stem  # e.g., tools_countable
    english_json[class_name] = {w: None for w in sorted(words)}

    # Pluralise if the file name suggests countable nouns
    if "countable" in class_name:
        for w in words:
            english_json[class_name][noun_to_noun_pl(w)] = None

# 3. Verb classes and their inflections
for kind in verbs:
    # base (VB*) entries as provided
    english_json[kind] = deepcopy(verbs[kind])

    # derive other forms from VB* sets
    english_json[kind.replace("vb", "vbd", 1)] = {vb_to_vbd(v): None for v in verbs[kind]}
    english_json[kind.replace("vb", "vbn", 1)] = {vb_to_vbn(v): None for v in verbs[kind]}
    english_json[kind.replace("vb", "vbg", 1)] = {vb_to_vbg(v): None for v in verbs[kind]}
    english_json[kind.replace("vb", "vbz", 1)] = {vb_to_vbz(v): None for v in verbs[kind]}
    english_json[kind.replace("vb", "vbp", 1)] = {v: None for v in verbs[kind]}  # plural present is the base verb

# 4. Prepositions (retain original structure + collapsed set)
for kind in prepositions:
    english_json[kind] = deepcopy(prepositions[kind])

english_json["preposition"] = {}
for kind in prepositions:
    for prep in prepositions[kind]:
        english_json["preposition"][prep] = None

# ---------------------------------------------------------------------------
# Transpose: word -> list-of-classes
# ---------------------------------------------------------------------------

transposed: Dict[str, list] = {}
for kind, words in english_json.items():
    for word in words:
        transposed.setdefault(word, []).append(kind)

# Sort class lists for consistency
for word in transposed:
    transposed[word].sort()

# Write output
Path("english.json").write_text(json.dumps(transposed, indent=4))
print(f"Wrote english.json with {len(transposed)} unique words.")