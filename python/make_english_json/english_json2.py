# %%
import json
import pathlib
from copy import deepcopy
from pathlib import Path
from typing import Dict, Set
from collections import defaultdict

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


def vb_to_vbp(verb: str) -> str:  # 3rd-person-plural (eg "they go")
    if verb in irregular_verbs:
        return irregular_verbs[verb]["VBP"]
    # usually just the same as VB
    return verb


# ---------------------------------------------------------------------------
# Extract verb categories from local VerbNet JSON dump
# ---------------------------------------------------------------------------

# Folders containing downloaded/converted VerbNet class files
VERBNET_DIRS = [Path("verb_verbnet"), Path("verb_custom")]


def _normalize_primary(primary):
    """
    Simplify a VerbNet primary frame to the slot sequence we care about.
    """
    return [p.split(".")[0] for p in primary]


# ---------------------------------------------------------------------------
# Determine verb category from a simplified primary frame
# ---------------------------------------------------------------------------


def cat_from_primary(slots):
    """Return one of our verb category names given *slots* sequence, or None."""
    if slots == ["NP", "V"]:
        return "vb"
    if slots == ["It", "V"]:
        return "vb"
    if slots == ["NP", "V", "ADJ"]:
        return "vb_adjp"
    if slots == ["NP", "V", "S_INF"]:
        return "vb_to_inf_cl"
    if slots == ["NP", "V", "bare_infinitive"]:
        return "vb_bare_inf_cl"
    if slots == ["NP", "V", "S_ING"]:
        return "vb_vbg_cl"
    if slots == ["NP", "V", "VP_VBN"]:
        return "vb_vbn_cl"
    if slots == ["NP", "V", "S"]:
        return "vb_bare_declarative_cl"
    if slots == ["NP", "V", "that", "S"]:
        return "vb_that_declarative_cl"
    if slots == ["NP", "V", "what", "S"]:
        return "vb_interrogative_cl"
    if slots == ["NP", "V", "NP"]:
        return "vb_np"
    if slots == ["NP", "V", "NP", "NP"]:
        return "vb_np_np"
    if slots == ["NP", "V", "NP", "ADJ"]:
        return "vb_np_adjp"
    if slots == ["NP", "V", "NP", "S_INF"]:
        return "vb_np_to_inf_cl"
    if slots == ["NP", "V", "NP", "S_ING"]:
        return "vb_np_vbg_cl"
    if slots == ["NP", "V", "NP", "VP_VBN"]:
        return "vb_np_vbn_cl"
    if slots == ["NP", "V", "NP", "S"]:
        return "vb_np_bare_declarative_cl"
    if slots == ["NP", "V", "NP", "that", "S"]:
        return "vb_np_that_declarative_cl"
    if slots == ["NP", "V", "NP", "what", "S"]:
        return "vb_np_interrogative_cl"
    return None


def extract_verb_categories() -> Dict[str, Dict[str, None]]:
    """Parse all VerbNet JSON files under each directory in *VERBNET_DIRS* and
    build verb sets per coarse syntactic category compatible with our grammar.
    """
    # Use defaultdict so categories appear lazily when first encountered
    categories: Dict[str, Set[str]] = defaultdict(set)

    for verb_dir in VERBNET_DIRS:
        if not verb_dir.exists():
            # Skip missing directories so the script can run even if some
            # optional sources are absent.
            continue
        for fp in verb_dir.glob("*.json"):
            data = json.loads(fp.read_text())
            members = [m.lower() for m in data.get("members", [])]
            for frame in data.get("frames", []):
                slots = _normalize_primary(frame["primary"])
                cat = cat_from_primary(slots)
                if cat is None:
                    continue

                for verb in members:
                    categories[cat].add(verb)

    # Convert to regular dict[cat] -> {verb: None, ...}
    # Any category that never received members simply won't appear.
    return {cat: {v: None for v in sorted(vset)} for cat, vset in categories.items()}


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
        """Depth-first traversal that accumulates words while checking
        countability compatibility between including and included classes.

        A class may optionally specify a boolean "countable" property. If both
        the including class *fname* **and** an included class specify this
        property explicitly and their values differ, we raise a
        ValueError – this signals inconsistent modelling in the noun
        hierarchy.
        """

        if fname in cache:
            return cache[fname]
        if fname in stack:
            # Cycle detected – ignore to prevent infinite recursion.
            return set()
        stack.add(fname)

        data = raw.get(fname, {})
        words = set(data.get("words", {}).keys())
        parent_countable = data.get("countable")  # True / False / None

        for inc in data.get("classes", []):
            # normalise inc to filename (ensure endswith .json)
            inc_file = inc if inc.endswith(".json") else inc + ".json"

            inc_countable = raw.get(inc_file, {}).get("countable")
            # Both sides explicit → must match
            if (
                parent_countable in (True, False)
                and inc_countable in (True, False)
                and parent_countable != inc_countable
            ):
                raise ValueError(
                    f"Countability mismatch: '{fname}' (countable={parent_countable}) "
                    f"includes '{inc_file}' (countable={inc_countable})"
                )

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
# Derive verb categories from VerbNet
verbs = extract_verb_categories()

english_json: Dict[str, Dict[str, None]] = {}

# 1. Indeclinable categories (determiners, conjunctions, etc.)
for kind in indeclinable:
    english_json[kind] = deepcopy(indeclinable[kind])

# 2. Noun classes (from nouns/ folder)
raw_noun_data = load_raw_noun_files()
resolved_nouns = resolve_nouns(raw_noun_data)

# Initialise aggregate noun classes
english_json["noun"] = {}
english_json["countable_noun"] = {}
english_json["uncountable_noun"] = {}

for fname, words in resolved_nouns.items():
    class_name = Path(fname).stem  # e.g., tools_countable
    english_json[class_name] = {w: None for w in sorted(words)}

    # Determine countability from the file's metadata. Only when explicitly
    # marked as countable do we add plural forms.
    is_countable = raw_noun_data.get(fname, {}).get("countable")

    for w in words:
        english_json["noun"][w] = None
        english_json["noun"][noun_to_noun_pl(w)] = None

    # If explicitly marked countable add plural forms and update global class
    if is_countable is True:
        for w in words:
            english_json[class_name][noun_to_noun_pl(w)] = None

            # Aggregate into countable_noun set (both singular & plural)
            english_json["countable_noun"][w] = None
            english_json["countable_noun"][noun_to_noun_pl(w)] = None

    elif is_countable is False:
        # Aggregate uncountable nouns
        for w in words:
            english_json["uncountable_noun"][w] = None

# 3. Verb classes and their inflections
for kind in verbs:
    # base (VB*) entries as provided
    english_json[kind] = deepcopy(verbs[kind])

    # derive other forms from VB* sets
    english_json[kind.replace("vb", "vbd", 1)] = {
        vb_to_vbd(v): None for v in verbs[kind]
    }
    english_json[kind.replace("vb", "vbn", 1)] = {
        vb_to_vbn(v): None for v in verbs[kind]
    }
    english_json[kind.replace("vb", "vbg", 1)] = {
        vb_to_vbg(v): None for v in verbs[kind]
    }
    english_json[kind.replace("vb", "vbz", 1)] = {
        vb_to_vbz(v): None for v in verbs[kind]
    }
    english_json[kind.replace("vb", "vbp", 1)] = {
        vb_to_vbp(v): None for v in verbs[kind]
    }

# ---------------------------------------------------------------------------
# 3b. Adverb classes (from adverbs/ folder)
# ---------------------------------------------------------------------------
ADVERBS_DIR = Path("adverbs")
for fp in ADVERBS_DIR.glob("*.json"):
    data = json.loads(fp.read_text())
    words = data.get("words", {}).keys()
    for cls in data.get("classes", []):
        # Ensure the class dictionary exists
        english_json.setdefault(cls, {})
        for w in words:
            english_json[cls][w] = None

# ---------------------------------------------------------------------------
# 4. Preposition classes (from prepositions/ folder)
# ---------------------------------------------------------------------------
english_json["preposition"] = {}
PREPOSITIONS_DIR = Path("prepositions")
for fp in PREPOSITIONS_DIR.glob("*.json"):
    data = json.loads(fp.read_text())
    words = data.get("words", {}).keys()
    for cls in data.get("classes", []):
        english_json.setdefault(cls, {})
        for w in words:
            english_json[cls][w] = None

# ---------------------------------------------------------------------------
# Transpose: word -> list-of-classes
# ---------------------------------------------------------------------------

transposed: Dict[str, Set[str]] = {}
for kind, words in english_json.items():
    for word in words:
        # lowercase all words and avoid duplicate class entries
        transposed.setdefault(word.lower(), set()).add(kind)

# Convert each set to a sorted list for consistent JSON output
transposed = {word: sorted(list(classes)) for word, classes in transposed.items()}

# Write output
Path("english.json").write_text(json.dumps(transposed, indent=4))
print(f"Wrote english.json with {len(transposed)} unique words.")
