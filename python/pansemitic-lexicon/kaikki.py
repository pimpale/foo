"""Kaikki adapter — bridges raw kaikki dictionary entries to the Word/IPA model.

PartialSource captures what one source knows about a (lang, word):
  - PartialSource itself: data extractable from the word's own kaikki entry
  - SharedSource (subclass): adds the `tr` field captured from an etymology
    template that cites this word from elsewhere

Romanization resolution uses up to three tiers ranked by `_romanization_score`:
  tier1 — the bare word, if it already looks romanized
  tier2 — `template_tr` from a citing etymology template (SharedSource only)
  tier3 — the kaikki entry's own forms[*].form with tag "romanization"
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Self

def _geminate(form: str) -> str:
    """Collapse runs of identical letters into letter + ː.

    Applied to the raw romanization *before* single-char substitutions so
    that doubled scholar-notation letters (e.g. ṣṣ) collapse cleanly and
    the ː survives downstream (ṣṣ → ṣː → sˤː).
    """
    return re.sub(r"([^\W\d_])\1+", r"\1ː", form)


_IPA_DELIMITED = re.compile(r"[/\[]([^/\]]+)[/\]]")
_HTML_TAG = re.compile(r"<[^>]+>")
_INVALID_ROMANIZATION_VALUES = frozenset({"", "-", "?", "??"})
_ROMANIZATION_EXTRA_LETTERS = frozenset({"ʾ", "ʿ", "ʔ", "ʕ", "θ"})
_ROMANIZATION_MARKUP_CHARS = frozenset("*-_'’()./^")


def _clean_romanization_candidate(s: str | None) -> str | None:
    if not s:
        return None
    s = s.strip()
    if not s:
        return None
    if _HTML_TAG.search(s):
        return None
    if "," in s:
        s = s.split(",")[0].strip()
    if s in _INVALID_ROMANIZATION_VALUES:
        return None
    return s or None


def _looks_romanized(s: str | None) -> bool:
    if not s:
        return False
    has_letter = False
    for c in unicodedata.normalize("NFD", s):
        if c in _ROMANIZATION_EXTRA_LETTERS:
            has_letter = True
            continue
        if c.isalpha():
            has_letter = True
            if "LATIN" in unicodedata.name(c, "") or unicodedata.category(c) == "Lm":
                continue
            return False
        if c.isdigit() or c.isspace():
            continue
        if unicodedata.category(c) == "Mn":
            continue
        if c in _ROMANIZATION_MARKUP_CHARS:
            continue
        return False
    return has_letter


def _romanization_vowel_count(text: str) -> int:
    count = 0
    for c in unicodedata.normalize("NFD", text.lower()):
        if unicodedata.category(c) == "Mn":
            continue
        if c in "aeiou":
            count += 1
    return count


def _romanization_letter_count(text: str) -> int:
    return sum(1 for c in text if c.isalpha())


def _romanization_score(text: str) -> tuple[int, int, int]:
    """Rank valid romanization candidates.

    Prefer reconstructions and vocalized transliterations over bare consonantal
    spellings.  This keeps scholarly proto-Semitic forms like ``*ʾaḥad-`` while
    letting vocalized template/entry transliterations beat consonant-only forms
    such as Pahlavi ``ʾmbl`` when a candidate like ``ambar`` is available.
    """
    vowels = _romanization_vowel_count(text)
    letters = _romanization_letter_count(text)
    score = 0
    if "*" in text:
        score += 100
    if letters >= 4 and vowels <= 1 and "*" not in text:
        score -= 30
    score += min(vowels, 4) * 10
    parts = [part for part in text.split("-") if part]
    if len(parts) >= 3 and all(len(part) <= 2 for part in parts):
        score -= 20
    markup_penalty = sum(1 for c in text if c in "^()/[]{}")
    score -= markup_penalty * 2
    return (score, vowels, -markup_penalty)


def _resolve_tiers(tiers: list[str | None]) -> str | None:
    """Pick the highest-scoring romanization candidate; lower index breaks ties."""
    best: tuple[tuple[int, int, int], int, str] | None = None
    for idx, value in enumerate(tiers):
        if not value:
            continue
        ranked = (_romanization_score(value), -idx, value)
        if best is None or ranked > best:
            best = ranked
    return best[2] if best else None





def canonical_from_entry(entry: dict[str, Any]) -> str:
    """Return the canonical form from a kaikki entry, or the headword as fallback."""
    for fm in entry.get("forms", []):
        if "canonical" in fm.get("tags", []):
            return fm.get("form", entry.get("word", ""))
    return entry.get("word", "")


def _romanization_from_entry(entry: dict[str, Any]) -> str | None:
    for fm in entry.get("forms", []):
        if "romanization" in fm.get("tags", []):
            return fm.get("form") or None
    return None


# Substrings in kaikki's free-form `note` field that we promote to synthetic
# tags so they participate in dialect selection.  Egyptian carries period
# information here (e.g. "Late Egyptian, c. 800 BCE") rather than as a real
# tag, so without this Egyptian period preference is unreachable.
_NOTE_TAG_SYNONYMS: list[tuple[str, str]] = [
    ("Late Egyptian", "Late-Egyptian"),
    ("Middle Egyptian", "Middle-Egyptian"),
    ("Old Egyptian", "Old-Egyptian"),
]


def _ipa_from_entry(entry: dict[str, Any]) -> list[IpaRealization]:
    ipa_list = []
    for s in entry.get("sounds", []):
        ipa = s.get("ipa")
        if not ipa:
            continue
        tags = list(s.get("tags", []))
        note = s.get("note") or ""
        for substr, synth in _NOTE_TAG_SYNONYMS:
            if substr in note and synth not in tags:
                tags.append(synth)
        ipa_list.append(IpaRealization(ipa=ipa, tags=tags).normalize())
    return ipa_list

@dataclass
class IpaRealization:
    """
    Structured IPA data including different dialects or variants.
    """
    ipa: str
    tags: list[str]
    
    def normalize(self) -> IpaRealization :
        """Strip kaikki's IPA wrapper (/…/ or […]), prosodic marks, and geminates."""

        m = _IPA_DELIMITED.search(self.ipa)
        out = m.group(1) if m else self.ipa
        out = out.split(",")[0].strip()
        # Some kaikki IPA strings are malformed and only include one edge
        # delimiter (e.g. "/braːɡ"); strip leftover wrapper chars.
        out = out.strip("/[]")
        for c in "ˈˌ.":
            out = out.replace(c, "")
        out = out.strip()
        # Some kaikki IPA spells gemination as a doubled letter split across
        # syllables (e.g. /ebˈbuː.bum/ → ebbuːbum).  Collapse to ː convention.
        out = _geminate(out)
        # set 
        return IpaRealization(ipa=out, tags=self.tags)
    

@dataclass
class PartialSource:
    """Surface data about a (lang, word) extracted from its own kaikki entry.

    Stores raw fields; tier1 / tier3 / romanization are derived properties.
    """
    lang: str
    word: str
    kaikki_romanization: str | None = None
    pronunciations: list[IpaRealization] = field(default_factory=list)

    @classmethod
    def from_kaikki_entry(cls, entry: dict[str, Any]) -> Self | None:
        """Build from a kaikki dictionary entry; None if lang/word are missing."""
        lang = entry.get("lang_code", "")
        word = canonical_from_entry(entry)
        if not lang or not word:
            return None
        return cls(
            lang=lang,
            word=word,
            kaikki_romanization=_romanization_from_entry(entry),
            pronunciations=_ipa_from_entry(entry),
        )

    @property
    def tier1(self) -> str | None:
        """The bare word itself, if it already looks like a romanization."""
        candidate = _clean_romanization_candidate(self.word)
        return candidate if _looks_romanized(candidate) else None

    @property
    def tier3(self) -> str | None:
        """The kaikki entry's own romanization, cleaned."""
        candidate = _clean_romanization_candidate(self.kaikki_romanization)
        return candidate if _looks_romanized(candidate) else None

    def _tier_candidates(self) -> list[str | None]:
        return [self.tier1, self.tier3]

    @property
    def romanization(self) -> str | None:
        """Best romanization candidate across this source's available tiers."""
        return _resolve_tiers(self._tier_candidates())

    def __str__(self) -> str:
        return f"{self.lang}:{self.word}"


@dataclass
class SharedSource(PartialSource):
    """A PartialSource enriched with `tr` data from a citing etymology template.

    Built either via `from_citation` (cite-side only) or `resolve` (combines
    an entry-side PartialSource with a cite-side SharedSource).
    """
    template_tr: str | None = None

    @classmethod
    def from_citation(
        cls, lang: str, word: str, template_tr: str | None,
    ) -> Self:
        """Build from an etymology template citing this word — captures `tr`."""
        return cls(lang=lang, word=word, template_tr=template_tr)

    @classmethod
    def resolve(
        cls,
        lang: str,
        word: str,
        entry: PartialSource | None,
        citation: Self | None,
    ) -> Self:
        """Combine an entry-derived PartialSource with a citation-derived SharedSource."""
        return cls(
            lang=lang,
            word=word,
            kaikki_romanization=entry.kaikki_romanization if entry else None,
            pronunciations=entry.pronunciations if entry else [],
            template_tr=citation.template_tr if citation else None,
        )

    @property
    def tier2(self) -> str | None:
        """The `tr` arg from a citing template, cleaned."""
        candidate = _clean_romanization_candidate(self.template_tr)
        return candidate if _looks_romanized(candidate) else None

    def _tier_candidates(self) -> list[str | None]:
        return [self.tier1, self.tier2, self.tier3]
