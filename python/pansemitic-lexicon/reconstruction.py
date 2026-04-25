from __future__ import annotations

import re
import unicodedata
from typing import Self, TYPE_CHECKING
from dataclasses import dataclass

from kaikki import _geminate

if TYPE_CHECKING:
    from kaikki import SharedSource


class ReconstructionError(Exception):
    """Raised when ancestor reconstruction fails, with a categorized reason."""
    pass


class UnsupportedLanguageError(ReconstructionError):
    def __init__(self, lang: str):
        self.lang = lang
        super().__init__(f"unsupported source language: {lang}")


class ConsonantMismatchError(ReconstructionError):
    def __init__(self, ar_count: int, he_count: int):
        self.ar_count = ar_count
        self.he_count = he_count
        super().__init__(f"consonant count mismatch: ar={ar_count} he={he_count}")


class MissingRomanizationError(ReconstructionError):
    def __init__(self, missing: str):
        self.missing = missing
        super().__init__(f"missing romanization: {missing}")


class EmptyAncestorError(ReconstructionError):
    def __init__(self):
        super().__init__("ancestor word is empty after normalization")


# ── Internal encoding: plain Unicode IPA ───────────────────────
#
# Word.word stores a lossless IPA transcription:
#   - Vowel length via ː             aː iː uː eː oː
#   - Gemination via ː               bː sˤː  (Wikipedia: ː marks length and
#                                              gemination alike)
#   - Pharyngealization via ˤ        sˤ tˤ dˤ ðˤ
#   - Affricates as tie-bar digraphs d͡ʒ (Arabic jīm), t͡s (Hebrew tsade), t͡ʃ
#
# Helpers below convert IPA to either "proto-Semitic scholar convention"
# (ṣ ṭ ḫ ṯ ḏ ḥ š ś) or to the pansemitic form.  Anything downstream that
# wants the scholar form must call Word.to_protosemitic_convention() — the
# internal string is no longer in that encoding.

VOWELS = set("aeiou")
IPA_MODIFIERS = set("ːˤʰʲʷˠʼ̃")
IPA_CONSONANT_DIGRAPHS = (
    "d͡ʒ", "t͡ʃ", "t͡s", "d͡z",
    "t͡ɬ", "d͡ɮ", "k͡x", "ɡ͡ɣ",
)


def _strip_combining(form: str) -> str:
    """Strip leftover combining-mark diacritics (NFD decompose + drop Mn).

    IPA modifier letters (ˤ, ː, ʰ, ʲ, ʷ, ˠ — category Lm) survive.  Used at
    the tail of the generic-Latin fallback path to clean assorted European /
    Iranian / Turkic stress and length marks (â, ñ, è, ą, ǭ, ı …) that
    leak through when a non-Semitic source has no specific subclass.
    """
    out = []
    for c in unicodedata.normalize("NFD", form):
        # Preserve U+0361 COMBINING DOUBLE INVERTED BREVE — the IPA tie bar
        # that makes d͡ʒ, t͡ʃ, t͡s single phonemes, U+0329 COMBINING VERTICAL
        # LINE BELOW for syllabic resonants such as r̩ and l̩, and U+0303
        # COMBINING TILDE for nasalized vowels such as ã.
        if unicodedata.category(c) == "Mn" and c not in {"͡", "̩", "̃"}:
            continue
        if c == "ı":  # dotless i — doesn't decompose; map to plain i
            out.append("i")
        else:
            out.append(c)
    return "".join(out)


def _strip_acute_vowels(form: str) -> str:
    return (form
            .replace("á", "a").replace("é", "e").replace("í", "i")
            .replace("ó", "o").replace("ú", "u"))


def _strip_tone_vowels(form: str) -> str:
    return (form
            .replace("â", "a").replace("ǎ", "a").replace("ā", "a")
            .replace("ê", "e").replace("ě", "e").replace("ē", "e")
            .replace("î", "i").replace("ǐ", "i").replace("ī", "i")
            .replace("ô", "o").replace("ǒ", "o").replace("ō", "o")
            .replace("û", "u").replace("ǔ", "u").replace("ū", "u"))


def _promote_circumflex_vowels(form: str) -> str:
    return (form
            .replace("â", "ā")
            .replace("ê", "ē")
            .replace("î", "ī")
            .replace("ô", "ō")
            .replace("û", "ū"))


def _normalize_hebrew_vowel_marks(form: str) -> str:
    """Drop Hebrew romanization stress marks but preserve macron length."""
    out = []
    for c in unicodedata.normalize("NFD", form):
        if unicodedata.category(c) == "Mn" and c != "̄":
            continue
        out.append(c)
    return unicodedata.normalize("NFC", "".join(out))

# IPA → proto-Semitic scholar notation.  Strips ː (length and gemination).
_IPA_TO_SCHOLAR: list[tuple[str, str]] = [
    # multi-char first
    ("ðˤ", "ṯ̣"),
    ("θˤ", "ṯ̣"),
    ("ɬˤ", "ṣ́"),
    ("sˤ", "ṣ"),
    ("tˤ", "ṭ"),
    ("dˤ", "ḍ"),
    # IPA palatal approximant → y first, so the j produced below for jīm
    # is not double-converted.
    ("j", "y"),
    ("d͡ʒ", "j"),
    # single-char  (t͡s, t͡ʃ are Hebrew-only; they have no scholar equivalent
    # in proto-Semitic notation, so we leave them as IPA.)
    ("θ", "ṯ"),
    ("ð", "ḏ"),
    ("χ", "ḫ"),
    ("x", "ḫ"),
    ("ħ", "ḥ"),
    ("ʃ", "š"),
    ("ɬ", "ś"),
    ("ʒ", "ž"),
]

@dataclass
class Word:
    """A word stored as a plain Unicode IPA string (lossless).

    Subclass `from_romanization` methods convert their source notation to IPA
    preserving vowel length and gemination.  Downstream consumers should call
    `to_protosemitic_convention()` for scholarly notation or
    `to_protopansemitic()` for the compressed pansemitic form.
    """
    word: str

    @property
    def lang(self) -> str:
        raise NotImplementedError("Subclasses must implement lang property")

    @classmethod
    def from_ipa(cls, ipa: str) -> Self:
        """Wrap a pre-existing IPA string, normalizing combining marks.

        Strips stress/length/prosodic accents (NFD + drop combining marks)
        while preserving the IPA tie bar.  This is the entry point used when
        kaikki supplies a native IPA directly.
        """
        # many languages represent an unknown vowel with capital V in even IPA.
        # We replace it with a concrete guessed vowel
        ipa = ipa.replace("V", "a")
        
        return cls(word=_strip_combining(ipa))

    def to_ipa(self) -> str:
        return self.word

    def to_pansemitic_ipa(self) -> str:
        """Return the IPA string that pansemitic reduction should consume."""
        return self.word

    def to_protosemitic_convention(self) -> str:
        """IPA → proto-Semitic scholar notation.  Strips ː (length/gemination)."""
        out = self.word
        for src, dst in _IPA_TO_SCHOLAR:
            out = out.replace(src, dst)
        return out.replace("ː", "")

    def countconsonants(self) -> int:
        return sum(1 for tok in _tokenize_phonemes(self.word) if not _is_vowel_token(tok))

    def consonants(self) -> str:
        return "".join(tok for tok in _tokenize_phonemes(self.word) if not _is_vowel_token(tok))

    def __str__(self) -> str:
        return f"{self.lang}:{self.word}"


class ArabicWord(Word):
    @property
    def lang(self) -> str:
        return "ar"
    
    @classmethod
    def from_romanization(cls, text: str) -> Self:
        """Arabic romanization → IPA (lossless: preserves length + gemination)."""
        if not text:
            return cls(word=text)
        form = text.lower()

        # Strip definite-article assimilated prefixes (al-, an-, aš-, aṣ-, aṭ-, aḍ-).
        form = re.sub(r"^a[lrnsṣṭḍ]-", "", form)

        # Gemination on the raw romanization.  Done before substitutions so
        # that doubled multi-char targets (ṣṣ → ṣː → sˤː) survive.
        form = _geminate(form)

        # Digraph first
        form = form.replace("sh", "ʃ")

        # Pharyngealized (emphatic) consonants
        form = form.replace("ṣ", "sˤ")
        form = form.replace("ṭ", "tˤ")
        form = form.replace("ḍ", "dˤ")
        form = form.replace("ẓ", "ðˤ")

        # Fricatives
        form = form.replace("ḥ", "ħ")
        form = form.replace("ḵ", "x")
        form = form.replace("ḡ", "ɣ")
        form = form.replace("ḏ", "ð")
        form = form.replace("ṯ", "θ")

        # Affricate jīm → /d͡ʒ/; then repurpose j as the IPA palatal approximant
        # for Arabic yāʾ.  Order matters — do j → d͡ʒ before y → j.
        form = form.replace("j", "d͡ʒ")
        form = form.replace("y", "j")

        # Vowel length
        form = form.replace("ā", "aː")
        form = form.replace("ī", "iː")
        form = form.replace("ū", "uː")

        return cls(word=form.strip())


class HebrewWord(Word):
    @property
    def lang(self) -> str:
        return "he"

    @classmethod
    def from_romanization(cls, text: str) -> Self:
        """Hebrew romanization → IPA.

        Kaikki Hebrew romanization marks stress with acute accents (not
        length), so accents are stripped.  Dagesh-forte gemination, where
        written as a doubled letter, is preserved via ː.
        """
        if not text:
            return cls(word=text)
        form = text.lower()

        # Parentheses in Hebrew romanization mark optional ayin enunciation;
        # make the enunciated form the default.
        form = form.replace("(", "").replace(")", "")

        # Circumflex vowels are used like macrons in some source romanizations.
        form = _promote_circumflex_vowels(form)

        # Scholarly Semitic letters used in shared-source romanizations.
        # these are slightly odd, since the hebrew evolution didn't follow the same path
        # we assign them slightly different values
        form = form.replace("ʾ", "ʔ")
        form = form.replace("ʿ", "ʕ")
        form = form.replace("ś", "s")
        form = form.replace("ḇ", "β")
        form = form.replace("ṯ", "t")
        form = form.replace("ḏ", "d")
        form = form.replace("ḵ", "x")
        form = form.replace("ḥ", "ħ")

        # Strip stress/extra accents while keeping macrons for length.
        form = _normalize_hebrew_vowel_marks(form)

        # Gemination on raw input, before digraph expansion.
        form = _geminate(form)

        # Digraphs (longest first).
        form = form.replace("tsh", "t͡ʃ")
        form = form.replace("ts", "t͡s")   # Hebrew tsade (affricate)
        form = form.replace("sh", "ʃ")
        form = form.replace("kh", "χ")     # modern Hebrew: uvular fricative
        form = form.replace("zh", "ʒ")

        # Drop kaikki schwa/syllable apostrophes (inconsistently used).
        form = form.replace("'", "")

        # Palatal approximant
        form = form.replace("y", "j")

        # Macron vowels mark length, unlike the acute stress marks above.
        form = form.replace("ā", "aː")
        form = form.replace("ī", "iː")
        form = form.replace("ū", "uː")
        form = form.replace("ē", "eː")
        form = form.replace("ō", "oː")

        form = form.strip()

        # Hebrew words don't start with a bare vowel phonetically — prepend ʔ.
        if form and form[0] in VOWELS:
            form = "ʔ" + form

        return cls(word=form)

    def to_pansemitic_ipa(self) -> str:
        # Hebrew tsade reflects the emphatic sibilant in the pansemitic layer,
        # unlike foreign /t͡s/ affricates.
        return self.word.replace("t͡s", "sˤ")


class SemProWord(Word):
    @property
    def lang(self) -> str:
        return "sem-pro"

    @classmethod
    def from_romanization(cls, text: str) -> Self:
        """proto-Semitic scholar notation → IPA."""
        if not text:
            return cls(word=text)

        # In sem-pro, capital `V` is the "unknown vowel" placeholder (see
        # Wiktionary Reconstruction:Proto-Semitic/ḏV-).  Substitute a central
        # vowel before lowercasing so it's not confused with consonant /v/.
        form = text.replace("V", "a").lower()

        # Strip reconstruction markers.
        form = form.lstrip("*").rstrip("-")

        # Strip acute stress accents (kaikki uses them on non-Semitic loans).
        form = _strip_acute_vowels(form)

        # Glottal / pharyngeal variants first
        form = form.replace("ʾ", "ʔ")
        form = form.replace("ʿ", "ʕ")
        form = form.replace("y", "j")

        # Gemination on raw input.
        form = _geminate(form)

        # Multi-char / combining scholar sequences first.
        # θ̣ and ṯ̣ both encode the emphatic interdental fricative.
        form = form.replace("θ̣", "θˤ")
        form = form.replace("ṯ̣", "θˤ")
        form = form.replace("ṣ́", "ɬˤ")     # emphatic lateral fricative

        # Single-char scholar → IPA
        form = form.replace("ṣ", "sˤ")
        form = form.replace("ṭ", "tˤ")
        form = form.replace("ḍ", "dˤ")
        form = form.replace("ṯ", "θ")
        form = form.replace("ḏ", "ð")
        form = form.replace("ḫ", "x")
        form = form.replace("ḥ", "ħ")
        form = form.replace("ś", "ɬ")
        form = form.replace("š", "ʃ")
        form = form.replace("ḡ", "ɣ")
        form = form.replace("ḳ", "q")

        # Vowel length
        form = form.replace("ā", "aː")
        form = form.replace("ī", "iː")
        form = form.replace("ū", "uː")
        form = form.replace("ē", "eː")
        form = form.replace("ō", "oː")
        form = form.replace("ô", "oː")

        # Final cleanup — strips any stray accents/stress marks that survive
        # from non-Semitic sources routed through SemProWord as a fallback
        # (Persian, Pahlavi, Sanskrit, Germanic, etc.).
        form = _strip_combining(form)

        return cls(word=form.strip())


class SemWesProWord(SemProWord):
    @property
    def lang(self) -> str:
        return "sem-wes-pro"

class ReconstructedSemProWord(SemProWord):
    @property
    def lang(self) -> str:
        return "recon-sem-pro"


_AKKADIAN_DETERMINATIVE_RE = re.compile(
    r"\^\([^)]*\)|\{[^}]*\}|[ᵈᶠᵐᵏᶫˢ]|[\u2070-\u209f]"
)


class AkkadianWord(SemProWord):
    @property
    def lang(self) -> str:
        return "akk"

    @classmethod
    def from_romanization(cls, text: str) -> Self:
        """Akkadian transliteration → IPA.

        Akkadian scholarly transliteration is close enough to the proto-Semitic
        path that we can reuse it after stripping determinatives/logogram
        markers (for example superscript ᵈ) and normalizing circumflex long
        vowels such as û to the macron series expected downstream.
        """
        if not text:
            return cls(word=text)

        form = text.strip()

        # Determinatives / logogram markers are orthographic and not pronounced.
        form = _AKKADIAN_DETERMINATIVE_RE.sub("", form)
        form = form.replace("^", "")
        form = form.replace(".", "")
        form = _promote_circumflex_vowels(form)

        base = SemProWord.from_romanization(form)
        return cls(word=base.word)


class ProtoItalicWord(Word):
    @property
    def lang(self) -> str:
        return "itc-pro"

    @classmethod
    def from_romanization(cls, text: str) -> Self:
        """Proto-Italic notation → IPA-ish Unicode.

        These entries are already close to IPA; the main normalization is
        flattening unknown-vowel V to a and converting macron long vowels to
        the repo's ː convention.
        """
        if not text:
            return cls(word=text)

        form = text.replace("V", "a").lower().lstrip("*").rstrip("-")
        form = _strip_acute_vowels(form)
        form = form.replace("ā", "aː")
        form = form.replace("ī", "iː")
        form = form.replace("ū", "uː")
        form = form.replace("ē", "eː")
        form = form.replace("ō", "oː")
        form = _strip_combining(form)
        return cls(word=form.strip())


class ProtoSouthDravidianWord(Word):
    @property
    def lang(self) -> str:
        return "dra-sou-pro"

    @classmethod
    def from_romanization(cls, text: str) -> Self:
        """Proto-South Dravidian notation → IPA-ish Unicode.

        This decoder keeps the broad contrastive structure while mapping the
        Dravidian retroflex series into explicit IPA. The reconstruction symbol
        V is treated as an unspecified vowel and flattened to a.
        """
        if not text:
            return cls(word=text)

        form = text.replace("V", "a").lower().lstrip("*").rstrip("-")
        form = _strip_acute_vowels(form)

        form = form.replace("ñ", "ɲ")
        form = form.replace("ṅ", "ŋ")
        form = form.replace("ṇ", "ɳ")
        form = form.replace("ṭ", "ʈ")
        form = form.replace("ḍ", "ɖ")
        form = form.replace("ḷ", "ɭ")
        form = form.replace("ṛ", "ɽ")
        form = form.replace("ẓ", "ɻ")

        # Dravidian ṯ is conventionally an alveolar stop/obstruent; we
        # approximate it as plain t in this IPA layer.
        form = form.replace("ṯ", "t")

        form = form.replace("y", "j")
        form = form.replace("ā", "aː")
        form = form.replace("ī", "iː")
        form = form.replace("ū", "uː")
        form = form.replace("ē", "eː")
        form = form.replace("ō", "oː")

        form = _strip_combining(form)
        return cls(word=form.strip())


class ProtoGermanicWord(Word):
    @property
    def lang(self) -> str:
        return "gem-pro"

    @classmethod
    def from_romanization(cls, text: str) -> Self:
        """Proto-Germanic notation → IPA-ish Unicode.

        This path is intentionally light-touch: preserve the near-IPA
        transliteration, map thorn to theta, normalize long vowels, and keep
        nasalized vowels explicit so the pansemitic reduction can drop them.
        """
        if not text:
            return cls(word=text)

        form = text.replace("V", "a").lower().lstrip("*").rstrip("-")
        form = _strip_acute_vowels(form)
        form = _promote_circumflex_vowels(form)
        form = _geminate(form)

        form = form.replace("þ", "θ")
        form = form.replace("ǭ", "õː")
        form = form.replace("ǫ", "õ")
        form = form.replace("ą", "ã")

        form = form.replace("ā", "aː")
        form = form.replace("ī", "iː")
        form = form.replace("ū", "uː")
        form = form.replace("ē", "eː")
        form = form.replace("ō", "oː")

        form = _strip_combining(form)
        return cls(word=form.strip())


class SumerianWord(Word):
    @property
    def lang(self) -> str:
        return "sux"

    @classmethod
    def from_romanization(cls, text: str) -> Self:
        """Sumerian transliteration → IPA-ish Unicode.

        The transliteration is already close to phonemic notation; we only map
        a handful of conventional Sumerological symbols and strip sign
        separators / gloss punctuation that are orthographic rather than
        phonetic.
        """
        if not text:
            return cls(word=text)

        form = text.lower().lstrip("*").rstrip("-")
        form = _strip_acute_vowels(form)
        form = _promote_circumflex_vowels(form)
        form = _geminate(form)

        form = form.replace("g̃", "ŋ")
        form = form.replace("ĝ", "ŋ")
        form = form.replace("ḫ", "x")
        form = form.replace("ř", "t͡sʰ")
        form = form.replace("z", "t͡s")
        form = form.replace("š", "ʃ")
        form = form.replace("y", "j")

        form = form.replace("ā", "aː")
        form = form.replace("ī", "iː")
        form = form.replace("ū", "uː")
        form = form.replace("ē", "eː")
        form = form.replace("ō", "oː")

        # Sumerological separators / optional gloss markers are not phonemic.
        form = form.replace(".", "").replace("(", "").replace(")", "")

        form = _strip_combining(form)
        return cls(word=form.strip())


_AFRASIANIST_TO_IPA: list[tuple[str, str]] = [
    # Multi-char / combining sequences first.
    ("i̭", "j"),
    ("ṯ̣", "θˤ"),
    ("ḏ̣", "ðˤ"),
    ("ć̣", "t͡sʼʲ"),
    ("č̣", "t͡ʃʼ"),
    ("c̣", "t͡sʼ"),
    ("ĉ̣", "t͡ɬʼ"),
    ("q̣", "qʼ"),
    ("x̣", "k͡xʼ"),
    ("ʒ̂", "d͡ɮ"),
    ("ʒ́", "d͡zʲ"),
    ("ṣ́", "sˤʲ"),
    ("ć", "t͡sʲ"),
    ("č", "t͡ʃ"),
    ("ĉ", "t͡ɬ"),
    ("l̀", "ɭ"),
    ("k̑", "q"),
    ("h̑", "χ"),
    # Single-char mappings.
    ("ɣ", "ʁ"),
    ("p̠", "ɸ"),
    ("ḇ", "β"),
    ("ṗ", "pʼ"),
    ("ḅ", "ɓ"),
    ("ṯ", "θ"),
    ("ḏ", "ð"),
    ("c", "t͡s"),
    ("ʒ", "d͡z"),
    ("ṣ", "sˤ"),
    ("ŝ", "ɬ"),
    ("ḡ", "ɣ"),
    ("ḳ", "kʼ"),
    ("q", "kʼ"),
    ("x", "k͡x"),
    ("9", "ɡ͡ɣ"),
    ("ś", "sʲ"),
    ("ź", "zʲ"),
    ("ń", "nʲ"),
    ("ĺ", "lʲ"),
    ("ŕ", "rʲ"),
    ("ǹ", "ɳ"),
    ("ṷ", "w"),
    ("y", "j"),
    ("ḥ", "ħ"),
]


class AfrasianWord(Word):
    @property
    def lang(self) -> str:
        return "afa-pro"

    @classmethod
    def from_romanization(cls, text: str) -> Self:
        """proto-Afrasianist notation → IPA.

        Afrasianist symbols overlap with proto-Semitic scholarship but are not
        equivalent (for example c = /t͡s/, ʒ = /d͡z/, q/ḳ = /kʼ/).  Keep this as
        a separate decoder so SemProWord can stay faithful to proto-Semitic
        conventions. Tone-marked vowels such as ê/ě/ē are flattened because
        the internal IPA layer does not model tone.
        """
        if not text:
            return cls(word=text)

        # As with proto-Semitic entries, treat capital V as an unspecified
        # vowel placeholder and collapse it to a concrete vowel for downstream
        # tokenization and pansemitic reduction.
        form = text.replace("V", "a").lower()
        form = form.lstrip("*").rstrip("-")
        form = _strip_acute_vowels(form)
        form = _strip_tone_vowels(form)
        form = form.replace("ʾ", "ʔ")
        form = form.replace("ʿ", "ʕ")

        # Preserve doubled consonants before expanding Afrasianist multigraphs.
        form = _geminate(form)

        for src, dst in _AFRASIANIST_TO_IPA:
            form = form.replace(src, dst)

        form = _strip_combining(form)
        return cls(word=form.strip())


# ── Greek script → proto-Semitic-compatible romanization ─────────
# Maps Greek as received by Semitic speakers (loanwords):
#   φ→p (not f), θ→t, χ→ḫ, aspirates lost, vowels collapsed to a/i/u
_GREEK_MAP = {
    'α': 'a', 'β': 'b', 'γ': 'g', 'δ': 'd', 'ε': 'i', 'ζ': 'z',
    'η': 'i', 'θ': 't', 'ι': 'i', 'κ': 'k', 'λ': 'l', 'μ': 'm',
    'ν': 'n', 'ξ': 'ks', 'ο': 'a', 'π': 'p', 'ρ': 'r', 'σ': 's',
    'ς': 's', 'τ': 't', 'υ': 'u', 'φ': 'p', 'χ': 'x', 'ψ': 'ps',
    'ω': 'a',
}


class GreekWord(Word):
    @property
    def lang(self) -> str:
        return "grc"

    @classmethod
    def from_greek(cls, text: str) -> Self:
        """Normalize Greek script to proto-Semitic-compatible encoding.

        Strips accents/breathing marks via NFD decomposition, then maps
        each base Greek letter through _GREEK_MAP.
        """
        if not text:
            return cls(word=text)

        # NFD decompose to separate combining accents, then strip them
        decomposed = unicodedata.normalize("NFD", text.lower())
        stripped = "".join(
            c for c in decomposed
            if unicodedata.category(c) != "Mn"  # drop combining marks
        )

        # Map Greek letters; drop anything not in the map (hyphens, spaces, etc.)
        form = "".join(_GREEK_MAP.get(c, '') for c in stripped)

        # Preserve gemination via ː.
        form = _geminate(form)

        return cls(word=form.strip())


# ── Egyptian transliteration → IPA (Egyptological pronunciation) ─
# Egyptological convention reads ꜣ, ꜥ, j, ı͗ as bare vowels /a/ or /i/,
# regardless of their phonetic reconstruction.  That's what the user sees in
# romanizations like "zbꜣt", "zbꜥwt" — we preserve that reading.
_EGYPTIAN_MAP: list[tuple[str, str]] = [
    # Multi-char / composed first
    ("ı͗", "i"),
    # Egyptian-specific consonants
    ("ꜣ", "a"),      # aleph → /a/
    ("ꜥ", "a"),      # ayin  → /a/
    ("ḥ", "ħ"),      # emphatic h → voiceless pharyngeal fricative
    ("ḫ", "x"),      # voiceless velar fricative
    ("ẖ", "ç"),      # palatal fricative
    ("š", "ʃ"),
    ("ṯ", "t͡ʃ"),    # affricate (later merged with t)
    ("ḏ", "d͡ʒ"),    # affricate (later merged with d)
    ("ṱ", "tˤ"),     # emphatic t (rare)
    ("q", "q"),
    # Semivowels
    ("j", "i"),      # Egyptological: j → /i/
    ("y", "j"),      # y → palatal approximant
]


class EgyptianWord(Word):
    @property
    def lang(self) -> str:
        return "egy"

    @classmethod
    def from_romanization(cls, text: str) -> Self:
        """Egyptian transliteration → IPA (Egyptological reading)."""
        if not text:
            return cls(word=text)
        form = text.replace("V", "a").lower().lstrip("*").rstrip("-")
        form = _geminate(form)
        for src, dst in _EGYPTIAN_MAP:
            form = form.replace(src, dst)
        form = _strip_combining(form)
        return cls(word=form.strip())


# ── Proto-Indo-European → IPA (best-guess reconstruction) ───────
# Laryngeals: h₁ = /h/ (neutral), h₂ = /χ/ (uvular), h₃ = /ʕ/ (voiced
# pharyngeal) — the most common phonetic guesses.  Palatovelars collapse
# to plain velars since the distinction doesn't survive into Semitic.
_PIE_MAP: list[tuple[str, str]] = [
    # Laryngeals (multi-char)
    ("h₁", "h"),
    ("h₂", "χ"),
    ("h₃", "ʕ"),
    # Labiovelars (preserve labialization)
    ("gʷʰ", "gʷʰ"),
    ("kʷ", "kʷ"),
    ("gʷ", "gʷ"),
    # Aspirated (voiced) stops keep ʰ
    ("bʰ", "bʰ"),
    ("dʰ", "dʰ"),
    ("ǵʰ", "gʰ"),
    ("gʰ", "gʰ"),
    # Palatovelars → plain velars (collapse)
    ("ḱ", "k"),
    ("ǵ", "g"),
    # Semivowels
    ("i̯", "j"),
    ("u̯", "w"),
    ("y", "j"),
    # Syllabic resonants
    ("l̥", "l̩"),
    ("r̥", "r̩"),
    ("m̥", "m̩"),
    ("n̥", "n̩"),
    # Vowel length
    ("ē", "eː"),
    ("ō", "oː"),
    ("ā", "aː"),
    ("ī", "iː"),
    ("ū", "uː"),
]


class PieWord(Word):
    @property
    def lang(self) -> str:
        return "ine-pro"

    @classmethod
    def from_romanization(cls, text: str) -> Self:
        """PIE scholar notation → IPA (best-guess)."""
        if not text:
            return cls(word=text)
        form = text.lower().lstrip("*").rstrip("-")
        # Strip acute stress accents.
        form = (form
                .replace("á", "a").replace("é", "e").replace("í", "i")
                .replace("ó", "o").replace("ú", "u"))
        form = _geminate(form)
        for src, dst in _PIE_MAP:
            form = form.replace(src, dst)
        form = _strip_combining(form)
        return cls(word=form.strip())


class IranianWord(Word):
    @property
    def lang(self) -> str:
        return "ira-pro"

    @classmethod
    def from_romanization(cls, text: str) -> Self:
        """Proto-Iranian reconstruction notation → IPA-ish Unicode.

        Pragmatic choices for this project:
        - capital H (laryngeal placeholder) is flattened to h
        - both c and č are accepted as /t͡ʃ/, since local data contains plain c
        - r̥ / l̥ are kept as syllabic resonants, as in the PIE path
        """
        if not text:
            return cls(word=text)
        form = text.lower().lstrip("*").rstrip("-")
        form = _strip_acute_vowels(form)

        # Preserve doubled scholarly letters before segment expansion.
        form = _geminate(form)

        # Iranian-specific segments and local reconstruction conventions.
        form = form.replace("ǰ", "d͡ʒ")
        form = form.replace("č", "t͡ʃ")
        form = form.replace("c", "t͡ʃ")
        form = form.replace("š", "ʃ")
        form = form.replace("y", "j")

        # Syllabic resonants and vowel length.
        form = form.replace("r̥̄", "r̩ː")
        form = form.replace("l̥̄", "l̩ː")
        form = form.replace("r̥", "r̩")
        form = form.replace("l̥", "l̩")
        form = form.replace("ā", "aː")
        form = form.replace("ī", "iː")
        form = form.replace("ū", "uː")
        
        # Old Median Δ
        form = form.replace("δ", "d") 

        form = _strip_combining(form)
        return cls(word=form.strip())


class OldPersianWord(Word):
    @property
    def lang(self) -> str:
        return "peo"

    @classmethod
    def from_romanization(cls, text: str) -> Self:
        """Old Persian transliteration → IPA-ish Unicode.

        Uses Old Persian scholarly conventions where c = /t͡ʃ/, j = /d͡ʒ/,
        y = /j/, and macrons mark vowel length.
        """
        if not text:
            return cls(word=text)
        form = text.lower().lstrip("*").rstrip("-")
        form = _strip_acute_vowels(form)

        form = _geminate(form)

        form = form.replace("c", "t͡ʃ")
        form = form.replace("j", "d͡ʒ")
        form = form.replace("š", "ʃ")
        form = form.replace("y", "j")

        form = form.replace("ā", "aː")
        form = form.replace("ī", "iː")
        form = form.replace("ū", "uː")

        form = _strip_combining(form)
        return cls(word=form.strip())


# ── Cyrillic → IPA ──────────────────────────────────────────────
# Handles Russian (ru), Old East Slavic (orv), and Proto-Slavic (sla-pro)
# — the latter two sometimes use mixed Latin / Cyrillic transliterations
# in kaikki data (e.g. sla-pro:*cěsařь).  Cyrillic chars get mapped;
# stray Latin chars pass through untouched.

_CYRILLIC_MAP: list[tuple[str, str]] = [
    # Iotated vowels (multi-output; order before plain vowel maps)
    ("ю", "ju"),
    ("я", "ja"),
    ("ё", "jo"),
    ("є", "je"),
    # Affricates & sibilants (IPA digraphs)
    ("щ", "ʃː"),
    ("ч", "t͡ʃ"),
    ("ц", "t͡s"),
    ("ж", "ʒ"),
    ("ш", "ʃ"),
    ("х", "x"),
    # Plain consonants
    ("б", "b"), ("в", "v"), ("г", "g"), ("д", "d"),
    ("з", "z"), ("й", "j"), ("к", "k"), ("л", "l"),
    ("м", "m"), ("н", "n"), ("п", "p"), ("р", "r"),
    ("с", "s"), ("т", "t"), ("ф", "f"),
    # Vowels
    ("а", "a"), ("е", "e"), ("и", "i"), ("о", "o"),
    ("у", "u"), ("ы", "i"), ("э", "e"),
    ("ѣ", "e"),   # yat (Old East Slavic / early Russian)
    # Yers and soft sign — drop (palatalization not tracked)
    ("ъ", ""), ("ь", ""),
    # Scholarly Latin transliterations of Cyrillic (kaikki's tr field uses
    # this form for Russian / Old East Slavic / Proto-Slavic: ʹ = soft sign,
    # ʺ = hard sign, č/š/ž/c = Slavic affricates/sibilants, ě = yat).
    ("ʹ", ""), ("ʺ", ""),
    ("č", "t͡ʃ"),
    ("š", "ʃ"),
    ("ž", "ʒ"),
    ("ě", "e"),
    ("ř", "r"),
    ("ć", "t͡ɕ"),
    ("ś", "ɕ"),
    ("ź", "ʑ"),
    ("ń", "n"),
    ("ł", "w"),
    ("c", "t͡s"),
]


class CyrillicWord(Word):
    @property
    def lang(self) -> str:
        return "ru"

    @classmethod
    def from_cyrillic(cls, text: str) -> Self:
        """Cyrillic (or mixed Cyrillic/Latin) → IPA."""
        if not text:
            return cls(word=text)
        form = text.lower().lstrip("*").rstrip("-")
        form = _geminate(form)
        for src, dst in _CYRILLIC_MAP:
            form = form.replace(src, dst)
        form = _strip_combining(form)
        return cls(word=form.strip())


# ── Aramaic (Hebrew script) → proto-Semitic encoding ────────────
_ARAMAIC_CONSONANTS = {
    'א': 'ʔ', 'ב': 'b', 'ג': 'g', 'ד': 'd', 'ה': 'h', 'ו': 'w',
    'ז': 'z', 'ח': 'ħ', 'ט': 'tˤ', 'י': 'j', 'כ': 'k', 'ך': 'k',
    'ל': 'l', 'מ': 'm', 'ם': 'm', 'נ': 'n', 'ן': 'n', 'ס': 's',
    'ע': 'ʕ', 'פ': 'p', 'ף': 'p', 'צ': 'sˤ', 'ץ': 'sˤ', 'ק': 'q',
    'ר': 'r', 'ש': 'ʃ', 'ת': 't',
}

# Nikkud (vowel points) → IPA vowels (e and o preserved).
_ARAMAIC_VOWELS = {
    '\u05B7': 'a',   # patach
    '\u05B8': 'a',   # qamats
    '\u05B6': 'e',   # segol
    '\u05B5': 'e',   # tsere
    '\u05B4': 'i',   # hiriq
    '\u05B9': 'o',   # holam
    '\u05BB': 'u',   # qubuts
    '\u05B2': 'a',   # hataf patach
    '\u05B1': 'e',   # hataf segol
    '\u05B3': 'a',   # hataf qamats
}

# Dagesh: gemination marker on the preceding consonant.
_ARAMAIC_DAGESH = '\u05BC'

# Nikkud and marks to skip (shva, rafe, etc.)
_ARAMAIC_SKIP = {
    '\u05B0',  # shva
    '\u05BF',  # rafe
    '\u05BD',  # meteg
    '\u05C1',  # shin dot
    '\u05C2',  # sin dot
}


class AramaicWord(Word):
    @property
    def lang(self) -> str:
        return "arc"

    @classmethod
    def from_aramaic(cls, text) -> Self:
        """Aramaic (Hebrew script + nikkud) → IPA.  Dagesh preserved as ː."""
        if not text:
            return cls(word=text)

        result: list[str] = []
        for c in text:
            if c in _ARAMAIC_CONSONANTS:
                result.append(_ARAMAIC_CONSONANTS[c])
            elif c in _ARAMAIC_VOWELS:
                result.append(_ARAMAIC_VOWELS[c])
            elif c == _ARAMAIC_DAGESH:
                # Gemination marker on the preceding consonant.  We can't
                # distinguish dagesh-lene from dagesh-forte without morphology,
                # so treat both as ː — preserves what's written.
                result.append("ː")
            elif c in _ARAMAIC_SKIP:
                continue
            # else: ignore (maqaf, sof pasuq, etc.)

        form = "".join(result)

        # Strip trailing ʔ (Aramaic emphatic state -א).
        form = form.rstrip('ʔ')

        return cls(word=form.strip())


class GenericWord(Word):
    """A Word with an arbitrary lang tag and generic normalization paths.

    Used for languages that have no dedicated subclass (fr, la, sa, fa, …).
    We still want a normalized IPA-ish representation, but we must preserve the
    original language tag so language-specific pansemitic rewrites (such as
    Semitic ``f -> p``) do not accidentally fire for unrelated languages.
    """

    def __init__(self, word: str, lang: str):
        self.word = word
        self._lang_tag = lang

    @classmethod
    def from_ipa(cls, ipa: str, lang: str) -> "GenericWord":
        return cls(word=_strip_combining(ipa), lang=lang)

    @classmethod
    def from_romanization(cls, text: str, lang: str) -> "GenericWord":
        # Reuse the broad SemProWord transliteration cleanup so dotted / marked
        # scholarly Latin input gets mapped into the same IPA-ish internal
        # alphabet, but preserve the original language tag.
        base = SemProWord.from_romanization(text)
        return cls(word=base.word, lang=lang)

    @property
    def lang(self) -> str:
        return self._lang_tag


_PANSEMITIC_IPA_TO_SCHOLAR: list[tuple[str, str]] = [
    ("sˤ", "ṣ"),
    ("tˤ", "ṭ"),
    # IPA palatal approximant → Semitic y (before d͡ʒ → j so the jīm we
    # produce isn't re-mapped to y).
    ("j", "y"),
    ("d͡ʒ", "j"),
    ("ʃ", "š"),
    # Pansemitic keeps `x` as-is (merged dorsal fricative) rather than ḫ.
    # Strip any leftover tie bar (from a foreign affricate not in our
    # inventory).  Runs last so the d͡ʒ → j rule above gets first crack.
    ("͡", ""),
]



# IPA → pansemitic IPA.  Lossy: compresses the consonant inventory to the
# pansemitic phoneme set, collapses non-{a,i,u} vowels.  Length and
# gemination are stripped separately by `PansemiticWord.from_word`.
#
# Pansemitic phoneme inventory (all IPA):
#   vowels: a i u
#   stops:  p b t d k g q ʔ   tˤ dˤ(→sˤ)
#   fric.:  f s z ʃ x  sˤ   (ħ/χ → x; θ/ð/ʒ → s/z; ɬ → s)
#   affr.:  d͡ʒ          (foreign affricates otherwise unfold / collapse)
#   other:  m n l r w j h ʕ ( v -> w, ɦ -> h)
_IPA_TO_PANSEMITIC_IPA: list[tuple[str, str]] = [
    # emphatics collapse (multi-char first).  sˤ and tˤ are inventory; other
    # emphatics fold into sˤ.
    ("ðˤ", "sˤ"),
    ("θˤ", "sˤ"),
    ("ɬˤ", "sˤ"),
    ("dˤ", "sˤ"),
    ("t͡ɬʼ", "sˤ"),
    ("t͡ʃʼ", "sˤ"),
    ("t͡sʼ", "sˤ"),
    # affricates: d͡ʒ is inventory; foreign t͡s unfolds to the cluster ts,
    # while the rest fold toward the nearest pansemitic segment.
    ("t͡ʃ", "ʃ"),
    ("t͡s", "ts"),
    ("t͡ɬ", "tl"),
    ("d͡z", "z"),
    ("d͡ɮ", "z"),
    # voiceless dorsals merge → x
    ("k͡xʼ", "x"),
    ("k͡x", "x"),
    ("χ", "x"),
    ("ħ", "x"),
    # voiced velar fricative → g; IPA single-story g variant → plain g
    ("ɡ͡ɣ", "g"),
    ("ɣ", "g"),
    ("ɡ", "g"),
    # IPA segments that can leak in from scholarly or generic fallback paths.
    ("β", "b"),
    ("ɓ", "b"),
    ("ɸ", "p"),
    ("ç", "x"),
    ("c", "k"),
    ("kʼ", "q"),
    ("qʼ", "q"),
    ("pʼ", "p"),
    # Non-a/i/u IPA vowels collapse toward the nearest e/o/a/i/u base.
    ("ɪ", "i"), ("ʏ", "i"), ("ɨ", "i"),
    ("ʊ", "u"), ("ɯ", "u"),
    ("ɛ", "e"), ("œ", "e"), ("ø", "e"),
    ("ɔ", "o"),
    ("ɐ", "a"), ("ɑ", "a"), ("ɒ", "a"), ("æ", "a"), ("ə", "a"),
    ("ɕ", "s"), ("ʑ", "z"),   # alveolo-palatal fricatives
    ("ɫ", "l"),               # velarized lateral
    # Rhotics collapse → r
    ("ɹ", "r"), ("ɾ", "r"), ("ʀ", "r"), ("ʁ", "r"), ("ɻ", "r"), ("ɽ", "r"),
    # Nasals
    ("ŋ", "n"), ("ɲ", "n"), ("ɳ", "n"),
    # Retroflex stops → dental
    ("ʈ", "t"), ("ɖ", "d"),
    # Retroflex laterals collapse → l
    ("ɭ", "l"),
    # Rhotic schwa / low vowel
    ("ɚ", "a"), ("ʌ", "a"),
    # Aspiration / palatalization aren't phonemic for Semitic — drop.
    ("ʰ", ""), ("ʱ", ""), ("ʲ", ""), ("ˠ", ""),
    ("ʼ", ""),
    ("̃", ""),
    # We are not preserving labiovelars in the pansemitic layer here.
    ("ʷ", ""),
    # Superscript letters used as IPA release/off-glide marks — drop.
    ("ᵗ", ""), ("ⁱ", ""), ("ⁿ", ""),
    # (Tie bar on d͡ʒ is preserved — d͡ʒ is in the pansemitic inventory.)
    # interdentals → sibilants
    ("θ", "s"),
    ("ð", "z"),
    # lateral fricative → s.  (ʃ is preserved; standalone ʒ is handled below
    # so d͡ʒ stays intact.)
    ("ɬ", "s"),
    # v → w
    ("v", "w"),
    # voiced glottal fricative → h
    ("ɦ", "h"),
    # Remove resonants
    ("l̩", "l"), ("r̩", "r"), ("m̩", "m"), ("n̩", "n"),
]

class PansemiticWord(Word):
    """A word reduced to the pansemitic phoneme inventory, stored as IPA.

    Built from any ancestor Word via `PansemiticWord.from_word`.  Apply the
    pansemitic phonetic compressions in IPA so that downstream consumers
    (notably the loss function in `loss.py`) can work uniformly in IPA.
    """

    @property
    def lang(self) -> str:
        return "pansemitic"

    def to_protosemitic_convention(self) -> str:
        """Human-readable pansemitic rendering.  Uses a bespoke table: the
        generic scholar mapping would fold x → ḫ, but pansemitic keeps x."""
        out = self.word
        for src, dst in _PANSEMITIC_IPA_TO_SCHOLAR:
            out = out.replace(src, dst)
        return out

    @classmethod
    def from_word(cls, ancestor: Word) -> "PansemiticWord":
        if not ancestor.word:
            return cls(word="")
        form = ancestor.to_pansemitic_ipa()

        # Semitic-family sources: f reflects older *p.
        if ancestor.lang in ("sem-pro", "sem-wes-pro", "ar", "akk", "arc"):
            form = form.replace("f", "p")

        for src, dst in _IPA_TO_PANSEMITIC_IPA:
            form = form.replace(src, dst)

        # Standalone /ʒ/ is not pansemitic; keep affricate /d͡ʒ/ untouched.
        form = re.sub(r"(?<!͡)ʒ", "d͡ʒ", form)

        # Keep inventory emphatics sˤ and tˤ, but drop stray pharyngealization
        # marks that leak in on other consonants.
        form = re.sub(r"(?<![st])ˤ", "", form)

        # Collapse vowels to a/i/u — long forms first so eː → i, oː → a.
        form = form.replace("eː", "i").replace("oː", "a")
        form = form.replace("aː", "a").replace("iː", "i").replace("uː", "u")
        form = form.replace("e", "i").replace("o", "a")

        # Drop remaining ː (consonant gemination), then dedupe any identical
        # consonants introduced by lowering rules and finally dedupe vowels.
        form = form.replace("ː", "")
        form = _dedupe_adjacent_consonants(form)
        form = re.sub(r"([aiu])\1+", r"\1", form)

        form = form.strip()
        # Pansemitic words don't start with a bare vowel — prepend a glottal
        # stop to match Semitic phonotactics.
        if form and form[0] in "aiu":
            form = "ʔ" + form

        return cls(word=form)


def word_from_sharedsource(src: SharedSource) -> Word:
    """Build the language-appropriate Word for a shared etymology source.

    Prefers native IPA from kaikki when available; falls back to romanization
    and finally to script-specific decoders.  GenericWord catches everything
    that has no dedicated subclass but does carry IPA or romanization.
    """
    match src.lang:
        case "ar":
            if src.ipa:
                return ArabicWord.from_ipa(src.ipa)
            if src.romanization:
                return ArabicWord.from_romanization(src.romanization)
            raise MissingRomanizationError("arabic")
        case "he":
            if src.ipa:
                return HebrewWord.from_ipa(src.ipa)
            if src.romanization:
                return HebrewWord.from_romanization(src.romanization)
            raise MissingRomanizationError("hebrew")
        case "akk":
            if src.ipa:
                return AkkadianWord.from_ipa(src.ipa)
            if src.romanization:
                return AkkadianWord.from_romanization(src.romanization)
            raise MissingRomanizationError("akkadian")
        case "sux":
            if src.ipa:
                return SumerianWord.from_ipa(src.ipa)
            if src.romanization:
                return SumerianWord.from_romanization(src.romanization)
            raise MissingRomanizationError("sumerian")
        case "sem-pro" |  "qfa-hur-pro":
            if src.ipa:
                return SemProWord.from_ipa(src.ipa)
            return SemProWord.from_romanization(src.word)
        case "sem-wes-pro":
            if src.ipa:
                return SemWesProWord.from_ipa(src.ipa)
            return SemWesProWord.from_romanization(src.word)
        case "afa-pro":
            if src.ipa:
                return AfrasianWord.from_ipa(src.ipa)
            return AfrasianWord.from_romanization(src.word)
        case "grc":
            if src.ipa:
                return GreekWord.from_ipa(src.ipa)
            return GreekWord.from_greek(src.word)
        case "arc":
            if src.ipa:
                return AramaicWord.from_ipa(src.ipa)
            return AramaicWord.from_aramaic(src.word)
        case "egy":
            if src.ipa:
                return EgyptianWord.from_ipa(src.ipa)
            if src.romanization:
                return EgyptianWord.from_romanization(src.romanization)
            raise MissingRomanizationError("egyptian")
        case "ine-pro":
            if src.ipa:
                return PieWord.from_ipa(src.ipa)
            if src.romanization:
                return PieWord.from_romanization(src.romanization)
            raise MissingRomanizationError("proto-indo-european")
        case "itc-pro":
            if src.ipa:
                return ProtoItalicWord.from_ipa(src.ipa)
            return ProtoItalicWord.from_romanization(src.word)
        case "gem-pro":
            if src.ipa:
                return ProtoGermanicWord.from_ipa(src.ipa)
            return ProtoGermanicWord.from_romanization(src.word)
        case "ira-pro" | "xme-old":
            if src.ipa:
                return IranianWord.from_ipa(src.ipa)
            if src.romanization:
                return IranianWord.from_romanization(src.romanization)
            raise MissingRomanizationError("proto-iranian")
        case "dra-sou-pro":
            if src.ipa:
                return ProtoSouthDravidianWord.from_ipa(src.ipa)
            return ProtoSouthDravidianWord.from_romanization(src.word)
        case "peo" | "pal" | "fa-cls" | "fa":
            if src.ipa:
                return OldPersianWord.from_ipa(src.ipa)
            if src.romanization:
                return OldPersianWord.from_romanization(src.romanization)
            raise MissingRomanizationError("old-persian")
        case "ru" | "orv" | "sla-pro":
            return CyrillicWord.from_cyrillic(src.word)
        case _:
            if src.ipa:
                return GenericWord.from_ipa(src.ipa, lang=src.lang)
            print(src.lang, src.word)
            raise UnsupportedLanguageError(src.lang)


def reconstruct_ancestor(
    ar_roman: str,
    he_roman: str,
    ancestor: Word | None = None,
) -> Word:
    """Return the best ancestor form.

    Priority:
      1. Pre-built ancestor Word (built by the caller from a shared etymology
         source — the LCA of the borrowing/inheritance graph)
      2. Reconstruction from Arabic/Hebrew romanizations
    """

    # 1. Pre-built ancestor from a shared etymology source.
    if ancestor is not None:
        if not ancestor.word:
            raise EmptyAncestorError()
        return ancestor

    # 2. Merge Arabic and Hebrew romanizations
    if not ar_roman or not he_roman:
        missing = "both" if (not ar_roman and not he_roman) else ("arabic" if not ar_roman else "hebrew")
        raise MissingRomanizationError(missing)

    ar = ArabicWord.from_romanization(ar_roman)
    he = HebrewWord.from_romanization(he_roman)

    if ar.countconsonants() != he.countconsonants():
        raise ConsonantMismatchError(ar.countconsonants(), he.countconsonants())

    result = merge_roots(ar, he)
    if not result or not result.word:
        raise EmptyAncestorError()
    return result


def _tokenize_phonemes(ipa: str) -> list[str]:
    """Split an IPA string into phoneme tokens.

    A token is a base (single letter or tie-bar digraph like d͡ʒ) plus any
    trailing modifier letters (ˤ, ː).  Non-letter / non-modifier characters
    (whitespace, punctuation, reconstruction markers) are skipped.
    """
    tokens: list[str] = []
    i = 0
    n = len(ipa)
    while i < n:
        c = ipa[i]
        if not c.isalpha() and c not in "ʔʕ":
            i += 1
            continue
        # Tie-bar digraph: base + U+0361 + base
        if i + 2 < n and ipa[i + 1] == '͡':
            two = ipa[i] + ipa[i + 1] + ipa[i + 2]
            if two in IPA_CONSONANT_DIGRAPHS:
                phon = two
                i += 3
                while i < n and ipa[i] in IPA_MODIFIERS:
                    phon += ipa[i]
                    i += 1
                tokens.append(phon)
                continue
        phon = c
        i += 1
        while i < n and ipa[i] in IPA_MODIFIERS:
            phon += ipa[i]
            i += 1
        tokens.append(phon)
    return tokens


def _is_vowel_token(tok: str) -> bool:
    return bool(tok) and tok[0] in VOWELS


def _dedupe_adjacent_consonants(ipa: str) -> str:
    """Collapse identical adjacent consonants while preserving unknown chars."""
    out: list[str] = []
    prev: str | None = None
    i = 0
    n = len(ipa)
    while i < n:
        c = ipa[i]
        if not c.isalpha() and c not in "ʔʕ":
            out.append(c)
            prev = None
            i += 1
            continue

        if i + 2 < n and ipa[i + 1] == '͡':
            two = ipa[i] + ipa[i + 1] + ipa[i + 2]
            if two in IPA_CONSONANT_DIGRAPHS:
                tok = two
                i += 3
                while i < n and ipa[i] in IPA_MODIFIERS:
                    tok += ipa[i]
                    i += 1
                if prev == tok and not _is_vowel_token(tok):
                    continue
                out.append(tok)
                prev = tok
                continue

        tok = c
        i += 1
        while i < n and ipa[i] in IPA_MODIFIERS:
            tok += ipa[i]
            i += 1
        if prev == tok and not _is_vowel_token(tok):
            continue
        out.append(tok)
        prev = tok
    return "".join(out)


def extract_phonemes(word: Word) -> list[tuple[str, str | None]]:
    """Extract (consonant, vowel) pairs from a Word's IPA string.

    Each consonant phoneme is paired with its immediately-following vowel
    phoneme, or None if no vowel follows (consonant clusters).  A leading
    bare vowel becomes ("", vowel).
    """
    tokens = _tokenize_phonemes(word.word)
    result: list[tuple[str, str | None]] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if _is_vowel_token(tok):
            result.append(("", tok))
            i += 1
        else:
            vowel: str | None = None
            if i + 1 < len(tokens) and _is_vowel_token(tokens[i + 1]):
                vowel = tokens[i + 1]
                i += 2
            else:
                i += 1
            result.append((tok, vowel))
    return result


def reconcile_phoneme(ar_c: str, he_c: str) -> str:
    """Reconcile a consonant pair between Arabic and Hebrew.

    If both are the same, return it. Otherwise apply known correspondences
    to recover the proto-Semitic form.
    """
    if ar_c == he_c:
        return ar_c

    pair = (ar_c, he_c)
    match pair:
        case ("w", "j"):
            return "w"
        case ("p", "p") | ("f", "p"):
            return "p"
        case ("d͡ʒ", "g"):
            return "g"
        case _:
            # Default: prefer Arabic (more conservative)
            return ar_c


def reconcile_vowel(ar_v: str | None, he_v: str | None) -> str | None:
    """Reconcile a vowel pair. Prefer Arabic; fall back to Hebrew if Arabic is null."""
    if ar_v is not None:
        return ar_v
    return he_v


def merge_roots(ar: ArabicWord, he: HebrewWord) -> Word:
    """Merge Arabic and Hebrew normalized forms into a reconstructed ancestor.

    Extracts phonemes from both, aligns by consonant position, reconciles
    each pair, and reassembles.
    """    
    ar_phon = extract_phonemes(ar)
    he_phon = extract_phonemes(he)

    result = []
    for (ar_c, ar_v), (he_c, he_v) in zip(ar_phon, he_phon):
        result.append(reconcile_phoneme(ar_c, he_c))
        v = reconcile_vowel(ar_v, he_v)
        if v is not None:
            result.append(v)

    return ReconstructedSemProWord(word="".join(result))
    
