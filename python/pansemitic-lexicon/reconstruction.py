from __future__ import annotations

import re
import unicodedata
from typing import Self, TYPE_CHECKING
from dataclasses import dataclass

from numpy import isin

from kaikki import IpaRealization, _geminate
from loss import Consonant, Phoneme, Vowel, VOWELS, IPA_MODIFIERS, IPA_CONSONANT_DIGRAPHS, trace_cost

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


def _expand_triliteral_root(form: str) -> str:
    """Expand a vowelless 3-consonant root template to CaCaCa.

    Accepts either explicit-hyphen form (``k-t-b``) or bare CCC (``ktb``).
    Multi-byte scholarly consonants (ṣ, ṭ, ṯ, ḫ, ʔ, …) count as one slot
    via NFD base+combining segmentation. Anything that already contains
    a vowel, or whose consonant count isn't 3, is returned unchanged.
    """
    if not form:
        return form
    nfd = unicodedata.normalize("NFD", form)
    if any(c in "aeiou" for c in nfd):
        return form

    if "-" in form:
        parts = form.split("-")
        if len(parts) == 3 and all(parts):
            return "a".join(parts) + "a"
        return form

    tokens: list[str] = []
    cur = ""
    for c in nfd:
        if unicodedata.category(c) == "Mn":
            cur += c
        else:
            if cur:
                tokens.append(cur)
            cur = c
    if cur:
        tokens.append(cur)

    if len(tokens) == 3:
        return unicodedata.normalize("NFC", "a".join(tokens) + "a")
    return form


_ARABIC_DIACRITICS = re.compile(r"[ً-ٰٟـ]")
_HEBREW_DIACRITICS = re.compile(r"[֑-ֽֿ-ׇ]")
# Syriac vowel marks and combining diacritics in the Syriac block (U+0700–U+074F).
_SYRIAC_DIACRITICS = re.compile(r"[ܑܰ-݊]")


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

        # eliminate parens
        ipa = ipa.replace("(", "").replace(")", "")

        ipa = _strip_combining(ipa).strip()

        # Apply hard attack to all initial vowels for synchronization purposes.
        if ipa and ipa[0] in VOWELS:
            ipa = "ʔ" + ipa

        return cls(word=ipa)

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
        return sum(1 for phon in Phoneme.parse(self.word) if isinstance(phon, Consonant))

    def __str__(self) -> str:
        return f"{self.lang}:{self.word}"


class ArabicWord(Word):
    @property
    def lang(self) -> str:
        return "ar"

    @classmethod
    def normalize(cls, text: str) -> str:
        """Strip Arabic harakat / tatweel — returns the bare consonantal skeleton."""
        return _ARABIC_DIACRITICS.sub("", text).strip()

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
        form = form.replace("š", "ʃ")

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
        form = form.replace("ē", "eː")
        form = form.replace("ō", "oː")

        # replace some alternate notation:
        form = form.replace("ʾ", "ʔ")  # hamza

        return cls.from_ipa(form)


class HebrewWord(Word):
    @property
    def lang(self) -> str:
        return "he"

    @classmethod
    def normalize(cls, text: str) -> str:
        """Strip niqud / cantillation — returns the bare consonantal skeleton."""
        return _HEBREW_DIACRITICS.sub("", text).strip()

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
        form = form.replace("ʻ", "ʕ")
        form = form.replace("ś", "s")
        form = form.replace("š", "ʃ")
        form = form.replace("ḇ", "β")
        form = form.replace("ṯ", "t")
        form = form.replace("ḏ", "d")
        form = form.replace("ḵ", "x")
        form = form.replace("ḥ", "ħ")
        form = form.replace("p̄", "p")

        # Strip stress/extra accents while keeping macrons for length.
        form = _normalize_hebrew_vowel_marks(form)

        # Gemination on raw input, before digraph expansion.
        form = _geminate(form)

        # Digraphs (longest first).
        form = form.replace("tsh", "t͡ʃ")
        form = form.replace("ts", "t͡s")   # Hebrew tsade (affricate)
        form = form.replace("tz", "t͡s")   # Hebrew tsade (affricate)
        form = form.replace("ch", "t͡ʃ")   # loan-word digraph (e.g., chek)
        form = form.replace("sh", "ʃ")
        form = form.replace("kh", "χ")     # modern Hebrew: uvular fricative
        form = form.replace("zh", "ʒ")

        # Palatal approximant
        form = form.replace("y", "j")

        # Macron vowels mark length, unlike the acute stress marks above.
        form = form.replace("ā", "aː")
        form = form.replace("ī", "iː")
        form = form.replace("ū", "uː")
        form = form.replace("ē", "eː")
        form = form.replace("ō", "oː")


        # Apostrophe between two vowels (long or short) is a glottal stop;
        # elsewhere it's a syllable/schwa marker and gets dropped.
        form = re.sub(r"([aeiou]ː?)'([aeiou])", r"\1ʔ\2", form)
        form = form.replace("'", "")


        return cls.from_ipa(form)

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

        # Triliteral root template (C-C-C or CCC) → CaCaCa.
        form = _expand_triliteral_root(form)

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

        return cls.from_ipa(form)


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
        return cls.from_ipa(base.word)


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

        return cls.from_ipa(form)


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

        return cls.from_ipa(form)


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

        
        return cls.from_ipa(form)


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

        return cls.from_ipa(form)


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

        return cls.from_ipa(form)


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

        return cls.from_ipa(form.strip())


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
        return cls.from_ipa(form)


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
        return cls.from_ipa(form)


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

        return cls.from_ipa(form)


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

        return cls.from_ipa(form)


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
        return cls.from_ipa(form)


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
    def normalize(cls, text: str) -> str:
        """Strip Hebrew-script niqud — Aramaic uses the same diacritic block."""
        return _HEBREW_DIACRITICS.sub("", text).strip()

    @classmethod
    def from_romanization(cls, text: str) -> Self:
        """Aramaic scholarly romanization → IPA. Reuses Hebrew conventions
        (ʾ, ʿ, ś, š, ḏ, ṯ, ḫ, ḥ, p̄, ḇ, macron length, etc.)."""
        if not text:
            return cls(word=text)
        return cls(word=HebrewWord.from_romanization(text).to_ipa())

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

        return cls.from_ipa(form)


# ── Syriac (Syriac script) → IPA ────────────────────────────────
_SYRIAC_CONSONANTS = {
    'ܐ': 'ʔ', 'ܒ': 'b', 'ܓ': 'g', 'ܔ': 'g', 'ܕ': 'd', 'ܖ': 'd',
    'ܗ': 'h', 'ܘ': 'w', 'ܙ': 'z', 'ܚ': 'ħ', 'ܛ': 'tˤ', 'ܜ': 'tˤ',
    'ܝ': 'j', 'ܞ': 'jh', 'ܟ': 'k', 'ܠ': 'l', 'ܡ': 'm', 'ܢ': 'n',
    'ܣ': 's', 'ܤ': 's', 'ܥ': 'ʕ', 'ܦ': 'p', 'ܧ': 'p', 'ܨ': 'sˤ',
    'ܩ': 'q', 'ܪ': 'r', 'ܫ': 'ʃ', 'ܬ': 't', 'ܭ': 't', 'ܮ': 'd͡ʒ',
    'ܯ': 'tˤ',
}

# Syriac vowel marks (East + West traditions).  We collapse them to the
# nearest IPA quality; zqapha is rendered /a/ to match the Eastern tradition.
_SYRIAC_VOWELS = {
    'ܰ': 'a',   # pthaha above
    'ܱ': 'a',   # pthaha below
    'ܲ': 'a',   # pthaha dotted
    'ܳ': 'a',   # zqapha above
    'ܴ': 'a',   # zqapha below
    'ܵ': 'a',   # zqapha dotted
    'ܶ': 'e',   # rbasa above
    'ܷ': 'e',   # rbasa below
    'ܸ': 'e',   # dotted zlama horizontal
    'ܹ': 'i',   # dotted zlama angular
    'ܺ': 'i',   # hbasa above
    'ܻ': 'i',   # hbasa below
    'ܼ': 'u',   # hbasa-esasa dotted
    'ܽ': 'u',   # esasa above
    'ܾ': 'u',   # esasa below
    'ܿ': 'o',   # rwaha
}

# Marks to skip (qushshaya/rukkakha pronunciation dots, syame plural marker,
# combining alaph/dalath, accents, etc.).  They aren't phonemic at this layer.
_SYRIAC_SKIP = {
    '݀',  # feminine dot
    '݁',  # qushshaya
    '݂',  # rukkakha
    '݃',  # two vertical dots above
    '݄',  # two vertical dots below
    '݅',  # three dots above
    '݆',  # three dots below
    '݇',  # oblique line above
    '݈',  # oblique line below
    '݉',  # music
    '݊',  # barrekh
    'ܑ',  # superscript alaph
}


class SyriacWord(Word):
    @property
    def lang(self) -> str:
        return "syc"

    @classmethod
    def normalize(cls, text: str) -> str:
        """Strip Syriac vowel marks and diacritics."""
        return _SYRIAC_DIACRITICS.sub("", text).strip()

    @classmethod
    def from_romanization(cls, text: str) -> Self:
        """Syriac scholarly romanization → IPA. Reuses Hebrew conventions."""
        if not text:
            return cls(word=text)
        return cls(word=HebrewWord.from_romanization(text).to_ipa())

    @classmethod
    def from_syriac(cls, text) -> Self:
        """Syriac (Syriac script + vowel marks) → IPA."""
        if not text:
            return cls(word=text)

        result: list[str] = []
        for c in text:
            if c in _SYRIAC_CONSONANTS:
                result.append(_SYRIAC_CONSONANTS[c])
            elif c in _SYRIAC_VOWELS:
                result.append(_SYRIAC_VOWELS[c])
            elif c in _SYRIAC_SKIP:
                continue
            # else: ignore punctuation, spaces, etc.

        form = "".join(result)

        # Strip trailing ʔ (Syriac emphatic state, written with final alaph).
        form = form.rstrip('ʔ')

        return cls.from_ipa(form)


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
        ipa = ipa.replace("V", "a")
        ipa = _strip_combining(ipa).strip()
        if ipa and ipa[0] in VOWELS:
            ipa = "ʔ" + ipa
        return cls(word=ipa, lang=lang)

    @classmethod
    def from_romanization(cls, text: str, lang: str) -> "GenericWord":
        # Reuse the broad SemProWord transliteration cleanup so dotted / marked
        # scholarly Latin input gets mapped into the same IPA-ish internal
        # alphabet, but preserve the original language tag.
        base = SemProWord.from_romanization(text)
        return cls.from_ipa(base.word, lang=lang)

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
    # R-colored vowels — unfold the rhoticity to an explicit /r/ so it
    # survives the pansemitic reduction (transistor /tɹænˈzɪstɚ/ → tranzistar).
    ("ɚ", "ar"), ("ɝ", "ar"),
    ("ʌ", "a"),
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
        if ancestor.lang in ("sem-pro", "sem-wes-pro", "ar", "akk", "arc", "syc"):
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

        return cls.from_ipa(form)

def get_dialect(pronounciations: list[IpaRealization], ordered_tags: list[str]) -> str | None:
    """
    Get the preferred dialect pronounciation with fallbacks
    """
    for tag in ordered_tags:
        for pronounciation in pronounciations:
            if tag in pronounciation.tags or tag == "*":
                return pronounciation.ipa
    return None


# Per-language dialect preference, applied to IpaRealization.tags.  The "*"
# sentinel in get_dialect accepts any remaining realization as a last resort.
_DIALECT_PREFERENCES: dict[str, list[str]] = {
    "he": ["Biblical-Hebrew", "Modern-Israeli-Hebrew", "*"],
    "en": ["General-American", "Received-Pronunciation", "*"],
    # Egyptian period tags are synthesized from sounds[].note in kaikki.py
    # (see _NOTE_TAG_SYNONYMS); the substring "Late Egyptian" also matches
    # the "latest" / "Amarna-period" / "reconstructed" Late Egyptian notes.
    "egy": ["Late-Egyptian", "Middle-Egyptian", "Old-Egyptian", "*"],
}


def _pick_ipa(src: SharedSource) -> str | None:
    prefs = _DIALECT_PREFERENCES.get(src.lang, ["*"])
    return get_dialect(src.pronunciations, prefs)


def word_from_sharedsource(src: SharedSource) -> Word:
    """Build the language-appropriate Word for a shared etymology source.

    Prefers native IPA from kaikki when available; falls back to romanization
    and finally to script-specific decoders.  GenericWord catches everything
    that has no dedicated subclass but does carry IPA or romanization.
    """
    ipa = _pick_ipa(src)
    match src.lang:
        case "ar":
            if ipa:
                return ArabicWord.from_ipa(ipa)
            if src.romanization:
                return ArabicWord.from_romanization(src.romanization)
            raise MissingRomanizationError("arabic")
        case "he":
            if ipa:
                return HebrewWord.from_ipa(ipa)
            if src.romanization:
                return HebrewWord.from_romanization(src.romanization)
            raise MissingRomanizationError("hebrew")
        case "akk":
            if ipa:
                return AkkadianWord.from_ipa(ipa)
            if src.romanization:
                return AkkadianWord.from_romanization(src.romanization)
            raise MissingRomanizationError("akkadian")
        case "sux":
            if ipa:
                return SumerianWord.from_ipa(ipa)
            if src.romanization:
                return SumerianWord.from_romanization(src.romanization)
            raise MissingRomanizationError("sumerian")
        case "sem-pro" |  "qfa-hur-pro":
            if ipa:
                return SemProWord.from_ipa(ipa)
            return SemProWord.from_romanization(src.word)
        case "sem-wes-pro":
            if ipa:
                return SemWesProWord.from_ipa(ipa)
            return SemWesProWord.from_romanization(src.word)
        case "afa-pro":
            if ipa:
                return AfrasianWord.from_ipa(ipa)
            return AfrasianWord.from_romanization(src.word)
        case "grc":
            if ipa:
                return GreekWord.from_ipa(ipa)
            return GreekWord.from_greek(src.word)
        case "arc":
            if ipa:
                return AramaicWord.from_ipa(ipa)
            return AramaicWord.from_aramaic(src.word)
        case "syc":
            if ipa:
                return SyriacWord.from_ipa(ipa)
            return SyriacWord.from_syriac(src.word)
        case "egy":
            if ipa:
                return EgyptianWord.from_ipa(ipa)
            if src.romanization:
                return EgyptianWord.from_romanization(src.romanization)
            raise MissingRomanizationError("egyptian")
        case "ine-pro":
            if ipa:
                return PieWord.from_ipa(ipa)
            if src.romanization:
                return PieWord.from_romanization(src.romanization)
            raise MissingRomanizationError("proto-indo-european")
        case "itc-pro":
            if ipa:
                return ProtoItalicWord.from_ipa(ipa)
            return ProtoItalicWord.from_romanization(src.word)
        case "gem-pro":
            if ipa:
                return ProtoGermanicWord.from_ipa(ipa)
            return ProtoGermanicWord.from_romanization(src.word)
        case "ira-pro" | "xme-old":
            if ipa:
                return IranianWord.from_ipa(ipa)
            if src.romanization:
                return IranianWord.from_romanization(src.romanization)
            raise MissingRomanizationError("proto-iranian")
        case "dra-sou-pro":
            if ipa:
                return ProtoSouthDravidianWord.from_ipa(ipa)
            return ProtoSouthDravidianWord.from_romanization(src.word)
        case "peo" | "pal" | "fa-cls" | "fa":
            if ipa:
                return OldPersianWord.from_ipa(ipa)
            if src.romanization:
                return OldPersianWord.from_romanization(src.romanization)
            raise MissingRomanizationError("old-persian")
        case "ru" | "orv" | "sla-pro":
            return CyrillicWord.from_cyrillic(src.word)
        case _:
            if ipa:
                return GenericWord.from_ipa(ipa, lang=src.lang)
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

    # Trace-based aligned merge handles unequal consonant counts on its own,
    # so the old `countconsonants()` guard / ConsonantMismatchError are not
    # invoked on this path.  To switch back to the strict zip path, replace
    # this call with `merge_roots(ar, he)` and re-add the count check above.
    result, _unresolved = merge_roots_aligned(ar, he)
    if not result or not result.word:
        raise EmptyAncestorError()
    return result



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
                if prev == tok and tok not in VOWELS:
                    continue
                out.append(tok)
                prev = tok
                continue

        tok = c
        i += 1
        while i < n and ipa[i] in IPA_MODIFIERS:
            tok += ipa[i]
            i += 1
        if prev == tok and tok not in VOWELS:
            continue
        out.append(tok)
        prev = tok
    return "".join(out)


def extract_phonemes(word: Word) -> list[tuple[Consonant, Vowel | None]]:
    """Extract (consonant, vowel) pairs from a Word's IPA string.

    Each consonant phoneme is paired with its immediately-following vowel
    phoneme, or None if no vowel follows (consonant clusters).  A leading
    bare vowel becomes ("", vowel).
    """
    tokens = Phoneme.parse(word.word)
    result: list[tuple[Consonant, Vowel | None]] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if isinstance(tok, Vowel):
            result.append((Consonant(tok="ʔ"), tok))
            i += 1
        else:
            vowel: Vowel | None = None
            if i + 1 < len(tokens) and isinstance(tokens[i + 1], Vowel):
                nexttoken = tokens[i + 1]
                assert isinstance(nexttoken, Vowel)
                vowel = nexttoken
                i += 2
            else:
                i += 1
            assert isinstance(tok, Consonant)
            result.append((tok, vowel))
    return result


def reconcile_consonant(ar: Consonant, he: Consonant) -> Consonant:
    """Reconcile a consonant pair between Arabic and Hebrew.

    If both are the same, return it. Otherwise apply known correspondences
    to recover the proto-Semitic form.
    """
    ar_c = ar.tok
    he_c = he.tok
    if ar_c == he_c:
        return ar

    pair = (ar_c, he_c)
    match pair:
        case ("w", "j"):
            return Consonant(tok="w")
        case ("p", "p") | ("f", "p"):
            return Consonant(tok="p")
        case ("d͡ʒ", "g"):
            return Consonant(tok="g")
        case ("ʃ", "s"):
            return Consonant(tok="ɬ")
        case _:
            # Default: prefer Arabic (more conservative)
            return ar


def reconcile_vowel(ar_v: Vowel | None, he_v: Vowel | None) -> Vowel | None:
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
        result.append(reconcile_consonant(ar_c, he_c))
        v = reconcile_vowel(ar_v, he_v)
        if v is not None:
            result.append(v)

    return ReconstructedSemProWord(word="".join(r.tok for r in result))


@dataclass(frozen=True)
class UnresolvedSite:
    """A point in the alignment where two-language reconstruction is ambiguous.

    `kind` names the source of ambiguity: "prothesis" (an initial /ʔ/ we
    dropped as romanization-side prothesis), "vowel_indel" (a vowel
    present on only one side — kept, but flagged because vowel templates
    diverge between Arabic and Hebrew), "metathesis" (segment order
    couldn't be settled from two languages), "cross_type" (the alignment
    paired a vowel with a consonant — should never happen in practice).

    `chosen` is what was emitted into the ancestor (empty tuple = nothing).
    Caller can revisit these later with a third witness (Aramaic, Akkadian).
    """
    kind: str
    arabic: tuple[Phoneme, ...]
    hebrew: tuple[Phoneme, ...]
    chosen: tuple[Phoneme, ...]


def merge_roots_aligned(
    ar: ArabicWord, he: HebrewWord
) -> tuple[Word, list[UnresolvedSite]]:
    """Reconstruct an ancestor by aligning Arabic and Hebrew via `trace_cost`.

    The trace is a *correspondence map*, not a transformation script —
    applying every step would just yield Hebrew. This walker interprets
    each step under Semitic priors:

      - SUBSTITUTE: reconcile_consonant / reconcile_vowel.
      - DELETE (Arabic-only segment): keep it. Arabic is more conservative,
        so assume Hebrew lost it. Word-initial /ʔ/ is dropped as
        romanization-side prothesis (the from_romanization paths inject
        /ʔ/ before any initial vowel).
      - INSERT (Hebrew-only segment): keep it as a probable Arabic-side
        loss; word-initial /ʔ/ likewise dropped as prothesis. Vowel
        insertions are kept but flagged.
      - METATHESIS: emit Arabic order, flagged.

    Unlike `merge_roots`, this handles unequal consonant counts: prothesis,
    metathesis, and segment loss show up as DELETE/INSERT/METATHESIS in
    the trace and are interpreted, not zipped.
    """
    if not ar.word or not he.word:
        raise EmptyAncestorError()

    a_phs = Phoneme.parse(ar.word)
    b_phs = Phoneme.parse(he.word)
    trace = trace_cost(a_phs, b_phs)

    out: list[Phoneme] = []
    unresolved: list[UnresolvedSite] = []
    seen_segment = False

    for sr in trace:
        a_win = sr.a_phonemes
        b_win = sr.b_phonemes
        name = sr.rule.name

        if name == "substitute":
            a0, b0 = a_win[0], b_win[0]
            if isinstance(a0, Vowel) and isinstance(b0, Vowel):
                v = reconcile_vowel(a0, b0)
                if v is not None:
                    out.append(v)
                    seen_segment = True
            elif isinstance(a0, Consonant) and isinstance(b0, Consonant):
                out.append(reconcile_consonant(a0, b0))
                seen_segment = True
            else:
                out.append(a0)
                seen_segment = True
                unresolved.append(UnresolvedSite("cross_type", a_win, b_win, (a0,)))

        elif name == "delete":
            a0 = a_win[0]
            if not seen_segment and a0.tok == "ʔ":
                unresolved.append(UnresolvedSite("prothesis", a_win, (), ()))
                continue
            out.append(a0)
            seen_segment = True
            if isinstance(a0, Vowel):
                unresolved.append(UnresolvedSite("vowel_indel", a_win, (), a_win))

        elif name == "insert":
            b0 = b_win[0]
            if not seen_segment and b0.tok == "ʔ":
                unresolved.append(UnresolvedSite("prothesis", (), b_win, ()))
                continue
            out.append(b0)
            seen_segment = True
            if isinstance(b0, Vowel):
                unresolved.append(UnresolvedSite("vowel_indel", (), b_win, b_win))

        elif name == "metathesis":
            out.extend(a_win)
            seen_segment = True
            unresolved.append(UnresolvedSite("metathesis", a_win, b_win, a_win))

        else:
            raise ValueError(f"merge_roots_aligned: unhandled rule {name!r}")

    word = "".join(p.tok for p in out)
    if not word:
        raise EmptyAncestorError()
    return ReconstructedSemProWord(word=word), unresolved

