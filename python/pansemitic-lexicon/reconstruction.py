import re
import unicodedata
from typing import Self
from dataclasses import dataclass


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


# ── Languages whose headwords are already Latin-script ────────
LATIN_SCRIPT_LANGS = frozenset({
    "akk", "la", "ine-pro", "egy", "afa-pro", "sux",
    "en", "fr", "de", "it", "es", "pt", "nl", "cs",
    "gem-pro", "ira-pro", "peo", "fa-cls",
    "sem-pro", "sem-wes-pro",
})


_IPA_DELIMITED = re.compile(r"[/\[]([^/\]]+)[/\]]")


@dataclass
class SharedSource:
    """A shared borrowing/etymology source.

    Priority for what we hand to the Word builder:
      1. `ipa` — native kaikki IPA if available (skips romanization step)
      2. `romanization` — Latin/scholar transliteration
      3. `word` — the raw source headword (often in native script)
    """
    lang: str
    word: str
    romanization: str | None = None
    ipa: str | None = None

    @staticmethod
    def resolve_romanization(
        lang: str,
        word: str,
        template_tr: str | None = None,
        kaikki_roman_index: dict[tuple[str, str], str] | None = None,
    ) -> str | None:
        """Resolve romanization using the three-tier strategy."""
        def _is_latin(s: str) -> bool:
            return bool(s) and all(
                c.isascii() or "LATIN" in unicodedata.name(c, "")
                for c in s if c.isalpha()
            )
        # Tier 1: Latin-script language AND word is actually in Latin.
        # (Some kaikki entries have the word in the native script even for
        # nominally-Latin langs like sem-pro — don't return those.)
        if lang in LATIN_SCRIPT_LANGS and _is_latin(word):
            return word
        # Check if the word is already Latin script regardless of language
        if _is_latin(word):
            return word
        # Tier 2: tr arg from etymology template
        if template_tr:
            # Take the first alternative if comma-separated
            return template_tr.split(",")[0].strip()
        # Tier 3: kaikki entry lookup
        if kaikki_roman_index:
            roman = kaikki_roman_index.get((lang, word))
            if roman:
                return roman
        return None

    @staticmethod
    def resolve_ipa(
        lang: str,
        word: str,
        kaikki_ipa_index: dict[tuple[str, str], str] | None = None,
    ) -> str | None:
        """Look up native IPA from kaikki sounds data and normalize it.

        kaikki wraps IPA in /…/ or […], may concatenate variants with commas,
        and sprinkles prosodic marks (ˈ ˌ .) we don't want.
        """
        if not kaikki_ipa_index:
            return None
        raw = kaikki_ipa_index.get((lang, word))
        if not raw:
            return None
        m = _IPA_DELIMITED.search(raw)
        out = m.group(1) if m else raw
        out = out.split(",")[0].strip()
        for c in "ˈˌ.":
            out = out.replace(c, "")
        out = out.strip()
        return out or None

    def to_word(self) -> "Word":
        """Build the language-appropriate Word for this source.

        Prefers native IPA from kaikki (when present) over romanization.  Falls
        back to the lang-specific from_romanization methods and finally to a
        generic SemProWord normalization for unsupported langs that do have a
        romanization.
        """
        match self.lang:
            case "ar":
                if self.ipa:
                    return ArabicWord.from_ipa(self.ipa)
                if self.romanization:
                    return ArabicWord.from_romanization(self.romanization)
                raise MissingRomanizationError("arabic")
            case "he":
                if self.ipa:
                    return HebrewWord.from_ipa(self.ipa)
                if self.romanization:
                    return HebrewWord.from_romanization(self.romanization)
                raise MissingRomanizationError("hebrew")
            case "sem-pro" | "sem-wes-pro":
                if self.ipa:
                    return SemProWord.from_ipa(self.ipa)
                if self.romanization:
                    return SemProWord.from_romanization(self.romanization)
                raise MissingRomanizationError("sem-pro")
            case "grc":
                if self.ipa:
                    return GreekWord.from_ipa(self.ipa)
                return GreekWord.from_greek(self.word)
            case "arc":
                if self.ipa:
                    return AramaicWord.from_ipa(self.ipa)
                return AramaicWord.from_aramaic(self.word)
            case "egy":
                if self.ipa:
                    return EgyptianWord.from_ipa(self.ipa)
                if self.romanization:
                    return EgyptianWord.from_romanization(self.romanization)
                raise MissingRomanizationError("egyptian")
            case "ine-pro":
                if self.ipa:
                    return PieWord.from_ipa(self.ipa)
                if self.romanization:
                    return PieWord.from_romanization(self.romanization)
                raise MissingRomanizationError("proto-indo-european")
            case "ru" | "orv" | "sla-pro":
                return CyrillicWord.from_cyrillic(self.word)
            case _:
                if self.ipa:
                    return GenericIpaWord(word=_strip_combining(self.ipa), lang=self.lang)
                if self.romanization:
                    # We have a romanization — treat it as a generic Latin-script
                    # source and normalize minimally via SemProWord
                    return SemProWord.from_romanization(self.romanization)
                raise UnsupportedLanguageError(self.lang)

    def __str__(self) -> str:
        return f"{self.lang}:{self.word}"


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
IPA_MODIFIERS = set("ːˤʰʲʷˠ")
IPA_CONSONANT_DIGRAPHS = ("d͡ʒ", "t͡ʃ", "t͡s", "d͡z")


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
        # that makes d͡ʒ, t͡ʃ, t͡s single phonemes.
        if unicodedata.category(c) == "Mn" and c != "͡":
            continue
        if c == "ı":  # dotless i — doesn't decompose; map to plain i
            out.append("i")
        else:
            out.append(c)
    return "".join(out)


def _geminate(form: str) -> str:
    """Collapse runs of identical letters into letter + ː.

    Applied to the raw romanization *before* single-char substitutions so
    that doubled scholar-notation letters (e.g. ṣṣ) collapse cleanly and
    the ː survives downstream (ṣṣ → ṣː → sˤː).
    """
    return re.sub(r"([^\W\d_])\1+", r"\1ː", form)


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


# IPA → pansemitic.  Lossy: strips length + gemination, collapses vowels
# to {a, i, u}, compresses the consonant inventory.
_IPA_TO_PANSEMITIC: list[tuple[str, str]] = [
    # emphatics (multi-char first)
    ("ðˤ", "ṣ"),
    ("θˤ", "ṣ"),
    ("ɬˤ", "ṣ"),
    ("sˤ", "ṣ"),
    ("tˤ", "ṭ"),
    ("dˤ", "ṣ"),
    # IPA palatal approximant → Semitic y.  Must run *before* d͡ʒ → j so
    # that the j we produce for jīm isn't double-converted to y.
    ("j", "y"),
    # affricates
    ("d͡ʒ", "j"),
    ("t͡ʃ", "š"),
    ("t͡s", "ṣ"),
    ("d͡z", "z"),
    # voiceless dorsals merge → x
    ("χ", "x"),
    ("ħ", "x"),
    # voiced velar fricative → g; IPA single-story g variant → plain g
    ("ɣ", "g"),
    ("ɡ", "g"),
    # Non-a/i/u IPA vowels collapse toward the nearest e/o/a/i/u base.
    ("ɪ", "i"), ("ʏ", "i"), ("ɨ", "i"),
    ("ʊ", "u"), ("ɯ", "u"),
    ("ɛ", "e"), ("œ", "e"), ("ø", "e"),
    ("ɔ", "o"),
    ("ɐ", "a"), ("ɑ", "a"), ("ɒ", "a"), ("æ", "a"), ("ə", "a"),
    ("ɕ", "s"), ("ʑ", "z"),   # alveolo-palatal fricatives
    ("ɫ", "l"),               # velarized lateral
    # Rhotics collapse → r
    ("ɹ", "r"), ("ɾ", "r"), ("ʀ", "r"), ("ʁ", "r"), ("ɻ", "r"),
    # Nasals
    ("ŋ", "n"), ("ɲ", "n"), ("ɳ", "n"),
    # Retroflex stops → dental
    ("ʈ", "t"), ("ɖ", "d"),
    # Rhotic schwa / low vowel
    ("ɚ", "a"), ("ʌ", "a"),
    # Aspiration / palatalization aren't phonemic for Semitic — drop.
    ("ʰ", ""), ("ʱ", ""), ("ʲ", ""), ("ˠ", ""),
    # Superscript letters used as IPA release/off-glide marks — drop.
    ("ᵗ", ""), ("ⁱ", ""), ("ⁿ", ""),
    # Drop any stray tie bar that wasn't consumed by an affricate rule above.
    ("͡", ""),
    # interdentals → sibilants
    ("θ", "s"),
    ("ð", "z"),
    # postalveolars and lateral
    ("ʃ", "š"),
    ("ʒ", "ž"),
    ("ɬ", "s"),
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
        return cls(word=_strip_combining(ipa))

    def to_ipa(self) -> str:
        return self.word

    def to_protosemitic_convention(self) -> str:
        """IPA → proto-Semitic scholar notation.  Strips ː (length/gemination)."""
        out = self.word
        for src, dst in _IPA_TO_SCHOLAR:
            out = out.replace(src, dst)
        return out.replace("ː", "")

    def to_protopansemitic(self) -> str:
        """IPA → pansemitic.  Lossy: strips length/gemination, collapses
        vowels to a/i/u, compresses the consonant inventory."""
        if not self.word:
            return ""
        form = self.word.lstrip("*").rstrip("-").lower()

        # Semitic-family sources: f reflects older *p.
        if self.lang in ("sem-pro", "sem-wes-pro", "ar", "akk", "arc"):
            form = form.replace("f", "p")

        for src, dst in _IPA_TO_PANSEMITIC:
            form = form.replace(src, dst)

        # Collapse vowels to a/i/u — handle long forms first so eː → i, oː → a.
        form = form.replace("eː", "i").replace("oː", "a")
        form = form.replace("aː", "a").replace("iː", "i").replace("uː", "u")
        form = form.replace("e", "i").replace("o", "a")

        # Drop any remaining ː (consonant gemination) and dedupe vowels.
        form = form.replace("ː", "")
        form = re.sub(r"([aiu])\1+", r"\1", form)
        return form.strip()

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

        # Strip stress marks.
        form = (form
                .replace("á", "a").replace("é", "e").replace("í", "i")
                .replace("ó", "o").replace("ú", "u"))

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

        form = form.strip()

        # Hebrew words don't start with a bare vowel phonetically — prepend ʔ.
        if form and form[0] in VOWELS:
            form = "ʔ" + form

        return cls(word=form)


class SemProWord(Word):
    @property
    def lang(self) -> str:
        return "sem-pro"

    @classmethod
    def from_romanization(cls, text: str) -> Self:
        """proto-Semitic scholar notation → IPA."""
        if not text:
            return cls(word=text)
        form = text.lower()

        # Strip reconstruction markers.
        form = form.lstrip("*").rstrip("-")

        # Strip acute stress accents (kaikki uses them on non-Semitic loans).
        form = (form
                .replace("á", "a").replace("é", "e").replace("í", "i")
                .replace("ó", "o").replace("ú", "u"))

        # Glottal / pharyngeal variants first
        form = form.replace("ʾ", "ʔ")
        form = form.replace("ʿ", "ʕ")

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
        form = text.lower().lstrip("*").rstrip("-")
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


class GenericIpaWord(Word):
    """A Word wrapping a pre-existing IPA string with an arbitrary lang tag.

    Used when we have native IPA from kaikki for a language that has no
    dedicated subclass (fr, la, sa, fa, …).  `from_romanization` is not
    supported — construct directly with a cleaned IPA string.
    """

    def __init__(self, word: str, lang: str):
        self.word = word
        self._lang_tag = lang

    @property
    def lang(self) -> str:
        return self._lang_tag



def reconstruct_ancestor(
    ar_roman: str,
    he_roman: str,
    shared_sources: list[SharedSource] | None = None,
) -> Word:
    """Return the best ancestor form.

    Priority:
      1. Shared etymology source — LCA from the borrowing/inheritance graph
         (already sorted by the caller, first entry is the best)
      2. Reconstruction from Arabic/Hebrew romanizations
    """

    # 1. Shared etymology source — pick the first (best) one
    if shared_sources:
        result = shared_sources[0].to_word()
        if not result.word:
            raise EmptyAncestorError()
        return result

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

    return SemProWord(word="".join(result))
    