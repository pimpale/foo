import re
import unicodedata
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
    "en", "fr", "de", "it", "es", "pt", "nl", "cs", "ru",
    "gem-pro", "ira-pro", "peo", "fa-cls",
    "sem-pro", "sem-wes-pro",
})


@dataclass
class SharedSource:
    """A shared borrowing/etymology source with its romanization.

    Romanization is resolved in priority order:
      1. Word itself, if the source language uses Latin script
      2. The `tr` (transliteration) arg from the etymology template
      3. Romanization from the word's own kaikki entry
    """
    lang: str
    word: str
    romanization: str | None = None

    @staticmethod
    def resolve_romanization(
        lang: str,
        word: str,
        template_tr: str | None = None,
        kaikki_roman_index: dict[tuple[str, str], str] | None = None,
    ) -> str | None:
        """Resolve romanization using the three-tier strategy."""
        # Tier 1: Latin-script language — word is its own romanization
        if lang in LATIN_SCRIPT_LANGS:
            return word
        # Check if the word is already Latin script regardless of language
        if word and all(
            c.isascii() or "LATIN" in unicodedata.name(c, "")
            for c in word if c.isalpha()
        ):
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

    def __str__(self) -> str:
        return f"{self.lang}:{self.word}"


# ── Normalization to a common proto-Semitic encoding ───────────
#
# Target encoding (proto-Semitic canonical):
#   Glottal stop:       ʔ
#   Pharyngeal:         ʕ
#   Voiceless pharyngeal fric.: ḥ
#   Voiceless velar fric.:      ḫ  (Arabic ḵ, Hebrew kh)
#   Interdental vl.:    ṯ
#   Interdental vd.:    ḏ
#   Velar fric. vd.:    ɣ  (Arabic ɣ/ḡ)
#   Emphatic s:         ṣ  (also ṣ́ kept as-is — distinct phoneme)
#   Emphatic t:         ṭ
#   Emphatic interdental fric.:         ṯ̣
#   Emphatic lateral fric.:         ṣ́
#   Lateral fric.:      ś
#   Postalveolar:       š
#   Labial stop:        p  (Arabic f < *p)
#   Uvular stop:        q  (sem-pro ḳ normalized to q)
#   Vowels:             a i u (no length, no e/o)

VOWELS = set("aiu")


@dataclass
class Word:
    """
    This represents a word that has been normalized to a common proto-Semitic encoding,
    with language-specific extensions
    """
    word: str
    
    @property
    def lang(self) -> str:
        raise NotImplementedError("Subclasses must implement lang property")

    def countconsonants(self) -> int:
        return sum(1 for c in self.word if c not in VOWELS and c.isalpha())

    def consonants(self) -> str:
        return "".join(c for c in self.word if c not in VOWELS and c.isalpha())

    def __str__(self) -> str:
        return f"{self.lang}:{self.word}"


class ArabicWord(Word):
    @property
    def lang(self) -> str:
        return "ar"
    
    @classmethod
    def from_romanization(cls, text) -> 'ArabicWord':
        """Normalize Arabic romanization to proto-Semitic encoding."""
        if not text:
            return cls(word=text)
        form = text.lower()

        # Remove Arabic-specific prefixes
        form = re.sub(r"^a[lrnsṣṭḍ]-", "", form)

        # Multi-char first (order matters)
        # Arabic doesn't use digraphs, but handle "sh" just in case
        form = form.replace("sh", "š")

        # Consonant mappings: Arabic → proto-Semitic
        form = form.replace("ḵ", "ḫ")   # Arabic ḵ → PS *ḫ
        form = form.replace("ḡ", "ɣ")   # normalize ḡ variant to ɣ
        form = form.replace("ẓ", "ṯ̣")  # Arabic ẓ → PS *ṯ̣ (emphatic interdental fricative)
        form = form.replace("ḍ", "ṣ́") # Arabic ḍ → PS *ṣ́ (emphatic lateral fricative)
        form = form.replace("f", "p")   # Arabic f → PS *p (f is a later development from *p)
        
        # Glottal: Arabic uses ʔ consistently, which is already canonical

        # Strip vowel length (macrons)
        form = form.replace("ā", "a").replace("ī", "i").replace("ū", "u")

        # Collapse gemination (bb→b, ss→s, etc.)
        form = re.sub(r'(.)\1+', r'\1', form)

        form = form.strip()

        return cls(word=form)


class HebrewWord(Word):
    @property
    def lang(self) -> str:
        return "he"

    @classmethod
    def from_romanization(cls, text) -> 'HebrewWord':
        """Normalize Hebrew romanization to proto-Semitic encoding."""
        if not text:
            return cls(word=text)
        form = text.lower()

        # Strip vowel length (acute accents)
        form = form.replace("á", "a").replace("é", "e").replace("í", "i")
        form = form.replace("ó", "o").replace("ú", "u")

        # Multi-char digraphs → single proto-Semitic chars (longest first)
        form = form.replace("tsh", "ṣ")   # edge case: tsh before ts/sh
        form = form.replace("ts", "ṣ")    # Hebrew tsade → PS *ṣ
        form = form.replace("sh", "š")    # Hebrew shin → PS *š
        form = form.replace("kh", "ḫ")    # Hebrew khaf → PS *ḫ (merged ḫ/ḥ)
        form = form.replace("zh", "ž")    # Hebrew zayin+geresh (loanwords)

        # Remove schwa markers: b'rakhá → brakhá (already lowered)
        form = form.replace("'", "")

        # Note: Hebrew merged *ḫ and *ḥ into kh — we map to ḫ here.
        # Without Arabic to disambiguate, we can't recover which it was.
        # The caller can cross-reference with Arabic if needed.

        # Vowel collapse: Hebrew e→i, o→u (Canaanite shift reversal)
        form = form.replace("e", "i").replace("o", "a")

        # Initial vowel → glottal stop + vowel (PS words don't start with vowels)
        form = form.strip()
        if form and form[0] in VOWELS:
            form = "ʔ" + form

        return cls(word=form)


class SemProWord(Word):
    @property
    def lang(self) -> str:
        return "sem-pro"

    @classmethod
    def from_romanization(cls, text) -> 'SemProWord':
        """Normalize sem-pro romanization to a consistent proto-Semitic encoding."""
        if not text:
            return cls(word=text)
        form = text.lower()

        # Strip reconstruction markers
        form = form.lstrip("*").rstrip("-")

        # Glottal stop variants: ʾ and ʿ(misused) → ʔ
        form = form.replace("ʾ", "ʔ")

        # Fix ʿ only when used for glottal stop (in *ʿiśrū- etc.)
        # We can't blindly replace ʿ since it's legitimately ayin.
        # But ʿ as a glottal stop only appears word-initially before i;
        # true ayin ʕ is the canonical target for pharyngeal.
        # Handle the known pattern: word-initial ʿ before a vowel
        # when it should be ʔ — this is rare, leave ʿ as ʕ mapping below.

        # Pharyngeal: ʕ is already canonical; also map ʿ → ʕ
        form = form.replace("ʿ", "ʕ")

        # Velar fricative variants: normalize
        form = form.replace("ḡ", "ɣ")    # ḡ → ɣ

        # ḫ is already canonical for voiceless velar fricative

        # Uvular stop: ḳ → q
        form = form.replace("ḳ", "q")

        # Emphatic theta variants: θ̣ and ṯ̣ both appear
        # Normalize θ to ṯ first, so θ̣ becomes ṯ̣
        form = form.replace("θ", "ṯ")

        # Vowel normalization
        form = form.replace("ā", "a").replace("ī", "i").replace("ū", "u")
        form = form.replace("ô", "a")
        form = form.replace("e", "i").replace("o", "a")

        # kʷ → k (labiovelar — no separate representation needed downstream)
        form = form.replace("kʷ", "k")

        return cls(word=form.strip())


# ── Greek script → proto-Semitic-compatible romanization ─────────
# Maps Greek as received by Semitic speakers (loanwords):
#   φ→p (not f), θ→t, χ→ḫ, aspirates lost, vowels collapsed to a/i/u
_GREEK_MAP = {
    'α': 'a', 'β': 'b', 'γ': 'g', 'δ': 'd', 'ε': 'i', 'ζ': 'z',
    'η': 'i', 'θ': 't', 'ι': 'i', 'κ': 'k', 'λ': 'l', 'μ': 'm',
    'ν': 'n', 'ξ': 'ks', 'ο': 'a', 'π': 'p', 'ρ': 'r', 'σ': 's',
    'ς': 's', 'τ': 't', 'υ': 'u', 'φ': 'p', 'χ': 'ḫ', 'ψ': 'ps',
    'ω': 'a',
}


class GreekWord(Word):
    @property
    def lang(self) -> str:
        return "grc"

    @classmethod
    def from_romanization(cls, text) -> 'GreekWord':
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

        # Collapse gemination
        form = re.sub(r'(.)\1+', r'\1', form)

        form = form.strip()
        return cls(word=form)


# ── Aramaic (Hebrew script) → proto-Semitic encoding ────────────
_ARAMAIC_CONSONANTS = {
    'א': 'ʔ', 'ב': 'b', 'ג': 'g', 'ד': 'd', 'ה': 'h', 'ו': 'w',
    'ז': 'z', 'ח': 'ḥ', 'ט': 'ṭ', 'י': 'y', 'כ': 'k', 'ך': 'k',
    'ל': 'l', 'מ': 'm', 'ם': 'm', 'נ': 'n', 'ן': 'n', 'ס': 's',
    'ע': 'ʕ', 'פ': 'p', 'ף': 'p', 'צ': 'ṣ', 'ץ': 'ṣ', 'ק': 'q',
    'ר': 'r', 'ש': 'š', 'ת': 't',
}

# Nikkud (vowel points) → proto-Semitic vowels (e→i, o→a already applied)
_ARAMAIC_VOWELS = {
    '\u05B7': 'a',   # patach
    '\u05B8': 'a',   # qamats
    '\u05B6': 'i',   # segol (e→i)
    '\u05B5': 'i',   # tsere (e→i)
    '\u05B4': 'i',   # hiriq
    '\u05B9': 'a',   # holam (o→a)
    '\u05BB': 'u',   # qubuts
    '\u05B2': 'a',   # hataf patach
    '\u05B1': 'i',   # hataf segol (e→i)
    '\u05B3': 'a',   # hataf qamats
}

# Nikkud and marks to skip (shva, dagesh, rafe, etc.)
_ARAMAIC_SKIP = {
    '\u05B0',  # shva
    '\u05BC',  # dagesh
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
    def from_romanization(cls, text) -> 'AramaicWord':
        """Normalize Aramaic (Hebrew script with nikkud) to proto-Semitic encoding.

        Extracts consonants from the base letters and vowels from nikkud,
        mapping both to the proto-Semitic target encoding.
        """
        if not text:
            return cls(word=text)

        result = []
        for c in text:
            if c in _ARAMAIC_CONSONANTS:
                result.append(_ARAMAIC_CONSONANTS[c])
            elif c in _ARAMAIC_VOWELS:
                result.append(_ARAMAIC_VOWELS[c])
            elif c in _ARAMAIC_SKIP:
                continue
            # else: ignore (maqaf, sof pasuq, etc.)

        form = "".join(result)

        # Collapse gemination
        form = re.sub(r'(.)\1+', r'\1', form)

        # Strip trailing ʔ (Aramaic emphatic state -א)
        form = form.rstrip('ʔ')

        form = form.strip()
        return cls(word=form)


def _word_from_source(source: SharedSource) -> Word:
    """Normalize a SharedSource into a Word using its romanization.

    Uses the romanization if available, otherwise the raw word.
    Raises UnsupportedLanguageError if the language has no normalizer
    and no romanization is available.
    """
    text = source.romanization or source.word
    match source.lang:
        case "ar":
            return ArabicWord.from_romanization(text)
        case "he":
            return HebrewWord.from_romanization(text)
        case "sem-pro" | "sem-wes-pro":
            return SemProWord.from_romanization(text)
        case "grc":
            return GreekWord.from_romanization(text)
        case "arc":
            return AramaicWord.from_romanization(text)
        case _:
            if source.romanization:
                # We have a romanization — treat it as a generic Latin-script
                # source and normalize minimally via SemProWord
                return SemProWord.from_romanization(source.romanization)
            raise UnsupportedLanguageError(source.lang)


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
        result = _word_from_source(shared_sources[0])
        if not result or not result.word:
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


def extract_phonemes(word: Word) -> list[tuple[str, str | None]]:
    """Extract phonemes as (consonant, vowel) pairs from a normalized word.

    Each consonant is paired with its following vowel, or None if there's
    no vowel (e.g. consonant clusters like "lb" in "kalb" → [(k,a), (l,None), (b,None)]).
    """
    result = []
    chars = list(word.word)
    i = 0
    while i < len(chars):
        c = chars[i]
        if c not in VOWELS:
            # It's a consonant — check if next char is a vowel
            vowel = None
            if i + 1 < len(chars) and chars[i + 1] in VOWELS:
                vowel = chars[i + 1]
                i += 2
            else:
                i += 1
            result.append((c, vowel))
        else:
            # Leading vowel with no consonant — shouldn't happen in normalized
            # forms, but handle gracefully
            result.append(("", c))
            i += 1
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
        case ("w", "y"):
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
    