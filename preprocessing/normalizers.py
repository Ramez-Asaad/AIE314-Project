"""
Text normalization utilities for the RAG preprocessing pipeline.

Normalizations applied (in order):
    1. Unicode NFKC — ligatures, superscripts, compatibility characters
    2. Typographic normalization — smart quotes, fancy dashes, bullet symbols
    3. OCR artifact repair — spaced-out words from bad PDF extraction
    4. Header/footer noise removal — isolated page numbers, repeated short lines
    5. LaTeX cleanup — common LaTeX math/symbols → readable text
    6. URL/email tagging — wrap detected URLs and emails in markers
"""

import re
import unicodedata
from functools import lru_cache


# ─── 1. Unicode NFKC Normalization ──────────────────────────────────────────

def normalize_unicode_nfkc(text: str) -> str:
    """
    Apply NFKC normalization to decompose compatibility characters.

    Handles:
        - Ligatures: ﬁ → fi, ﬂ → fl, ﬀ → ff
        - Superscripts/subscripts: ² → 2, ³ → 3, ₂ → 2
        - Fullwidth chars: Ａ → A, ０ → 0
        - Symbols: ™ → TM, © → (C)
    """
    return unicodedata.normalize("NFKC", text)


# ─── 2. Typographic Normalization ────────────────────────────────────────────

# Smart / curly quotes → ASCII straight quotes
_QUOTE_MAP = {
    "\u2018": "'",   # '
    "\u2019": "'",   # '
    "\u201A": "'",   # ‚
    "\u201B": "'",   # ‛
    "\u201C": '"',   # "
    "\u201D": '"',   # "
    "\u201E": '"',   # „
    "\u201F": '"',   # ‟
    "\u2039": "'",   # ‹
    "\u203A": "'",   # ›
    "\u00AB": '"',   # «
    "\u00BB": '"',   # »
}

# Dash variants → standard forms
_DASH_MAP = {
    "\u2012": "-",   # figure dash
    "\u2013": "-",   # en dash
    "\u2014": " - ", # em dash → spaced hyphen (preserves readability)
    "\u2015": " - ", # horizontal bar
    "\u2212": "-",   # minus sign
    "\uFE58": "-",   # small em dash
    "\uFE63": "-",   # small hyphen-minus
    "\uFF0D": "-",   # fullwidth hyphen-minus
}

# Bullet symbols → standard dash
_BULLET_MAP = {
    "\u2022": "-",   # •
    "\u2023": "-",   # ‣
    "\u25E6": "-",   # ◦
    "\u2043": "-",   # ⁃
    "\u25AA": "-",   # ▪
    "\u25CF": "-",   # ●
    "\u25CB": "-",   # ○
    "\u25A0": "-",   # ■
    "\u25A1": "-",   # □
    "\u25B6": "-",   # ▶
    "\u25B8": "-",   # ▸
    "\u27A2": "-",   # ➢
}

# Other typographic symbols
_MISC_MAP = {
    "\u2026": "...",  # … ellipsis
    "\u00A0": " ",    # non-breaking space
    "\u200B": "",     # zero-width space
    "\u200C": "",     # zero-width non-joiner
    "\u200D": "",     # zero-width joiner
    "\uFEFF": "",     # BOM / zero-width no-break space
}


def normalize_typography(text: str) -> str:
    """
    Normalize typographic characters to their ASCII equivalents.

    Handles smart quotes, fancy dashes, bullet symbols, ellipsis,
    and invisible Unicode characters.
    """
    for mapping in (_QUOTE_MAP, _DASH_MAP, _BULLET_MAP, _MISC_MAP):
        for old, new in mapping.items():
            text = text.replace(old, new)
    return text


# ─── 3. OCR Artifact Repair ─────────────────────────────────────────────────

def repair_ocr_artifacts(text: str) -> str:
    """
    Fix common OCR and PDF extraction artifacts.

    Handles:
        - Spaced-out words: "m a c h i n e" → "machine"
        - Broken hyphenation: "ma-\\nchine" → "machine"
        - Double spaces within sentences
    """
    # Fix spaced-out words (e.g., "m a c h i n e   l e a r n i n g")
    # Pattern: single characters separated by single spaces, at least 4 chars
    def _fix_spaced_word(match: re.Match) -> str:
        return match.group(0).replace(" ", "")

    text = re.sub(
        r'\b([a-zA-Z] ){3,}[a-zA-Z]\b',
        _fix_spaced_word,
        text,
    )

    # Fix broken hyphenation across lines: "ma-\nchine" → "machine"
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)

    # Collapse double spaces (but not leading indentation)
    text = re.sub(r'(?<=\S)  +(?=\S)', ' ', text)

    return text


# ─── 3b. Broken Word Repair (dictionary-based) ─────────────────────────────

@lru_cache(maxsize=1)
def _load_word_set() -> set[str]:
    """
    Load the NLTK English word corpus as a lowercase set.

    Downloads the corpus automatically on first use if not already present.
    Falls back to an empty set if NLTK is unavailable.
    """
    try:
        import nltk
        try:
            from nltk.corpus import words
            word_list = words.words()
        except LookupError:
            nltk.download("words", quiet=True)
            from nltk.corpus import words
            word_list = words.words()
        return {w.lower() for w in word_list}
    except ImportError:
        return set()


def _is_word(token: str) -> bool:
    """Check if a token is a recognized English word."""
    return token.lower() in _load_word_set()


def repair_broken_words(text: str) -> str:
    """
    Heal mid-word spaces using dictionary lookup.

    PDF extractors sometimes insert a space mid-word due to glyph
    positioning (e.g., justified text). This function detects adjacent
    fragments that individually are not real words but form a valid word
    when merged.

    Examples:
        "Struct ures" → "Structures"
        "acce ss" → "access"
        "experimentati on" → "experimentation"
    """
    word_set = _load_word_set()
    if not word_set:
        return text  # No dictionary available, skip

    # Pattern: two alpha fragments separated by a single space,
    # where at least one fragment is 2+ chars (to avoid merging
    # real single-letter words like "a" or "I")
    pattern = re.compile(r'\b([A-Za-z]{2,})\s([A-Za-z]{2,})\b')

    def _try_merge(match: re.Match) -> str:
        left, right = match.group(1), match.group(2)
        merged = left + right

        # Only merge if:
        # - Neither fragment is a recognized word on its own
        # - The merged result IS a recognized word
        left_is_word = left.lower() in word_set
        right_is_word = right.lower() in word_set

        if not left_is_word and not right_is_word and merged.lower() in word_set:
            return merged

        return match.group(0)  # Keep original

    # Run multiple passes since merging can reveal new merge opportunities
    for _ in range(3):
        new_text = pattern.sub(_try_merge, text)
        if new_text == text:
            break
        text = new_text

    return text


def repair_spaced_hyphens(text: str) -> str:
    """
    Fix spaces around hyphens in compound words.

    PDF extraction sometimes inserts spaces around hyphens:
        "step -by-step" → "step-by-step"
        "high -level" → "high-level"
        "re -use" → "re-use"
    """
    # Space before hyphen: "step -by" → "step-by"
    text = re.sub(r'(\w)\s+-(\w)', r'\1-\2', text)
    # Space after hyphen: "step- by" → "step-by"
    text = re.sub(r'(\w)-\s+(\w)', r'\1-\2', text)
    return text


# ─── 4. Header/Footer Noise Removal ─────────────────────────────────────────

def remove_header_footer_noise(text: str) -> str:
    """
    Remove common header/footer noise from extracted document text.

    Handles:
        - Standalone page numbers (e.g., "42", "Page 5", "- 12 -")
        - Repeated short lines that look like headers/footers
    """
    lines = text.split("\n")
    cleaned_lines: list[str] = []

    for line in lines:
        stripped = line.strip()

        # Skip standalone page numbers: "42", "Page 5 of 100", "- 12 -", "5 | "
        if re.match(r'^[-–—\s]*\d{1,4}[-–—\s]*$', stripped):
            continue
        if re.match(r'^[Pp]age\s+\d+(\s+of\s+\d+)?$', stripped):
            continue
        if re.match(r'^\d+\s*\|?\s*$', stripped):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


# ─── 5. LaTeX Cleanup ───────────────────────────────────────────────────────

# Common LaTeX commands → readable text
_LATEX_SYMBOLS = {
    r"\alpha": "alpha",
    r"\beta": "beta",
    r"\gamma": "gamma",
    r"\delta": "delta",
    r"\epsilon": "epsilon",
    r"\theta": "theta",
    r"\lambda": "lambda",
    r"\mu": "mu",
    r"\sigma": "sigma",
    r"\pi": "pi",
    r"\omega": "omega",
    r"\infty": "infinity",
    r"\partial": "partial",
    r"\nabla": "nabla",
    r"\sum": "sum",
    r"\prod": "product",
    r"\int": "integral",
    r"\sqrt": "sqrt",
    r"\approx": "approximately",
    r"\neq": "!=",
    r"\leq": "<=",
    r"\geq": ">=",
    r"\rightarrow": "->",
    r"\leftarrow": "<-",
    r"\Rightarrow": "=>",
    r"\in": "in",
    r"\notin": "not in",
    r"\subset": "subset of",
    r"\times": "x",
    r"\cdot": "*",
    r"\ldots": "...",
    r"\dots": "...",
}


def normalize_latex(text: str) -> str:
    """
    Convert common LaTeX notation to readable plaintext.

    Handles:
        - Greek letters: \\alpha → alpha
        - Math operators: \\leq → <=, \\approx → approximately
        - Fractions: \\frac{a}{b} → (a/b)
        - Superscripts/subscripts: x^{2} → x^2, x_{i} → x_i
        - Inline math delimiters: $...$ → stripped
        - Common commands: \\textbf{...} → the text inside
    """
    # Replace known LaTeX symbol commands
    for latex_cmd, replacement in _LATEX_SYMBOLS.items():
        text = text.replace(latex_cmd, replacement)

    # \frac{a}{b} → (a/b)
    text = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'(\1/\2)', text)

    # \textbf{...}, \textit{...}, \text{...}, \emph{...} → just the content
    text = re.sub(r'\\(?:textbf|textit|text|emph|mathrm|mathbf)\{([^}]*)\}', r'\1', text)

    # x^{2} → x^2  and  x_{i} → x_i  (remove braces around single-level)
    text = re.sub(r'\^\{([^}]*)\}', r'^\1', text)
    text = re.sub(r'_\{([^}]*)\}', r'_\1', text)

    # Strip inline math delimiters: $...$ (but not $$...$$)
    text = re.sub(r'(?<!\$)\$(?!\$)([^$]+)\$(?!\$)', r'\1', text)

    # Remove remaining common LaTeX commands: \command → empty
    text = re.sub(r'\\[a-zA-Z]+\s*', ' ', text)

    # Remove stray braces that were part of LaTeX
    text = re.sub(r'(?<!\w)[{}](?!\w)', '', text)

    return text


# ─── 6. URL/Email Tagging ───────────────────────────────────────────────────

_URL_PATTERN = re.compile(
    r'https?://[^\s<>\"\')\]]+',
    re.IGNORECASE,
)

_EMAIL_PATTERN = re.compile(
    r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
)


def tag_urls_and_emails(text: str) -> str:
    """
    Detect and tag URLs and emails with markers for downstream processing.

    Converts:
        - URLs → [URL: https://...]
        - Emails → [EMAIL: user@example.com]
    """
    text = _URL_PATTERN.sub(r'[URL: \g<0>]', text)
    text = _EMAIL_PATTERN.sub(r'[EMAIL: \g<0>]', text)
    return text


# ─── Full Normalization Pipeline ─────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """
    Apply the full normalization pipeline to a piece of text.

    Steps (in order):
        1. Unicode NFKC normalization
        2. Typographic normalization (quotes, dashes, bullets)
        3a. OCR artifact repair (spaced-out words, broken hyphens)
        3b. Broken word repair (dictionary-based merging)
        3c. Spaced hyphen repair (compound word fixes)
        4. Header/footer noise removal (page numbers)
        5. LaTeX cleanup (math symbols -> readable text)
        6. URL/email tagging
    """
    text = normalize_unicode_nfkc(text)
    text = normalize_typography(text)
    text = repair_ocr_artifacts(text)
    text = repair_broken_words(text)
    text = repair_spaced_hyphens(text)
    text = remove_header_footer_noise(text)
    text = normalize_latex(text)
    text = tag_urls_and_emails(text)
    return text
