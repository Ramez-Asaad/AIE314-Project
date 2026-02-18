"""
Text cleaning and normalization utilities.
Prepares extracted text for chunking and downstream embedding.
"""

import re
import unicodedata

from .normalizers import normalize_text


def normalize_unicode(text: str) -> str:
    """Normalize Unicode characters to NFKC form (handles ligatures, superscripts, etc)."""
    return unicodedata.normalize("NFKC", text)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters into single spaces, preserving newlines."""
    # Replace tabs and other whitespace (except newlines) with spaces
    text = re.sub(r"[^\S\n]+", " ", text)
    # Collapse multiple consecutive newlines into at most two
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_control_characters(text: str) -> str:
    """Remove non-printable control characters (except newlines and tabs)."""
    return "".join(
        char
        for char in text
        if unicodedata.category(char) != "Cc" or char in ("\n", "\t")
    )


def remove_excessive_punctuation(text: str) -> str:
    """Reduce repeated punctuation (e.g., '!!!' -> '!')."""
    text = re.sub(r"([!?.]){2,}", r"\1", text)
    text = re.sub(r"[-=_]{3,}", "---", text)
    return text


def fix_encoding_artifacts(text: str) -> str:
    """Fix common encoding issues like mojibake patterns."""
    replacements = {
        "\u00e2\u0080\u0099": "'",
        "\u00e2\u0080\u009c": '"',
        "\u00e2\u0080\x9d": '"',
        "\u00e2\u0080\u0094": "\u2014",  # em dash
        "\u00e2\u0080\u0093": "\u2013",  # en dash
        "\u00c3\u00a9": "\u00e9",        # é
        "\u00c3\u00a8": "\u00e8",        # è
        "\u00c3\u00bc": "\u00fc",        # ü
        "\ufeff": "",  # BOM
        "\x00": "",    # Null bytes
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def clean_text(text: str) -> str:
    """
    Apply the full cleaning + normalization pipeline to a piece of text.

    Steps:
        1. Fix encoding artifacts
        2. Remove control characters
        3. Normalize Unicode (NFKC)
        4. Normalize whitespace
        5. Remove excessive punctuation
        6. Apply text normalization (typography, OCR repair, LaTeX, etc.)
    """
    text = fix_encoding_artifacts(text)
    text = remove_control_characters(text)
    text = normalize_unicode(text)
    text = normalize_whitespace(text)
    text = remove_excessive_punctuation(text)
    text = normalize_text(text)
    return text
