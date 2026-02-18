"""
Text cleaning and normalization utilities.
Prepares extracted text for chunking and downstream embedding.
"""

import re
import unicodedata


def normalize_unicode(text: str) -> str:
    """Normalize Unicode characters to NFC form."""
    return unicodedata.normalize("NFC", text)


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
        "â€™": "'",
        "â€œ": '"',
        "â€\x9d": '"',
        "â€"": "—",
        "â€"": "–",
        "Ã©": "é",
        "Ã¨": "è",
        "Ã¼": "ü",
        "\ufeff": "",  # BOM
        "\x00": "",    # Null bytes
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def clean_text(text: str) -> str:
    """
    Apply the full cleaning pipeline to a piece of text.

    Steps:
        1. Fix encoding artifacts
        2. Remove control characters
        3. Normalize Unicode
        4. Normalize whitespace
        5. Remove excessive punctuation
    """
    text = fix_encoding_artifacts(text)
    text = remove_control_characters(text)
    text = normalize_unicode(text)
    text = normalize_whitespace(text)
    text = remove_excessive_punctuation(text)
    return text
