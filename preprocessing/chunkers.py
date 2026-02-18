"""
Text chunking strategies for splitting documents into embeddable segments.
Includes fixed-size, paragraph-based, and semantic chunking.
"""

import re
import numpy as np
from sentence_transformers import SentenceTransformer


# ─── Lazy-loaded shared embedding model ──────────────────────────────────────
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Load the sentence-transformer model once and cache it."""
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


# ─── Utility ─────────────────────────────────────────────────────────────────

def _split_into_sentences(text: str) -> list[str]:
    """
    Naïve sentence splitter that handles common abbreviations.
    Splits on '.', '!', '?' followed by whitespace or end-of-string.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


# ─── Fixed-size chunking ─────────────────────────────────────────────────────

def chunk_by_fixed_size(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """
    Split text into fixed-size character chunks with optional overlap.

    Args:
        text: The text to chunk.
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        A list of text chunks.
    """
    if not text.strip():
        return []

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Try to break at a sentence or word boundary
        if end < len(text):
            last_period = chunk.rfind(". ")
            last_newline = chunk.rfind("\n")
            break_point = max(last_period, last_newline)

            if break_point > chunk_size * 0.3:
                chunk = chunk[: break_point + 1]
                end = start + break_point + 1

        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap if end < len(text) else end

    return chunks


# ─── Paragraph chunking ──────────────────────────────────────────────────────

def chunk_by_paragraph(text: str, min_chunk_size: int = 100) -> list[str]:
    """
    Split text into chunks by paragraph boundaries.
    Short paragraphs are merged with the next to avoid tiny chunks.

    Args:
        text: The text to chunk.
        min_chunk_size: Minimum characters for a chunk; shorter ones get merged.

    Returns:
        A list of text chunks.
    """
    if not text.strip():
        return []

    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_length = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        current_chunk.append(para)
        current_length += len(para)

        if current_length >= min_chunk_size:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = []
            current_length = 0

    if current_chunk:
        if chunks and current_length < min_chunk_size:
            chunks[-1] += "\n\n" + "\n\n".join(current_chunk)
        else:
            chunks.append("\n\n".join(current_chunk))

    return chunks


# ─── Semantic chunking ───────────────────────────────────────────────────────

def chunk_by_semantic(
    text: str,
    breakpoint_percentile: int = 80,
    min_chunk_size: int = 100,
    max_chunk_size: int = 1500,
) -> list[str]:
    """
    Split text into semantically coherent chunks using sentence embeddings.

    How it works:
        1. Split text into sentences.
        2. Embed each sentence with a sentence-transformer model.
        3. Compute cosine similarity between consecutive sentence embeddings.
        4. Identify breakpoints where similarity drops below a percentile
           threshold — these are topic shifts.
        5. Group sentences between breakpoints into chunks.
        6. Merge tiny chunks and split oversized ones as a post-processing step.

    Args:
        text: The text to chunk.
        breakpoint_percentile: Percentile threshold (0–100) for detecting
            topic shifts. Lower = more chunks, higher = fewer chunks.
        min_chunk_size: Minimum characters per chunk; smaller chunks get merged.
        max_chunk_size: Maximum characters per chunk; larger chunks get split.

    Returns:
        A list of semantically coherent text chunks.
    """
    if not text.strip():
        return []

    sentences = _split_into_sentences(text)

    # If very few sentences, return as a single chunk
    if len(sentences) <= 2:
        combined = " ".join(sentences)
        return [combined] if combined.strip() else []

    # 1. Embed all sentences
    model = _get_model()
    embeddings = model.encode(sentences, show_progress_bar=False)

    # 2. Compute cosine similarities between consecutive sentences
    similarities: list[float] = []
    for i in range(len(embeddings) - 1):
        a = embeddings[i]
        b = embeddings[i + 1]
        cosine_sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        similarities.append(cosine_sim)

    # 3. Determine the breakpoint threshold
    threshold = float(np.percentile(similarities, 100 - breakpoint_percentile))

    # 4. Find breakpoints (where similarity is BELOW the threshold = topic shift)
    breakpoints: list[int] = []
    for i, sim in enumerate(similarities):
        if sim < threshold:
            breakpoints.append(i + 1)  # break AFTER sentence i

    # 5. Group sentences into chunks between breakpoints
    raw_chunks: list[str] = []
    start = 0
    for bp in breakpoints:
        chunk_text = " ".join(sentences[start:bp]).strip()
        if chunk_text:
            raw_chunks.append(chunk_text)
        start = bp

    # Don't forget the last group
    last_chunk = " ".join(sentences[start:]).strip()
    if last_chunk:
        raw_chunks.append(last_chunk)

    # 6. Post-process: merge tiny chunks, split oversized ones
    chunks = _postprocess_chunks(raw_chunks, min_chunk_size, max_chunk_size)

    return chunks


def _postprocess_chunks(
    chunks: list[str],
    min_size: int,
    max_size: int,
) -> list[str]:
    """Merge small chunks and split oversized ones."""
    # Merge small chunks with their neighbor
    merged: list[str] = []
    buffer = ""

    for chunk in chunks:
        if buffer:
            combined = buffer + " " + chunk
            if len(combined) <= max_size:
                buffer = combined
            else:
                merged.append(buffer)
                buffer = chunk
        else:
            buffer = chunk

        if len(buffer) >= min_size:
            merged.append(buffer)
            buffer = ""

    if buffer:
        if merged and len(buffer) < min_size:
            merged[-1] += " " + buffer
        else:
            merged.append(buffer)

    # Split any chunks that are still too large
    final: list[str] = []
    for chunk in merged:
        if len(chunk) <= max_size:
            final.append(chunk)
        else:
            # Fall back to fixed-size splitting for oversized chunks
            sub_chunks = chunk_by_fixed_size(chunk, chunk_size=max_size, overlap=50)
            final.extend(sub_chunks)

    return final


# ─── Default strategy ────────────────────────────────────────────────────────

DEFAULT_CHUNKER = chunk_by_semantic
