"""
Text chunking strategies for splitting documents into embeddable segments.
"""


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
            # Look for the last sentence-ending punctuation
            last_period = chunk.rfind(". ")
            last_newline = chunk.rfind("\n")
            break_point = max(last_period, last_newline)

            if break_point > chunk_size * 0.3:  # Only if we have reasonable content
                chunk = chunk[: break_point + 1]
                end = start + break_point + 1

        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap if end < len(text) else end

    return chunks


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

    # Don't forget the last chunk
    if current_chunk:
        if chunks and current_length < min_chunk_size:
            # Merge tiny trailing chunk with the previous one
            chunks[-1] += "\n\n" + "\n\n".join(current_chunk)
        else:
            chunks.append("\n\n".join(current_chunk))

    return chunks


# Default chunking strategy
DEFAULT_CHUNKER = chunk_by_fixed_size
