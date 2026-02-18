"""
Document text extractors for various file formats.
Each extractor returns a list of (page/section, text) tuples along with metadata.
"""

import os
from datetime import datetime, timezone
from typing import Any

import PyPDF2
import docx
import openpyxl
from pptx import Presentation
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup


def extract_metadata_base(filepath: str) -> dict[str, Any]:
    """Extract common metadata from a file path."""
    filename = os.path.basename(filepath)
    ext = os.path.splitext(filename)[1].lower().lstrip(".")
    return {
        "filename": filename,
        "filetype": ext,
        "title": os.path.splitext(filename)[0],
        "author": None,
        "page_count": None,
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }


def extract_pdf(filepath: str) -> tuple[dict[str, Any], list[tuple[int, str]]]:
    """
    Extract text and metadata from a PDF file.

    Returns:
        Tuple of (metadata dict, list of (page_number, text) tuples).
    """
    metadata = extract_metadata_base(filepath)
    pages: list[tuple[int, str]] = []

    with open(filepath, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        info = reader.metadata

        if info:
            metadata["title"] = info.title or metadata["title"]
            metadata["author"] = info.author
        metadata["page_count"] = len(reader.pages)

        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((i + 1, text))

    return metadata, pages


def extract_docx(filepath: str) -> tuple[dict[str, Any], list[tuple[int, str]]]:
    """
    Extract text and metadata from a Word document.

    Returns:
        Tuple of (metadata dict, list of (section_number, text) tuples).
    """
    metadata = extract_metadata_base(filepath)

    doc = docx.Document(filepath)
    core = doc.core_properties
    metadata["title"] = core.title or metadata["title"]
    metadata["author"] = core.author

    paragraphs: list[tuple[int, str]] = []
    section_num = 1
    current_section: list[str] = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        # Split on headings to create logical sections
        if para.style and para.style.name.startswith("Heading"):
            if current_section:
                paragraphs.append((section_num, "\n".join(current_section)))
                section_num += 1
                current_section = []
        current_section.append(text)

    if current_section:
        paragraphs.append((section_num, "\n".join(current_section)))

    metadata["page_count"] = section_num

    return metadata, paragraphs


def extract_xlsx(filepath: str) -> tuple[dict[str, Any], list[tuple[int, str]]]:
    """
    Extract text and metadata from an Excel file.

    Returns:
        Tuple of (metadata dict, list of (sheet_number, text) tuples).
    """
    metadata = extract_metadata_base(filepath)

    wb = openpyxl.load_workbook(filepath, read_only=True, data_only=True)
    sheets: list[tuple[int, str]] = []

    for i, sheet_name in enumerate(wb.sheetnames):
        ws = wb[sheet_name]
        rows: list[str] = []
        rows.append(f"[Sheet: {sheet_name}]")

        for row in ws.iter_rows(values_only=True):
            cell_values = [str(cell) if cell is not None else "" for cell in row]
            row_text = " | ".join(cell_values).strip()
            if row_text.replace("|", "").strip():
                rows.append(row_text)

        if len(rows) > 1:  # More than just the header
            sheets.append((i + 1, "\n".join(rows)))

    metadata["page_count"] = len(wb.sheetnames)
    wb.close()

    return metadata, sheets


def extract_pptx(filepath: str) -> tuple[dict[str, Any], list[tuple[int, str]]]:
    """
    Extract text and metadata from a PowerPoint file.

    Returns:
        Tuple of (metadata dict, list of (slide_number, text) tuples).
    """
    metadata = extract_metadata_base(filepath)

    prs = Presentation(filepath)
    core = prs.core_properties
    metadata["title"] = core.title or metadata["title"]
    metadata["author"] = core.author
    metadata["page_count"] = len(prs.slides)

    slides: list[tuple[int, str]] = []

    for i, slide in enumerate(prs.slides):
        texts: list[str] = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for paragraph in shape.text_frame.paragraphs:
                    text = paragraph.text.strip()
                    if text:
                        texts.append(text)
        if texts:
            slides.append((i + 1, "\n".join(texts)))

    return metadata, slides


def extract_epub(filepath: str) -> tuple[dict[str, Any], list[tuple[int, str]]]:
    """
    Extract text and metadata from an EPUB file.

    Returns:
        Tuple of (metadata dict, list of (chapter_number, text) tuples).
    """
    metadata = extract_metadata_base(filepath)

    book = epub.read_epub(filepath)

    # Extract metadata
    title = book.get_metadata("DC", "title")
    if title:
        metadata["title"] = title[0][0]

    creator = book.get_metadata("DC", "creator")
    if creator:
        metadata["author"] = creator[0][0]

    chapters: list[tuple[int, str]] = []
    chapter_num = 1

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        content = item.get_content()
        soup = BeautifulSoup(content, "html.parser")
        text = soup.get_text(separator="\n", strip=True)

        if text.strip():
            chapters.append((chapter_num, text))
            chapter_num += 1

    metadata["page_count"] = len(chapters)

    return metadata, chapters


# Registry mapping file extensions to their extractor functions
EXTRACTORS: dict[str, callable] = {
    "pdf": extract_pdf,
    "docx": extract_docx,
    "xlsx": extract_xlsx,
    "pptx": extract_pptx,
    "epub": extract_epub,
}


def get_extractor(filepath: str):
    """Get the appropriate extractor function for a given file."""
    ext = os.path.splitext(filepath)[1].lower().lstrip(".")
    extractor = EXTRACTORS.get(ext)
    if extractor is None:
        raise ValueError(
            f"Unsupported file format: .{ext}. "
            f"Supported formats: {', '.join(EXTRACTORS.keys())}"
        )
    return extractor
