"""
Main preprocessing pipeline orchestrator.
Ties together extraction, cleaning, and chunking into a single workflow.
"""

import json
import os
from typing import Any

from tqdm import tqdm

from .extractors import get_extractor
from .cleaners import clean_text
from .chunkers import DEFAULT_CHUNKER


class PreprocessingPipeline:
    """
    Orchestrates the full preprocessing pipeline:
    1. Extract text & metadata from documents
    2. Clean and normalize extracted text
    3. Chunk text into segments
    4. Output normalized JSON
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".pptx", ".epub"}

    def __init__(
        self,
        input_dir: str = "data",
        output_dir: str = "output",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        os.makedirs(self.output_dir, exist_ok=True)

    def discover_files(self) -> list[str]:
        """Find all supported documents in the input directory."""
        if not os.path.exists(self.input_dir):
            print(f"⚠ Input directory '{self.input_dir}' does not exist. Creating it...")
            os.makedirs(self.input_dir, exist_ok=True)
            return []

        files: list[str] = []
        for root, _, filenames in os.walk(self.input_dir):
            for fname in filenames:
                ext = os.path.splitext(fname)[1].lower()
                if ext in self.SUPPORTED_EXTENSIONS:
                    files.append(os.path.join(root, fname))

        return sorted(files)

    def process_file(self, filepath: str) -> dict[str, Any]:
        """
        Process a single document through the full pipeline.

        Returns:
            Normalized document dict with metadata and chunks.
        """
        # 1. Extract
        extractor = get_extractor(filepath)
        metadata, pages = extractor(filepath)

        # 2. Clean
        cleaned_pages: list[tuple[int, str]] = []
        for page_num, text in pages:
            cleaned = clean_text(text)
            if cleaned:
                cleaned_pages.append((page_num, cleaned))

        # 3. Chunk
        chunks: list[dict[str, Any]] = []
        chunk_id = 0

        for page_num, text in cleaned_pages:
            page_chunks = DEFAULT_CHUNKER(
                text,
                chunk_size=self.chunk_size,
                overlap=self.chunk_overlap,
            )
            for chunk_text in page_chunks:
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "text": chunk_text,
                        "page": page_num,
                    }
                )
                chunk_id += 1

        return {
            "metadata": metadata,
            "chunks": chunks,
        }

    def save_output(self, document: dict[str, Any], filepath: str) -> str:
        """Save processed document as a JSON file."""
        basename = os.path.splitext(os.path.basename(filepath))[0]
        output_path = os.path.join(self.output_dir, f"{basename}.json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(document, f, indent=2, ensure_ascii=False)

        return output_path

    def run(self) -> list[str]:
        """
        Run the full preprocessing pipeline on all documents.

        Returns:
            List of output JSON file paths.
        """
        files = self.discover_files()

        if not files:
            print("📭 No supported documents found in the input directory.")
            print(f"   Place your files in: {os.path.abspath(self.input_dir)}")
            print(f"   Supported formats: {', '.join(self.SUPPORTED_EXTENSIONS)}")
            return []

        print(f"📄 Found {len(files)} document(s) to process.\n")

        output_paths: list[str] = []
        errors: list[tuple[str, str]] = []

        for filepath in tqdm(files, desc="Processing documents"):
            try:
                document = self.process_file(filepath)
                output_path = self.save_output(document, filepath)
                output_paths.append(output_path)
                tqdm.write(f"  ✅ {os.path.basename(filepath)} → {output_path}")
            except Exception as e:
                errors.append((filepath, str(e)))
                tqdm.write(f"  ❌ {os.path.basename(filepath)}: {e}")

        # Summary
        print(f"\n{'='*50}")
        print(f"📊 Pipeline Summary")
        print(f"   Processed: {len(output_paths)}/{len(files)}")
        print(f"   Errors:    {len(errors)}")
        print(f"   Output:    {os.path.abspath(self.output_dir)}")

        if errors:
            print(f"\n⚠ Failed files:")
            for path, err in errors:
                print(f"   - {os.path.basename(path)}: {err}")

        return output_paths
