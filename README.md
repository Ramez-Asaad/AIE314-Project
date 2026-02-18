# AIE314 Tutorial 1: Preprocessing Unstructured Data for LLM Applications

## Team Members
- Ramez Ezzat Asaad (ID: 22100506)
- Mariam Mohamed Sobhy (ID: 22100778)
- Abdelraheman Hamdy Khalil (ID: 22100692)
- Mahmoud Eid Khamis (ID: 22100680)

## Project Description
This project is part of the **AI-Based Programming (AIE314)** course. The overarching goal is to build an **AI Agentic Study & Courses Assistant** powered by Retrieval-Augmented Generation (RAG).

This tutorial focuses on the **preprocessing pipeline** — the foundational step of any RAG system. The pipeline ingests unstructured data from various document formats (PDF, Word, Excel, PowerPoint, EPUB) and normalizes them into structured JSON suitable for embedding and retrieval.

### Key Features
- **Multi-format support**: PDF, DOCX, XLSX, PPTX, EPUB
- **Text extraction & cleaning**: Strips noise, handles encoding artifacts, normalizes whitespace
- **Text normalization**: Unicode NFKC, typographic normalization, OCR repair, LaTeX cleanup, URL/email tagging
- **Metadata extraction**: Captures document title, author, page count, etc.
- **Semantic chunking**: Splits documents into topically coherent chunks using sentence embeddings (all-MiniLM-L6-v2)
- **Normalized JSON output**: Consistent schema across all document types

## Pipeline Architecture
```
 Raw Documents (PDF, DOCX, XLSX, PPTX, EPUB)
                 │
                 ▼
        ┌─────────────────┐
        │   1. EXTRACTION  │  extractors.py — format-specific text & metadata extraction
        └────────┬────────┘
                 ▼
        ┌─────────────────┐
        │   2. CLEANING    │  cleaners.py — encoding fixes, control chars, whitespace
        └────────┬────────┘
                 ▼
        ┌─────────────────┐
        │ 3. NORMALIZATION │  normalizers.py — NFKC, typography, OCR, LaTeX, URLs
        └────────┬────────┘
                 ▼
        ┌─────────────────┐
        │   4. CHUNKING    │  chunkers.py — semantic chunking via sentence embeddings
        └────────┬────────┘
                 ▼
        ┌─────────────────┐
        │   5. OUTPUT      │  pipeline.py — normalized JSON with metadata + chunks
        └─────────────────┘
```

## Project Structure
```
├── output/                  # Normalized JSON files for each processed document
├── data/                    # Raw input documents (not tracked in git)
├── preprocessing/           # Preprocessing pipeline modules
│   ├── __init__.py
│   ├── extractors.py        # Document text extractors (PDF, DOCX, etc.)
│   ├── cleaners.py          # Text cleaning & encoding fixes
│   ├── normalizers.py       # Text normalization (NFKC, typography, OCR, LaTeX, URLs)
│   ├── chunkers.py          # Chunking strategies (semantic, fixed-size, paragraph)
│   └── pipeline.py          # Main pipeline orchestrator
├── main.py                  # Entry point — run the full pipeline
├── requirements.txt         # Python dependencies
├── .gitignore
└── README.md
```

## How to Run the Code

### Prerequisites
- Python 3.12+
- pip

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Ramez-Asaad/AIE314-Project.git
   cd AIE314-Project
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Pipeline
1. Place your raw documents in the `data/` folder.

2. Run the preprocessing pipeline:
   ```bash
   python main.py
   ```

3. Processed JSON files will be saved to the `output/` folder.

### CLI Options
```bash
python main.py --input data --output output --breakpoint-percentile 80 --min-chunk-size 100 --max-chunk-size 1500
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | `data` | Directory containing raw documents |
| `--output` | `output` | Directory for output JSON files |
| `--breakpoint-percentile` | `80` | Sensitivity for semantic topic-shift detection (0-100). Lower = more chunks |
| `--min-chunk-size` | `100` | Minimum chunk size in characters |
| `--max-chunk-size` | `1500` | Maximum chunk size in characters |

## Normalization Details

The pipeline applies 6 normalization techniques chosen specifically for RAG (preserving semantic meaning while reducing noise):

| # | Technique | Example |
|---|-----------|---------|
| 1 | **Unicode NFKC** | `ﬁ` → `fi`, `²` → `2`, `Ａ` → `A` |
| 2 | **Typographic** | Smart quotes → straight quotes, `•` → `-`, `…` → `...` |
| 3 | **OCR Artifact Repair** | `m a c h i n e` → `machine`, broken hyphens fixed |
| 4 | **Header/Footer Removal** | Strips standalone page numbers (`Page 5`, `- 42 -`) |
| 5 | **LaTeX Cleanup** | `\alpha` → `alpha`, `\frac{1}{2}` → `(1/2)` |
| 6 | **URL/Email Tagging** | `https://...` → `[URL: https://...]` |

**Techniques intentionally avoided** (they degrade embedding quality):
- Case folding — embeddings are case-aware
- Stopword removal — destroys sentence structure
- Lemmatization/Stemming — modern embeddings handle morphology natively

## Chunking Strategy

The default chunking strategy is **semantic chunking** using the `all-MiniLM-L6-v2` sentence-transformer model:

1. Split text into sentences
2. Embed each sentence
3. Compute cosine similarity between consecutive sentences
4. Detect topic shifts where similarity drops below a percentile threshold
5. Group sentences between breakpoints into coherent chunks
6. Post-process: merge tiny chunks, split oversized ones

Alternative strategies (`chunk_by_fixed_size`, `chunk_by_paragraph`) are available as fallbacks.

## Output Format
Each processed document produces a JSON file with the following schema:
```json
{
  "metadata": {
    "filename": "example.pdf",
    "filetype": "pdf",
    "title": "Document Title",
    "author": "Author Name",
    "page_count": 10,
    "processed_at": "2026-02-18T18:00:00Z"
  },
  "chunks": [
    {
      "chunk_id": 0,
      "text": "Extracted, cleaned, and normalized text content...",
      "page": 1
    }
  ]
}
```
