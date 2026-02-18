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
- **Text extraction & cleaning**: Strips noise, normalizes whitespace, handles encoding
- **Metadata extraction**: Captures document title, author, page count, etc.
- **Chunking**: Splits documents into semantically meaningful chunks for downstream embedding
- **Normalized JSON output**: Consistent schema across all document types

## Project Structure
```
├── output/                  # Normalized JSON files for each processed document
├── data/                    # Raw input documents (not tracked in git)
├── preprocessing/           # Preprocessing pipeline modules
│   ├── __init__.py
│   ├── extractors.py        # Document text extractors (PDF, DOCX, etc.)
│   ├── cleaners.py          # Text cleaning & normalization utilities
│   ├── chunkers.py          # Text chunking strategies
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
      "text": "Extracted and cleaned text content...",
      "page": 1
    }
  ]
}
```
