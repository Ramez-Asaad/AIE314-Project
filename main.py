"""
Entry point for the preprocessing pipeline.

Usage:
    python main.py
    python main.py --input data --output output --chunk-size 500 --overlap 50
"""

import argparse

from preprocessing import PreprocessingPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess unstructured documents for RAG pipeline."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data",
        help="Directory containing raw documents (default: data)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Directory for output JSON files (default: output)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Maximum chunk size in characters (default: 500)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Overlap between chunks in characters (default: 50)",
    )

    args = parser.parse_args()

    pipeline = PreprocessingPipeline(
        input_dir=args.input,
        output_dir=args.output,
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
    )

    print("🚀 Starting preprocessing pipeline...\n")
    pipeline.run()
    print("\n✨ Done!")


if __name__ == "__main__":
    main()
