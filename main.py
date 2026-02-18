"""
Entry point for the preprocessing pipeline.

Usage:
    python main.py
    python main.py --input data --output output --breakpoint-percentile 80
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
        "--breakpoint-percentile",
        type=int,
        default=80,
        help="Sensitivity for semantic topic-shift detection, 0-100 (default: 80). Lower = more chunks.",
    )
    parser.add_argument(
        "--min-chunk-size",
        type=int,
        default=100,
        help="Minimum chunk size in characters (default: 100)",
    )
    parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=1500,
        help="Maximum chunk size in characters (default: 1500)",
    )

    args = parser.parse_args()

    pipeline = PreprocessingPipeline(
        input_dir=args.input,
        output_dir=args.output,
        breakpoint_percentile=args.breakpoint_percentile,
        min_chunk_size=args.min_chunk_size,
        max_chunk_size=args.max_chunk_size,
    )

    print("🚀 Starting preprocessing pipeline...\n")
    pipeline.run()
    print("\n✨ Done!")


if __name__ == "__main__":
    main()
