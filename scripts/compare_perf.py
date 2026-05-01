#!/usr/bin/env python3
"""
Compare encoding performance: C++ tokenizer vs tiktoken vs HuggingFace.

Usage:
    # First build the C++ tokenizer and generate a CLI:
    #   cmake --build build --target bpe_bench
    #
    # Then run this comparison:
    python3 scripts/compare_perf.py [--corpus PATH] [--vocab-size 10000]

This script:
  1. Trains/loads each tokenizer on the same corpus.
  2. Benchmarks encoding throughput (MB/s).
  3. Benchmarks decoding throughput (tokens/s).
  4. Prints a comparison table.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def bench_tiktoken(text: str, n_iters: int = 100):
    """Benchmark tiktoken GPT-2 encoding."""
    try:
        import tiktoken
    except ImportError:
        return None

    enc = tiktoken.get_encoding("gpt2")

    # Warmup
    for _ in range(5):
        enc.encode(text)

    # Encode benchmark
    start = time.perf_counter()
    for _ in range(n_iters):
        ids = enc.encode(text)
    elapsed_encode = time.perf_counter() - start

    # Decode benchmark
    ids = enc.encode(text)
    start = time.perf_counter()
    for _ in range(n_iters):
        enc.decode(ids)
    elapsed_decode = time.perf_counter() - start

    text_bytes = len(text.encode("utf-8"))
    return {
        "name": "tiktoken (gpt2)",
        "encode_mb_s": (text_bytes * n_iters) / elapsed_encode / 1e6,
        "decode_tok_s": (len(ids) * n_iters) / elapsed_decode,
        "encode_time_ms": elapsed_encode / n_iters * 1000,
        "decode_time_ms": elapsed_decode / n_iters * 1000,
        "n_tokens": len(ids),
        "compression_ratio": text_bytes / len(ids),
    }


def bench_huggingface(text: str, n_iters: int = 100):
    """Benchmark HuggingFace tokenizers (GPT-2)."""
    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_pretrained("gpt2")
    except (ImportError, Exception):
        return None

    # Warmup
    for _ in range(5):
        tok.encode(text)

    # Encode benchmark
    start = time.perf_counter()
    for _ in range(n_iters):
        output = tok.encode(text)
    elapsed_encode = time.perf_counter() - start

    ids = tok.encode(text).ids

    # Decode benchmark
    start = time.perf_counter()
    for _ in range(n_iters):
        tok.decode(ids)
    elapsed_decode = time.perf_counter() - start

    text_bytes = len(text.encode("utf-8"))
    return {
        "name": "HuggingFace tokenizers (gpt2)",
        "encode_mb_s": (text_bytes * n_iters) / elapsed_encode / 1e6,
        "decode_tok_s": (len(ids) * n_iters) / elapsed_decode,
        "encode_time_ms": elapsed_encode / n_iters * 1000,
        "decode_time_ms": elapsed_decode / n_iters * 1000,
        "n_tokens": len(ids),
        "compression_ratio": text_bytes / len(ids),
    }


def print_table(results, text_size_kb):
    """Print a formatted comparison table."""
    print(f"\n{'='*75}")
    print(f"  Performance Comparison — {text_size_kb:.1f} KB input")
    print(f"{'='*75}")
    print(f"{'Tokenizer':<35} {'Encode MB/s':>12} {'Decode tok/s':>14} {'Tokens':>8} {'Ratio':>6}")
    print(f"{'-'*75}")

    for r in results:
        if r is None:
            continue
        print(f"{r['name']:<35} {r['encode_mb_s']:>11.1f} {r['decode_tok_s']:>13,.0f} "
              f"{r['n_tokens']:>8,} {r['compression_ratio']:>5.1f}x")

    print(f"{'='*75}")

    # Relative performance
    if len([r for r in results if r]) >= 2:
        valid = [r for r in results if r]
        baseline = valid[0]
        print(f"\nRelative to {baseline['name']}:")
        for r in valid[1:]:
            encode_rel = r['encode_mb_s'] / baseline['encode_mb_s']
            print(f"  {r['name']}: {encode_rel:.2f}x encode speed")


def main():
    parser = argparse.ArgumentParser(description="BPE tokenizer performance comparison")
    parser.add_argument("--corpus", type=str, help="Path to corpus file")
    parser.add_argument("--iters", type=int, default=100, help="Number of iterations")
    args = parser.parse_args()

    # Generate test text
    if args.corpus:
        with open(args.corpus) as f:
            text = f.read()
    else:
        # Default: medium English text (~60KB)
        base = (
            "In a small village, there lived a young girl named Lily. She loved to explore "
            "the forest near her home. One day, she found a shiny stone by the river. "
            "The stone glowed with a warm light. Lily picked it up and felt happy. "
            "She showed it to her friend Tom. Tom said it was a magic stone. "
            "They decided to go on an adventure together. They walked through the tall trees "
            "and crossed the old bridge. On the other side, they found a beautiful garden. "
            "The garden had flowers of every color. Lily and Tom played there until sunset. "
            "When they went home, Lily put the stone on her shelf. It still glowed softly. "
            "She smiled and went to sleep, dreaming of their next adventure.\n"
        )
        text = base * 100

    text_size_kb = len(text.encode("utf-8")) / 1024
    print(f"Corpus size: {text_size_kb:.1f} KB")
    print(f"Iterations: {args.iters}")

    results = []

    # Benchmark tiktoken
    print("\nBenchmarking tiktoken...")
    r = bench_tiktoken(text, args.iters)
    if r:
        results.append(r)
        print(f"  Encode: {r['encode_mb_s']:.1f} MB/s")
    else:
        print("  [skipped — tiktoken not installed]")

    # Benchmark HuggingFace
    print("Benchmarking HuggingFace tokenizers...")
    r = bench_huggingface(text, args.iters)
    if r:
        results.append(r)
        print(f"  Encode: {r['encode_mb_s']:.1f} MB/s")
    else:
        print("  [skipped — tokenizers not installed]")

    # Note about C++ benchmark
    print("\nC++ benchmark:")
    print("  Run: ./build/bpe_bench --benchmark_format=console")
    print("  Then compare the encode throughput numbers above.")

    print_table(results, text_size_kb)

    # Save results for later comparison.
    output_path = Path(__file__).parent.parent / "tests" / "fixtures" / "perf_baseline.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "corpus_size_bytes": len(text.encode("utf-8")),
            "iterations": args.iters,
            "results": [r for r in results if r],
        }, f, indent=2)
    print(f"\nBaseline saved to {output_path}")


if __name__ == "__main__":
    main()
