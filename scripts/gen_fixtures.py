#!/usr/bin/env python3
"""
Generate reference fixtures for correctness tests.

This script produces:
  1. GPT-2 vocab/merges files (for loading into the C++ tokenizer)
  2. tiktoken reference encodings for various test strings
  3. Training reference (merges from a known corpus)

Usage:
    pip install tiktoken
    python3 scripts/gen_fixtures.py
    python3 scripts/gen_fixtures.py --holdout   # generate held-out reward-hacking check set

Output:
    tests/fixtures/reference_encodings.json
    tests/fixtures/gpt2/vocab.json
    tests/fixtures/gpt2/merges.txt
    tests/fixtures/holdout_encodings.json  (--holdout only)

HOLDOUT notes:
    The held-out fixture set is NEVER shown to the mutation LLMs — it's
    consumed by the orchestrator's periodic reward-hacking check (see plan
    "Reward-hacking mitigation"). Regenerating it each time would let an
    overfit candidate game the check, so we write it once and then only
    refresh if the user passes --holdout --force.
"""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path


def _build_holdout_strings(seed: int) -> list[str]:
    """Synthesize a diverse, reproducible holdout set.

    Strategy: mix short / medium / Unicode / contractions / code-like /
    whitespace categories, then add 8 randomly-sampled slices of Lorem
    Ipsum + CJK seed phrases to guarantee the agents can't just memoize
    the public fixture set.
    """
    rng = random.Random(seed)
    base = [
        "Quantum entanglement doesn't care about your feelings.",
        "She'd already told him they'd never be back.",
        "\n\n   \t  \n  ",
        "for(int i=0;i<n;++i){arr[i]*=2;}",
        "Résumé — café naïve jalapeño",
        "東京タワーは高いです。 서울은 한국의 수도입니다。 你好, 世界! 🌏",
        "https://example.com/path?q=42&lang=fr#anchor",
        "-3.1415e-7 0xDEADBEEF 0b1010 1_000_000",
        "   Lead space, trail space   ",
        "MixedCASE identifier_like_this and kebab-cased-thing",
    ]
    lorem = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do "
        "eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim "
        "ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut "
        "aliquip ex ea commodo consequat. Duis aute irure dolor in "
        "reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla "
        "pariatur."
    )
    for _ in range(8):
        start = rng.randint(0, max(0, len(lorem) - 120))
        length = rng.randint(40, 200)
        base.append(lorem[start:start + length])
    return base


def run_holdout(force: bool) -> int:
    try:
        import tiktoken
    except ImportError:
        print("ERROR: tiktoken not installed. Run: .venv/bin/pip install tiktoken")
        return 1

    fixtures_dir = Path(__file__).parent.parent / "tests" / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    out = fixtures_dir / "holdout_encodings.json"
    if out.exists() and not force:
        print(f"{out} already exists; pass --force to regenerate.")
        return 0

    # Fixed seed keeps the set reproducible across machines / clones, but the
    # file content itself is the source of truth once written.
    seed = 0xC0DEC
    strings = _build_holdout_strings(seed)
    enc = tiktoken.get_encoding("gpt2")
    encodings = {s: enc.encode(s, allowed_special=set()) for s in strings}

    payload = {
        "seed": seed,
        "hash": hashlib.sha256(json.dumps(encodings, sort_keys=True).encode()).hexdigest(),
        "note": (
            "NEVER expose this file to mutation LLMs. It is consumed by the "
            "orchestrator's periodic reward-hacking check — if any candidate "
            "regresses on these strings, its lineage is flagged for review."
        ),
        "count": len(encodings),
        "encodings": encodings,
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Wrote {out} ({len(encodings)} strings, hash {payload['hash'][:12]}...)")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Generate reference fixtures")
    ap.add_argument("--holdout", action="store_true",
                    help="Generate the held-out reward-hacking check set and exit")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing holdout file")
    args = ap.parse_args()

    if args.holdout:
        sys.exit(run_holdout(args.force))

    try:
        import tiktoken
    except ImportError:
        print("ERROR: tiktoken not installed. Run: pip install tiktoken")
        sys.exit(1)

    fixtures_dir = Path(__file__).parent.parent / "tests" / "fixtures"
    gpt2_dir = fixtures_dir / "gpt2"
    gpt2_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Export GPT-2 vocab and merges ────────────────────────────
    print("Exporting GPT-2 vocab and merges...")
    enc = tiktoken.get_encoding("gpt2")

    # tiktoken doesn't directly expose vocab/merges in the traditional format.
    # We need to get them from the underlying data.
    # The mergeable_ranks maps byte strings to token IDs.
    mergeable_ranks = enc._mergeable_ranks

    # Build vocab.json: map from hex-encoded byte string to ID.
    # We use hex encoding because byte strings may contain non-UTF8 bytes.
    vocab = {}
    for token_bytes, token_id in mergeable_ranks.items():
        # Encode as list of ints for unambiguous serialization.
        key = token_bytes.hex()
        vocab[key] = token_id

    with open(gpt2_dir / "vocab.json", "w") as f:
        json.dump(vocab, f, indent=2)

    # Build merges.txt from the BPE merge file.
    # tiktoken uses a different format, so we reconstruct merges.
    # The merge order is implied by the token IDs (lower ID = earlier merge).
    # Tokens 0-255 are single bytes; 256+ are merges.
    sorted_tokens = sorted(
        [(tid, tb) for tb, tid in mergeable_ranks.items()],
        key=lambda x: x[0]
    )

    merges = []
    for tid, token_bytes in sorted_tokens:
        if tid < 256:
            continue  # single byte, not a merge
        # Find the split point: try all possible splits and find the one
        # where both halves have lower IDs.
        best_split = None
        for i in range(1, len(token_bytes)):
            left = token_bytes[:i]
            right = token_bytes[i:]
            if left in mergeable_ranks and right in mergeable_ranks:
                left_id = mergeable_ranks[left]
                right_id = mergeable_ranks[right]
                if left_id < tid and right_id < tid:
                    if best_split is None or (left_id + right_id) < (best_split[0] + best_split[1]):
                        best_split = (left_id, right_id, left.hex(), right.hex())
        if best_split:
            merges.append(f"{best_split[2]} {best_split[3]}")

    with open(gpt2_dir / "merges.txt", "w") as f:
        f.write("\n".join(merges))

    print(f"  Exported {len(vocab)} vocab entries, {len(merges)} merges")

    # ── 2. Generate reference encodings ─────────────────────────────
    print("Generating reference encodings...")
    test_strings = {
        # Basic ASCII
        "Hello, world!": None,
        "The quick brown fox jumps over the lazy dog.": None,
        # Whitespace
        "  hello  world  ": None,
        "\n\n\n": None,
        "\t\ttab": None,
        # Contractions
        "I'm happy. You're great. They've gone. We'll see. He'd know. She's here.": None,
        # Numbers
        "12345 3.14 -42": None,
        # Punctuation
        "Hello!!! What??? Really...": None,
        # Unicode
        "café résumé naïve": None,
        # Mixed
        "Hello 123 world! I'm fine.": None,
        # Empty
        "": None,
        # Single char
        "a": None,
        # Repeated
        "aaaa": None,
        # Code-like
        "def foo(x): return x + 1": None,
        "import numpy as np": None,
        # Long
        "the " * 100: None,
    }

    tiktoken_encodings = {}
    tiktoken_counts = {}
    for text in test_strings:
        ids = enc.encode(text, allowed_special=set())
        tiktoken_encodings[text] = ids
        tiktoken_counts[text] = len(ids)

    # ── 3. Generate training reference ──────────────────────────────
    print("Generating training reference...")

    # Use a simple known corpus so we can verify merge order.
    training_corpus = (
        "the cat sat on the mat. the cat ate the rat. "
        "the dog sat on the log. the dog ate the frog."
    )

    # We'll compute the expected merges using a simple Python BPE.
    def simple_bpe_train(text, num_merges):
        """Minimal BPE trainer matching CS336 spec."""
        # Pre-tokenize by splitting on spaces (simplified).
        import re
        pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+""")
        chunks = [list(match.encode("utf-8")) for match in pat.findall(text)]

        merges_list = []
        for _ in range(num_merges):
            # Count pairs.
            pair_counts = {}
            for chunk in chunks:
                for i in range(len(chunk) - 1):
                    pair = (chunk[i], chunk[i + 1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1
            if not pair_counts:
                break

            # Find best pair (highest count, tie-break by larger pair).
            best = max(pair_counts, key=lambda p: (pair_counts[p], p))
            new_id = 256 + len(merges_list)

            # Apply merge.
            for chunk in chunks:
                i = 0
                new_chunk = []
                while i < len(chunk):
                    if i < len(chunk) - 1 and chunk[i] == best[0] and chunk[i + 1] == best[1]:
                        new_chunk.append(new_id)
                        i += 2
                    else:
                        new_chunk.append(chunk[i])
                        i += 1
                chunk[:] = new_chunk

            merges_list.append({
                "left": best[0],
                "right": best[1],
                "merged": new_id,
                "count": pair_counts[best],
            })
        return merges_list

    num_merges = 24  # 256 + 24 = 280 vocab size
    training_merges = simple_bpe_train(training_corpus, num_merges)

    # ── 4. Write everything out ─────────────────────────────────────
    reference = {
        "tiktoken_gpt2": tiktoken_encodings,
        "tiktoken_counts": tiktoken_counts,
        "training_reference": {
            "corpus": training_corpus,
            "vocab_size": 256 + num_merges,
            "merges": training_merges,
        },
    }

    with open(fixtures_dir / "reference_encodings.json", "w") as f:
        json.dump(reference, f, indent=2)

    print(f"\nDone! Fixtures written to {fixtures_dir}")
    print(f"  reference_encodings.json: {len(tiktoken_encodings)} test cases")
    print(f"  gpt2/vocab.json: {len(vocab)} entries")
    print(f"  gpt2/merges.txt: {len(merges)} merges")
    print(f"  training_reference: {len(training_merges)} merges")


if __name__ == "__main__":
    main()
