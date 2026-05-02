# Encode-Only SOTA Proof Plan

This project should only claim SOTA after the benchmark is apples-to-apples,
reproducible, and parity-checked. The current claim target is:

> Encode-only CPU throughput for a tokenizer model with exact token-ID parity.

## Gates

1. Define the category: encode only, CPU, single process, no file I/O in timing.
2. Lock correctness: every timed corpus records token count and token-ID hash.
3. Use real baselines: Hugging Face `tokenizers`, especially the tokenizer used
   by `Qwen/Qwen3-0.6B`, plus `tiktoken`/GPT-2 for the current C++
   compatibility path.
4. Use a neutral harness: same corpus files, warmup, repeated median timing.
5. Use serious corpora: short, medium, large, Unicode, code, punctuation, and
   Qwen chat-template-like text.
6. Report real metrics: MiB/s, token count, token hash, corpus hashes, machine
   metadata, dependency paths.
7. Control hardware: run on a quiet machine, pin versions, and use Release
   builds. For a publishable run, record CPU governor/turbo state separately.
8. Publish artifacts: keep the harness and raw JSON output reproducible.
9. Adversarial review: unsupported paths and parity failures must appear in the
   report instead of being hidden in prose.

## Current Limitation

The C++ tokenizer currently supports the native/GPT-2-style byte-level BPE
format. Qwen3 uses Hugging Face `tokenizer.json` semantics:

- NFC normalization
- a Qwen-specific regex pre-tokenizer
- ByteLevel vocabulary encoding/decoding
- added special tokens

So a Qwen3 SOTA claim is not valid until the C++ library can load and encode
the Qwen3 tokenizer with exact token-ID parity. The harness marks this as an
explicit unsupported gate.

## Run

Install Python dependencies in any environment:

```bash
python3 -m pip install tokenizers huggingface_hub tiktoken
```

Then run:

```bash
python3 scripts/sota_encode_report.py
```

The script writes `runs/sota_encode_report.json` and prints a compact summary.
If Rust/Cargo is available, it also builds a native Rust `tokenizers` probe so
we are not relying only on Python binding throughput.
