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
6. Report real metrics: MiB/s, token count, token hash, p50/p95/p99 per-call
   latency, peak RSS, corpus hashes, machine metadata, dependency paths, batch
   size, repeat count, and thread count.
7. Control hardware: run on a quiet machine, pin versions, and use Release
   builds. For a publishable run, record CPU governor/turbo state separately.
8. Publish artifacts: keep the harness and raw JSON output reproducible.
9. Adversarial review: unsupported paths and parity failures must appear in the
   report instead of being hidden in prose.
10. Throughput gate: C++ must beat the recorded HF/tiktoken baselines on every
    corpus before the generated report marks the scoped claim as met.
11. Serving measurement: measure a loaded shared tokenizer under concurrent
    encode calls, with the configured request-thread count recorded in JSON.

## Current Status

The C++ tokenizer supports the native/GPT-2-style byte-level BPE format and a
strict Hugging Face ByteLevel BPE `tokenizer.json` subset. That subset is
intended to cover the tokenizer used by `Qwen/Qwen3-0.6B` for encode-only
benchmarks:

- NFC normalization during encode
- ByteLevel vocabulary decoding
- ordered BPE merges
- the Qwen Split+ByteLevel pre-tokenizer sequence
- literal added-token matching for Qwen chat/control tokens

Unsupported Hugging Face features fail closed instead of being approximated:
`byte_fallback`, `ignore_merges`, added-token boundary/strip/normalizer flags,
pre-tokenizer graphs other than the supported Qwen sequence, and general
normalizer/post-processor pipelines.

The current benchmark harness NFC-normalizes its corpora before timing and then
checks exact token-count and token-ID-hash parity against Hugging Face Python
and Rust tokenizers. Corpus files are written and read as UTF-8 bytes so CRLF
and other whitespace cases are not changed by platform newline translation. The
corpus suite includes short prompts, medium and large
prose, C++/Python/JavaScript code, Markdown, JSONL logs, URL/base64-like text,
Unicode number forms, dense CJK, RTL Arabic/Hebrew, emoji/ZWJ sequences,
punctuation, blank-line and whitespace-heavy cases, long repeated words, and
Qwen chat-template-like text.

The generated status is marked met only when parity passes and C++ throughput
beats the recorded baselines on every corpus. That supports a scoped claim:

> Encode-only throughput on the recorded NFC-normalized harness, for models and
> corpora whose token-ID parity passes.

It does not support a broad claim that the library is a complete Hugging Face
tokenizers replacement or a universal tokenizer SOTA across arbitrary HF
pipeline configurations, decode/post-processing workloads, or hardware not
represented by the raw report.

Latency percentiles are per encode call and estimated from repeated fixed-size
batches. For public claims, run with enough repeats for stable tails. Peak RSS
is reported by the measured process: Python baselines run inside the report
script, while the C++ and Rust probes report their own process peaks.

## Run

Install Python dependencies in any environment:

```bash
python3 -m pip install tokenizers==0.23.1 huggingface_hub==1.10.0 tiktoken==0.12.0
```

Then run:

```bash
python3 scripts/sota_encode_report.py
```

For a quick local smoke that still exercises parity, latency fields, RSS fields,
and the report schema:

```bash
python3 scripts/sota_encode_report.py \
  --repeats 2 \
  --target-bytes 1048576 \
  --service-threads 4
```

For publishable numbers, use a quiet machine and a larger run, for example:

```bash
python3 scripts/sota_encode_report.py \
  --repeats 21 \
  --target-bytes 16777216 \
  --service-threads 8
```

The script writes `runs/sota_encode_report.json` and prints a compact summary.
The JSON includes machine metadata, git metadata, command-line arguments,
corpus hashes, single-thread results, multi-thread serving results, parity
gates, and throughput gates. If Rust/Cargo is available, the script also builds
a native Rust `tokenizers` probe so we are not relying only on Python binding
throughput.
