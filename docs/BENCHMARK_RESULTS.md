# Benchmark Results

This summary is derived from `runs/sota_encode_report.json` generated on clean
commit `0f3ecd5d63798754634de661f9baba38e682acae` with `git.dirty=false`.

## Scope

The supported public claim is narrow:

> Encode-only CPU throughput on the recorded NFC-normalized harness, with exact
> GPT-2 and Qwen3 token-ID parity against the recorded baselines.

This is not a claim about decode speed, training speed, arbitrary Hugging Face
tokenizer pipelines, GPU workloads, or hardware outside the recorded run.

## Run Configuration

```bash
python3 scripts/sota_encode_report.py \
  --repeats 21 \
  --target-bytes 16777216 \
  --service-threads 8 \
  --out runs/sota_encode_report.json
```

- Machine: Apple M4 Max, Darwin arm64, 14 logical CPUs
- Python: 3.14.4
- Python baselines: `tokenizers==0.23.1`, `tiktoken==0.12.0`,
  `huggingface_hub==1.10.0`
- Rust baseline crate: `tokenizers==0.23.1`
- Corpora: 19 byte-exact UTF-8 files, NFC-normalized before timing
- Metrics recorded in JSON: MiB/s, token count, token-ID hash, corpus hash,
  p50/p95/p99 per-call latency, peak RSS, batch size, repeat count, thread
  count, machine metadata, dependency versions, git commit, and dirty state

## Gates

| Gate | Status |
| --- | --- |
| Qwen3 HF Rust vs HF Python parity | pass |
| GPT-2 HF Rust vs HF Python parity | pass |
| GPT-2 C++ vs HF Python parity | pass |
| GPT-2 C++ vs tiktoken parity | pass |
| Qwen3 C++ vs HF Python parity | pass |
| Qwen3 C++ vs HF Rust parity | pass |
| Qwen3 C++ throughput beats HF baselines on every corpus | pass |
| GPT-2 C++ throughput beats HF/tiktoken baselines on every corpus | pass |

## Single-Thread Throughput

Ranges are min to max MiB/s across the 19 recorded corpora.

| Implementation | Range |
| --- | ---: |
| C++ GPT-2 | 7.83-261.05 MiB/s |
| C++ Qwen3 | 8.57-174.59 MiB/s |
| tiktoken GPT-2 | 0.42-36.98 MiB/s |
| HF Rust GPT-2 | 2.94-14.13 MiB/s |
| HF Rust Qwen3 | 4.84-16.19 MiB/s |
| HF Python GPT-2 | 3.41-9.88 MiB/s |
| HF Python Qwen3 | 3.93-9.77 MiB/s |

## Speedups

Ranges are corpus-by-corpus speedups from the same report.

| Comparison | Range |
| --- | ---: |
| C++ Qwen3 vs HF Rust Qwen3 | 1.39x-24.04x |
| C++ Qwen3 vs HF Python Qwen3 | 1.59x-30.07x |
| C++ GPT-2 vs tiktoken GPT-2 | 1.26x-18.80x |
| C++ GPT-2 vs HF Rust GPT-2 | 1.20x-39.34x |

## Shared-Tokenizer Serving

The serving measurement uses one loaded shared tokenizer with 8 concurrent
request threads. File I/O and model loading are outside the timed region.
Ranges are min to max MiB/s across the 19 corpora.

| Implementation | Range |
| --- | ---: |
| C++ GPT-2 | 57.91-1981.62 MiB/s |
| C++ Qwen3 | 64.05-1305.38 MiB/s |
| HF Rust GPT-2 | 16.81-68.61 MiB/s |
| HF Rust Qwen3 | 20.82-70.82 MiB/s |

## Corpora

The corpus set covers `short`, `medium`, `large`, `unicode`,
`unicode_numbers`, `code`, `python`, `javascript`, `jsonl_logs`, `markdown`,
`urls_base64`, `emoji_zwj`, `cjk_dense`, `rtl`, `punct`, `blank_lines`,
`whitespace_pathological`, `long_word`, and `qwen_chat`.

## Caveats

- The Qwen3 path is a strict supported Hugging Face ByteLevel BPE subset, not a
  complete implementation of every HF tokenizer graph.
- Corpora are NFC-normalized for this harness. That is part of the claim.
- The raw JSON report is the source of truth for per-corpus details; this file
  is a sanitized, tracked summary for public release.
