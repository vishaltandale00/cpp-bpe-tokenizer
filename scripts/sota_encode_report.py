#!/usr/bin/env python3
"""
Encode-only SOTA proof harness.

This script is intentionally stricter than the library benchmarks:
  - it records the exact claim scope,
  - validates token-id parity where an apples-to-apples model is available,
  - measures Hugging Face tokenizers for the Qwen3 tokenizer through both the
    Rust crate and the Python binding when possible,
  - writes raw machine, corpus, parity, and timing data to JSON.

The C++ library supports its native/GPT-2-style byte-level BPE format and a
strict subset of Hugging Face ByteLevel BPE tokenizer.json files, including the
Qwen3 tokenizer used by the default benchmark. The benchmark normalizes corpora
to NFC before timing because the library does not implement a general HF
normalizer pipeline.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import textwrap
import time
import unicodedata
from pathlib import Path
from typing import Callable

try:
    import resource
except ImportError:  # pragma: no cover - resource is available on the intended Unix hosts.
    resource = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "runs" / "sota_encode_report.json"
DEFAULT_CACHE = REPO_ROOT / ".cache" / "sota_encode"
TARGET_SAMPLE_BYTES = 8 * 1024 * 1024
DEFAULT_SERVICE_THREADS = max(1, min(8, os.cpu_count() or 1))
HF_RUST_TOKENIZERS_VERSION = "0.23.1"


def run(cmd: list[str], cwd: Path = REPO_ROOT) -> None:
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def load_corpora() -> dict[str, str]:
    corpora = {
        "short": "The quick brown fox jumps over the lazy dog. " * 2,
        "medium": (
            "In the beginning the Universe was created. "
            "This has made a lot of people very angry and been widely regarded as a bad move. "
        ) * 12,
        "unicode": (
            "こんにちは世界 🌍 你好世界 안녕하세요 세계 "
            "مرحبا بالعالم Привет мир 🚀✨🔥 "
        ) * 20,
        "unicode_numbers": (
            " ①test ¼test Ⅷ apples ٢٣ things १२३ items １２３ boxes "
        ) * 40,
        "code": (
            "template <typename It>\n"
            "uint64_t hash_bytes(It begin, It end) {\n"
            "    uint64_t h = 1469598103934665603ULL;\n"
            "    for (It it = begin; it != end; ++it) {\n"
            "        h ^= static_cast<uint8_t>(*it);\n"
            "        h *= 1099511628211ULL;\n"
            "    }\n"
            "    return h;\n"
            "}\n"
        ) * 64,
        "python": (
            "def tokenize_batch(tokenizer, rows):\n"
            "    result = []\n"
            "    for row in rows:\n"
            "        if row.get('enabled'):\n"
            "            result.append(tokenizer.encode(row['text']))\n"
            "    return result\n\n"
        ) * 80,
        "javascript": (
            "export function groupBy(items, keyFn) {\n"
            "  const groups = new Map();\n"
            "  for (const item of items) {\n"
            "    const key = keyFn(item);\n"
            "    groups.set(key, [...(groups.get(key) ?? []), item]);\n"
            "  }\n"
            "  return groups;\n"
            "}\n"
        ) * 70,
        "jsonl_logs": (
            '{"ts":"2026-05-04T10:15:30Z","level":"INFO","path":"/v1/chat/completions",'
            '"request_id":"req_01HY7D4S9","latency_ms":37.42,"tokens":512}\n'
        ) * 180,
        "markdown": (
            "# Tokenizer Benchmark Notes\n\n"
            "| corpus | bytes | tokens | parity |\n"
            "| --- | ---: | ---: | --- |\n"
            "| unicode | 2048 | 540 | pass |\n\n"
            "- Keep file I/O outside the timed region.\n"
            "- Record exact hashes for each reference implementation.\n\n"
        ) * 60,
        "urls_base64": (
            "GET /api/v2/search?q=tokenizer%20benchmark&cursor=eyJwYWdlIjoyLCJsaW1pdCI6NTAwfQ== "
            "https://example.com/a/b/c?utm_source=bench&utm_campaign=sota#fragment "
        ) * 100,
        "emoji_zwj": (
            "👩‍💻👨‍👩‍👧‍👦🧑🏽‍🚀🏳️‍🌈✨🔥🚀 — shipping reliable tokenizers requires boring tests. "
        ) * 120,
        "cjk_dense": (
            "今天天气很好，我们一起测试分词器的性能。東京都市圏の交通データを解析します。"
            "한국어 문장도 함께 포함해서 경계 조건을 확인합니다。"
        ) * 80,
        "rtl": (
            "مرحبا بالعالم هذا اختبار لاتجاه النص. שלום עולם זהו מבחן קצר. "
            "الأرقام ١٢٣٤٥ والرموز ؟! يجب أن تبقى صحيحة. "
        ) * 90,
        "punct": (
            "!!! ??? ... :: == != <= >= -> <- => {[]} () <> // ## @@ ~~ || && %% $$ `` '' \"\"\n"
            "*** --- ___ +++ ::: ;;; ,,, ... ??? !!! ::: === !== >>> <<< [[ ]] {{ }}\n"
        ) * 80,
        "blank_lines": ("\n \n2\n \n0\n\t\n") * 200,
        "whitespace_pathological": (" \t  \n\n\r\n    \tword\t\tword\n" * 220),
        "long_word": ("antidisestablishmentarianism" * 500) + "\n",
        "qwen_chat": (
            "<|im_start|>system\nYou are a concise assistant.<|im_end|>\n"
            "<|im_start|>user\nExplain tokenizer parity in one paragraph.<|im_end|>\n"
            "<|im_start|>assistant\n"
        ) * 80,
    }
    corpora["large"] = corpora["medium"] * 100
    return {name: unicodedata.normalize("NFC", text) for name, text in corpora.items()}


def write_corpora(corpora: dict[str, str], corpus_dir: Path) -> dict[str, Path]:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, text in corpora.items():
        path = corpus_dir / f"{name}.txt"
        path.write_bytes(text.encode("utf-8"))
        paths[name] = path
    return paths


def fnv_ids(ids: list[int]) -> int:
    h = 1469598103934665603
    for token_id in ids:
        for shift in range(0, 32, 8):
            h ^= (token_id >> shift) & 0xFF
            h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def peak_rss_bytes() -> int:
    if resource is None:
        return 0
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform == "darwin":
        return int(usage.ru_maxrss)
    return int(usage.ru_maxrss) * 1024


def measure_encode(
    encode: Callable[[str], list[int]],
    text: str,
    repeats: int,
    target_bytes: int,
    threads: int = 1,
) -> dict[str, float | int]:
    if threads != 1:
        raise ValueError("Python in-process baselines only support single-thread measurement")
    n_bytes = len(text.encode("utf-8"))
    batch = max(1, target_bytes // max(n_bytes, 1))
    for _ in range(min(batch, 8)):
        encode(text)

    durations = []
    token_count = 0
    for _ in range(repeats):
        start = time.perf_counter()
        local_tokens = 0
        for _ in range(batch):
            ids = encode(text)
            local_tokens += len(ids)
        durations.append(time.perf_counter() - start)
        token_count = local_tokens // batch

    median = statistics.median(durations)
    mib_s = (n_bytes * batch / median) / (1024 * 1024)
    per_call_seconds = [duration / batch for duration in durations]
    ids = encode(text)
    return {
        "mib_s": mib_s,
        "tokens": token_count,
        "hash": fnv_ids(ids),
        "batch": batch,
        "repeats": repeats,
        "threads": threads,
        "p50_s": percentile(per_call_seconds, 0.50),
        "p95_s": percentile(per_call_seconds, 0.95),
        "p99_s": percentile(per_call_seconds, 0.99),
        "peak_rss_bytes": peak_rss_bytes(),
    }


def bench_hf_python(model_id: str, corpora: dict[str, str], repeats: int, target_bytes: int) -> dict[str, dict[str, float | int]]:
    from tokenizers import Tokenizer  # type: ignore

    tokenizer = Tokenizer.from_pretrained(model_id)
    results = {}
    for name, text in corpora.items():
        results[name] = measure_encode(lambda t: tokenizer.encode(t).ids, text, repeats, target_bytes)
    return results


def bench_tiktoken_gpt2(corpora: dict[str, str], repeats: int, target_bytes: int) -> dict[str, dict[str, float | int]] | None:
    try:
        import tiktoken  # type: ignore
    except ImportError:
        return None

    encoding = tiktoken.get_encoding("gpt2")
    results = {}
    for name, text in corpora.items():
        results[name] = measure_encode(encoding.encode, text, repeats, target_bytes)
    return results


def download_tokenizer_json(model_id: str) -> Path:
    from huggingface_hub import hf_hub_download  # type: ignore

    return Path(hf_hub_download(model_id, "tokenizer.json"))


def ensure_hf_rust_probe(cache_dir: Path) -> Path | None:
    if shutil.which("cargo") is None:
        return None

    project = cache_dir / "hf_tokenizers_probe"
    src_dir = project / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    (project / "Cargo.toml").write_text(
        textwrap.dedent(
            f"""
            [package]
            name = "hf_tokenizers_probe"
            version = "0.1.0"
            edition = "2021"

            [dependencies]
            libc = "0.2"
            tokenizers = "{HF_RUST_TOKENIZERS_VERSION}"
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (src_dir / "main.rs").write_text(
        textwrap.dedent(
            r'''
            use std::env;
            use std::fs;
            use std::hint::black_box;
            use std::sync::{Arc, Barrier};
            use std::thread;
            use std::time::Instant;
            use tokenizers::Tokenizer;

            fn hash_ids(ids: &[u32]) -> u64 {
                let mut h: u64 = 1469598103934665603;
                for id in ids {
                    for shift in (0..32).step_by(8) {
                        h ^= ((*id >> shift) & 0xff) as u8 as u64;
                        h = h.wrapping_mul(1099511628211);
                    }
                }
                h
            }

            fn percentile(mut values: Vec<f64>, q: f64) -> f64 {
                values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                if values.is_empty() {
                    return 0.0;
                }
                let pos = q * ((values.len() - 1) as f64);
                let lo = pos.floor() as usize;
                let hi = std::cmp::min(lo + 1, values.len() - 1);
                let frac = pos - (lo as f64);
                values[lo] * (1.0 - frac) + values[hi] * frac
            }

            fn peak_rss_bytes() -> u64 {
                unsafe {
                    let mut usage: libc::rusage = std::mem::zeroed();
                    if libc::getrusage(libc::RUSAGE_SELF, &mut usage) != 0 {
                        return 0;
                    }
                    #[cfg(target_os = "macos")]
                    {
                        usage.ru_maxrss as u64
                    }
                    #[cfg(not(target_os = "macos"))]
                    {
                        (usage.ru_maxrss as u64) * 1024
                    }
                }
            }

            fn encode_batch(tokenizer: &Tokenizer, text: &str, batch: usize) -> usize {
                let mut local_tokens = 0usize;
                for _ in 0..batch {
                    let enc = tokenizer.encode(text, false).unwrap();
                    local_tokens += enc.get_ids().len();
                    black_box(enc.get_ids());
                }
                local_tokens
            }

            fn main() {
                let args: Vec<String> = env::args().collect();
                if args.len() < 6 {
                    eprintln!("usage: hf_tokenizers_probe TOKENIZER_JSON REPEATS TARGET_BYTES THREADS CORPUS_FILE...");
                    std::process::exit(2);
                }
                let tokenizer = Arc::new(Tokenizer::from_file(&args[1]).unwrap());
                let repeats: usize = args[2].parse().unwrap();
                let target_bytes: usize = args[3].parse().unwrap();
                let threads: usize = std::cmp::max(1, args[4].parse().unwrap());

                for path in &args[5..] {
                    let text = Arc::new(fs::read_to_string(path).unwrap());
                    let n_bytes = text.as_bytes().len();
                    let service_batch_bytes = std::cmp::max(n_bytes.saturating_mul(threads), 1);
                    let batch = std::cmp::max(1, target_bytes / service_batch_bytes);
                    for _ in 0..std::cmp::min(batch, 8) {
                        let enc = tokenizer.encode(text.as_str(), false).unwrap();
                        black_box(enc.get_ids());
                    }

                    let mut durations = Vec::with_capacity(repeats);
                    let mut token_count = 0usize;
                    for _ in 0..repeats {
                        let start = Instant::now();
                        let local_tokens = if threads == 1 {
                            encode_batch(&tokenizer, text.as_str(), batch)
                        } else {
                            let barrier = Arc::new(Barrier::new(threads));
                            let mut handles = Vec::with_capacity(threads);
                            for _ in 0..threads {
                                let tokenizer = Arc::clone(&tokenizer);
                                let text = Arc::clone(&text);
                                let barrier = Arc::clone(&barrier);
                                handles.push(thread::spawn(move || {
                                    barrier.wait();
                                    encode_batch(&tokenizer, text.as_str(), batch)
                                }));
                            }
                            let mut total = 0usize;
                            for handle in handles {
                                total += handle.join().unwrap();
                            }
                            total
                        };
                        durations.push(start.elapsed().as_secs_f64());
                        token_count = local_tokens / (batch * threads);
                    }

                    let enc = tokenizer.encode(text.as_str(), false).unwrap();
                    let calls_per_repeat = (batch * threads) as f64;
                    let per_call: Vec<f64> = durations.iter().map(|d| d / calls_per_repeat).collect();
                    let mib_s = (n_bytes as f64 * calls_per_repeat) /
                        percentile(durations.clone(), 0.50) / (1024.0 * 1024.0);
                    let name = std::path::Path::new(path).file_stem().unwrap().to_str().unwrap();
                    println!(
                        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                        name,
                        mib_s,
                        token_count,
                        hash_ids(enc.get_ids()),
                        batch,
                        repeats,
                        threads,
                        percentile(per_call.clone(), 0.50),
                        percentile(per_call.clone(), 0.95),
                        percentile(per_call, 0.99),
                        peak_rss_bytes()
                    );
                }
            }
            '''
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    run(["cargo", "build", "--release"], cwd=project)
    return project / "target" / "release" / "hf_tokenizers_probe"


def parse_probe_output(output: str) -> dict[str, dict[str, float | int]]:
    results = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) == 4:
            name, mib_s, tokens, token_hash = parts
            results[name] = {
                "mib_s": float(mib_s),
                "tokens": int(tokens),
                "hash": int(token_hash),
            }
            continue
        if len(parts) != 11:
            raise ValueError(f"unexpected probe output line: {line}")
        name, mib_s, tokens, token_hash, batch, repeats, threads, p50_s, p95_s, p99_s, peak_rss = parts
        results[name] = {
            "mib_s": float(mib_s),
            "tokens": int(tokens),
            "hash": int(token_hash),
            "batch": int(batch),
            "repeats": int(repeats),
            "threads": int(threads),
            "p50_s": float(p50_s),
            "p95_s": float(p95_s),
            "p99_s": float(p99_s),
            "peak_rss_bytes": int(peak_rss),
        }
    return results


def bench_hf_rust(
    probe: Path | None,
    tokenizer_json: Path,
    corpus_paths: dict[str, Path],
    repeats: int,
    target_bytes: int,
    threads: int,
) -> dict[str, dict[str, float | int]] | None:
    if probe is None:
        return None
    cmd = [str(probe), str(tokenizer_json), str(repeats), str(target_bytes), str(threads)]
    cmd.extend(str(path) for path in corpus_paths.values())
    output = subprocess.check_output(cmd, cwd=REPO_ROOT, text=True)
    return parse_probe_output(output)


def ensure_cpp_probe(build_dir: Path) -> Path:
    run([
        "cmake", "-S", str(REPO_ROOT), "-B", str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBPE_BUILD_TESTS=OFF",
        "-DBPE_BUILD_BENCHMARKS=OFF",
    ])
    run(["cmake", "--build", str(build_dir), "--target", "bpe_tokenizer", "--parallel"])
    probe = build_dir / "cpp_encode_probe"
    json_include = build_dir / "_deps" / "json-src" / "include"
    utf8proc_lib = build_dir / "_deps" / "utf8proc-build" / "libutf8proc.a"
    run([
        "c++", "-std=c++20", "-O3", "-DNDEBUG",
        "-I", str(REPO_ROOT / "include"),
        "-I", str(json_include),
        str(REPO_ROOT / "scripts" / "cpp_encode_probe.cpp"),
        str(build_dir / "libbpe_tokenizer.a"),
        str(utf8proc_lib),
        "-o", str(probe),
    ])
    return probe


def bench_cpp(
    probe: Path,
    model_dir: Path,
    corpus_paths: dict[str, Path],
    repeats: int,
    target_bytes: int,
    threads: int,
) -> dict[str, dict[str, float | int]] | None:
    if not model_dir.exists():
        return None
    cmd = [str(probe), str(model_dir), str(repeats), str(target_bytes), str(threads)]
    cmd.extend(str(path) for path in corpus_paths.values())
    output = subprocess.check_output(cmd, cwd=REPO_ROOT, text=True)
    return parse_probe_output(output)


def parity_status(a: dict[str, dict[str, float | int]] | None, b: dict[str, dict[str, float | int]] | None) -> dict[str, object]:
    if a is None or b is None:
        return {"status": "missing_baseline"}
    missing_left = sorted(set(b) - set(a))
    missing_right = sorted(set(a) - set(b))
    mismatches = []
    for name in sorted(set(a) & set(b)):
        if a[name]["tokens"] != b[name]["tokens"] or a[name]["hash"] != b[name]["hash"]:
            mismatches.append({
                "corpus": name,
                "left_tokens": a[name]["tokens"],
                "right_tokens": b[name]["tokens"],
                "left_hash": a[name]["hash"],
                "right_hash": b[name]["hash"],
            })
    status = "pass" if not mismatches and not missing_left and not missing_right else "fail"
    return {
        "status": status,
        "mismatches": mismatches,
        "missing_left": missing_left,
        "missing_right": missing_right,
    }


def throughput_status(
    candidate: dict[str, dict[str, float | int]] | None,
    baselines: dict[str, dict[str, dict[str, float | int]] | None],
) -> dict[str, object]:
    if candidate is None:
        return {"status": "missing_candidate"}
    failures = []
    missing = []
    for baseline_name, baseline in baselines.items():
        if baseline is None:
            missing.append(baseline_name)
            continue
        for corpus, row in baseline.items():
            if corpus not in candidate:
                failures.append({"corpus": corpus, "baseline": baseline_name, "reason": "missing_candidate_corpus"})
                continue
            candidate_mib_s = candidate[corpus]["mib_s"]
            baseline_mib_s = row["mib_s"]
            if candidate_mib_s <= baseline_mib_s:
                failures.append({
                    "corpus": corpus,
                    "baseline": baseline_name,
                    "candidate_mib_s": candidate_mib_s,
                    "baseline_mib_s": baseline_mib_s,
                })
    status = "pass" if not failures and not missing else "fail"
    return {"status": status, "failures": failures, "missing_baselines": missing}


def machine() -> dict[str, str]:
    data = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "cpu_count": str(os.cpu_count() or ""),
    }
    if platform.system() == "Darwin":
        try:
            brand = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
            if brand:
                data["cpu_brand"] = brand
        except Exception:
            pass
    return data


def git_metadata() -> dict[str, object]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        commit = None
    try:
        status = subprocess.check_output(["git", "status", "--short"], cwd=REPO_ROOT, text=True)
        dirty = bool(status.strip())
    except Exception:
        dirty = None
    return {
        "commit": commit,
        "dirty": dirty,
    }


def python_package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ["tokenizers", "huggingface_hub", "tiktoken"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--gpt2-model", default="gpt2")
    parser.add_argument("--cpp-gpt2-dir", default=str(REPO_ROOT / "tests" / "fixtures" / "gpt2"))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--build-dir", default=str(REPO_ROOT / "build-sota"))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--target-bytes", type=int, default=TARGET_SAMPLE_BYTES)
    parser.add_argument("--service-threads", type=int, default=DEFAULT_SERVICE_THREADS)
    parser.add_argument("--skip-rust", action="store_true")
    parser.add_argument("--skip-cpp", action="store_true")
    parser.add_argument("--skip-serving", action="store_true")
    args = parser.parse_args()
    if args.service_threads < 1:
        raise SystemExit("--service-threads must be >= 1")

    cache_dir = Path(args.cache_dir)
    corpus_paths = write_corpora(load_corpora(), cache_dir / "corpora")
    corpora = {name: path.read_bytes().decode("utf-8") for name, path in corpus_paths.items()}

    print(f"Downloading tokenizer files for {args.qwen_model} and {args.gpt2_model}")
    qwen_json = download_tokenizer_json(args.qwen_model)
    gpt2_json = download_tokenizer_json(args.gpt2_model)

    results: dict[str, dict[str, dict[str, float | int]] | None] = {}
    serving_results: dict[str, dict[str, dict[str, float | int]] | None] = {}
    print("\n[HF tokenizers Python: Qwen3]")
    results["hf_python_qwen3"] = bench_hf_python(args.qwen_model, corpora, args.repeats, args.target_bytes)

    print("\n[HF tokenizers Python: GPT-2]")
    results["hf_python_gpt2"] = bench_hf_python(args.gpt2_model, corpora, args.repeats, args.target_bytes)

    print("\n[tiktoken Python: GPT-2]")
    results["tiktoken_gpt2"] = bench_tiktoken_gpt2(corpora, args.repeats, args.target_bytes)

    if args.skip_rust:
        results["hf_rust_qwen3"] = None
        results["hf_rust_gpt2"] = None
        hf_rust_probe = None
    else:
        hf_rust_probe = ensure_hf_rust_probe(cache_dir)
        print("\n[HF tokenizers Rust: Qwen3]")
        results["hf_rust_qwen3"] = bench_hf_rust(hf_rust_probe, qwen_json, corpus_paths, args.repeats, args.target_bytes, 1)
        print("\n[HF tokenizers Rust: GPT-2]")
        results["hf_rust_gpt2"] = bench_hf_rust(hf_rust_probe, gpt2_json, corpus_paths, args.repeats, args.target_bytes, 1)

    if args.skip_cpp:
        results["cpp_gpt2"] = None
        results["cpp_qwen3"] = None
        cpp_probe = None
    else:
        cpp_probe = ensure_cpp_probe(Path(args.build_dir))
        print("\n[C++ tokenizer: GPT-2 fixture]")
        results["cpp_gpt2"] = bench_cpp(cpp_probe, Path(args.cpp_gpt2_dir), corpus_paths, args.repeats, args.target_bytes, 1)
        print("\n[C++ tokenizer: Qwen3 tokenizer.json]")
        results["cpp_qwen3"] = bench_cpp(cpp_probe, qwen_json, corpus_paths, args.repeats, args.target_bytes, 1)

    if args.skip_serving:
        serving_results["cpp_gpt2"] = None
        serving_results["cpp_qwen3"] = None
        serving_results["hf_rust_qwen3"] = None
        serving_results["hf_rust_gpt2"] = None
    else:
        if not args.skip_cpp and cpp_probe is not None:
            print(f"\n[C++ tokenizer serving: GPT-2 fixture, {args.service_threads} threads]")
            serving_results["cpp_gpt2"] = bench_cpp(
                cpp_probe, Path(args.cpp_gpt2_dir), corpus_paths, args.repeats, args.target_bytes, args.service_threads)
            print(f"\n[C++ tokenizer serving: Qwen3 tokenizer.json, {args.service_threads} threads]")
            serving_results["cpp_qwen3"] = bench_cpp(
                cpp_probe, qwen_json, corpus_paths, args.repeats, args.target_bytes, args.service_threads)
        else:
            serving_results["cpp_gpt2"] = None
            serving_results["cpp_qwen3"] = None
        if not args.skip_rust and hf_rust_probe is not None:
            print(f"\n[HF tokenizers Rust serving: Qwen3, {args.service_threads} threads]")
            serving_results["hf_rust_qwen3"] = bench_hf_rust(
                hf_rust_probe, qwen_json, corpus_paths, args.repeats, args.target_bytes, args.service_threads)
            print(f"\n[HF tokenizers Rust serving: GPT-2, {args.service_threads} threads]")
            serving_results["hf_rust_gpt2"] = bench_hf_rust(
                hf_rust_probe, gpt2_json, corpus_paths, args.repeats, args.target_bytes, args.service_threads)
        else:
            serving_results["hf_rust_qwen3"] = None
            serving_results["hf_rust_gpt2"] = None

    parity = {
        "qwen3_hf_rust_vs_python": parity_status(results.get("hf_rust_qwen3"), results.get("hf_python_qwen3")),
        "gpt2_hf_rust_vs_python": parity_status(results.get("hf_rust_gpt2"), results.get("hf_python_gpt2")),
        "gpt2_cpp_vs_hf_python": parity_status(results.get("cpp_gpt2"), results.get("hf_python_gpt2")),
        "gpt2_cpp_vs_tiktoken": parity_status(results.get("cpp_gpt2"), results.get("tiktoken_gpt2")),
        "qwen3_cpp_vs_hf_python": parity_status(results.get("cpp_qwen3"), results.get("hf_python_qwen3")),
        "qwen3_cpp_vs_hf_rust": parity_status(results.get("cpp_qwen3"), results.get("hf_rust_qwen3")),
    }
    throughput = {
        "qwen3_cpp_vs_hf": throughput_status(results.get("cpp_qwen3"), {
            "hf_python_qwen3": results.get("hf_python_qwen3"),
            "hf_rust_qwen3": results.get("hf_rust_qwen3"),
        }),
        "gpt2_cpp_vs_hf_tiktoken": throughput_status(results.get("cpp_gpt2"), {
            "hf_python_gpt2": results.get("hf_python_gpt2"),
            "hf_rust_gpt2": results.get("hf_rust_gpt2"),
            "tiktoken_gpt2": results.get("tiktoken_gpt2"),
        }),
    }
    all_required_parity_passed = all(row["status"] == "pass" for row in parity.values())
    all_required_throughput_passed = all(row["status"] == "pass" for row in throughput.values())
    sota_claim_status = (
        "met_for_recorded_encode_only_nfc_harness"
        if all_required_parity_passed and all_required_throughput_passed
        else "not_met_until_required_parity_and_throughput_gates_pass"
    )

    payload = {
        "scope": {
            "claim": "encode-only tokenizer throughput",
            "qwen_model": args.qwen_model,
            "full_parity_required": True,
            "throughput_leadership_required": True,
            "sota_claim_status": sota_claim_status,
            "normalization_scope": "input corpora are NFC-normalized by this harness; the C++ Qwen path also applies NFC during encode",
            "benchmark_modes": [
                "single-thread encode throughput and per-call latency percentiles",
                "shared-tokenizer multi-thread serving throughput for native C++ and HF Rust probes",
            ],
        },
        "methodology_gates": {
            "1_define_category": "encode-only",
            "2_lock_correctness": "token-id hash parity recorded per corpus",
            "3_real_baselines": "HF tokenizers Qwen3 and GPT-2, Python binding plus Rust crate when cargo is available, plus tiktoken GPT-2 when installed",
            "4_neutral_harness": "same NFC-normalized corpus files, warmup, repeated median throughput",
            "5_serious_corpora": sorted(corpora),
            "6_metrics": "MiB/s, token count, token-id hash, p50/p95/p99 per-call latency, peak RSS, batch size, threads, machine metadata",
            "7_control_hardware": "machine metadata recorded; caller should run on pinned/quiet host for publishable numbers",
            "8_publish_artifacts": "this script, generated Rust probe, C++ probe source, command metadata, git metadata, and JSON output are reproducible artifacts",
            "9_adversarial_review": "parity failures, missing baselines, and normalization scope are explicit in JSON",
            "10_throughput_gate": "C++ throughput must beat each recorded baseline on every corpus",
            "11_serving_measurement": f"shared tokenizer measured with {args.service_threads} request threads unless --skip-serving is used",
        },
        "measurement_notes": {
            "latency_percentiles": "p50/p95/p99 are per encode call, estimated from repeated fixed-size batches; increase --repeats for publishable tail estimates",
            "memory": "peak_rss_bytes is process peak RSS reported by the measured process; Python baselines run in this script process, native probes report their own process RSS",
            "serving": "serving_results use one loaded shared tokenizer and concurrent encode calls; file I/O and model load are outside the timed region",
            "corpus_io": "corpus files are written and read as UTF-8 bytes to avoid platform newline translation",
        },
        "implementation_scope": {
            "cpp_hf_tokenizer_json": "strict ByteLevel BPE subset with Qwen-style pre-tokenization and literal added-token matching",
            "unsupported_hf_features": [
                "byte_fallback",
                "ignore_merges",
                "added-token single_word/lstrip/rstrip/normalized flags",
                "HF pre-tokenizer graphs other than the supported Qwen Split+ByteLevel sequence",
                "general HF normalizer/post-processor pipelines",
            ],
        },
        "machine": machine(),
        "reproducibility": {
            "script": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
            "argv": sys.argv,
            "git": git_metadata(),
            "cache_dir": str(cache_dir),
            "build_dir": str(Path(args.build_dir)),
            "target_bytes": args.target_bytes,
            "repeats": args.repeats,
            "service_threads": args.service_threads,
        },
        "versions": {
            "python_executable": sys.executable,
            "python_packages": python_package_versions(),
            "hf_rust_tokenizers_crate": HF_RUST_TOKENIZERS_VERSION,
            "qwen_tokenizer_json": str(qwen_json),
            "gpt2_tokenizer_json": str(gpt2_json),
        },
        "corpora": {
            name: {
                "bytes": len(text.encode("utf-8")),
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "normalization": "NFC",
            }
            for name, text in corpora.items()
        },
        "results": results,
        "serving_results": serving_results,
        "parity": parity,
        "throughput": throughput,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nWrote {out}")

    for name, table in results.items():
        if not table:
            print(f"{name}: skipped")
            continue
        print(name)
        for corpus, row in table.items():
            p95_us = float(row.get("p95_s", 0.0)) * 1_000_000.0
            print(
                f"  {corpus:22s} {row['mib_s']:9.2f} MiB/s "
                f"tokens={row['tokens']} p95={p95_us:.2f}us hash={row['hash']}"
            )
    print("parity")
    for name, row in parity.items():
        print(f"  {name}: {row['status']}")
    print("throughput")
    for name, row in throughput.items():
        print(f"  {name}: {row['status']}")
    print("serving")
    for name, table in serving_results.items():
        if not table:
            print(f"  {name}: skipped")
            continue
        fastest = max(table.items(), key=lambda item: item[1]["mib_s"])
        print(f"  {name}: {fastest[0]} {fastest[1]['mib_s']:.2f} MiB/s threads={fastest[1]['threads']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
