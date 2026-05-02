#!/usr/bin/env python3
"""
Encode-only SOTA proof harness.

This script is intentionally stricter than the library benchmarks:
  - it records the exact claim scope,
  - validates token-id parity where an apples-to-apples model is available,
  - measures Hugging Face tokenizers for the Qwen3 tokenizer through both the
    Rust crate and the Python binding when possible,
  - writes raw machine, corpus, parity, and timing data to JSON.

The current C++ library supports GPT-2-style byte-level BPE models. Qwen3 uses
Hugging Face tokenizer.json semantics with NFC normalization and a Qwen-specific
regex pre-tokenizer, so this harness marks the C++ Qwen3 parity gate as
unsupported until that model format is implemented.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "runs" / "sota_encode_report.json"
DEFAULT_CACHE = REPO_ROOT / ".cache" / "sota_encode"
TARGET_SAMPLE_BYTES = 8 * 1024 * 1024


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
        "punct": (
            "!!! ??? ... :: == != <= >= -> <- => {[]} () <> // ## @@ ~~ || && %% $$ `` '' \"\"\n"
            "*** --- ___ +++ ::: ;;; ,,, ... ??? !!! ::: === !== >>> <<< [[ ]] {{ }}\n"
        ) * 80,
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
        path.write_text(text, encoding="utf-8")
        paths[name] = path
    return paths


def fnv_ids(ids: list[int]) -> int:
    h = 1469598103934665603
    for token_id in ids:
        for shift in range(0, 32, 8):
            h ^= (token_id >> shift) & 0xFF
            h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def median_mib_s(encode: Callable[[str], list[int]], text: str, repeats: int, target_bytes: int) -> tuple[float, int, int]:
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
    ids = encode(text)
    return mib_s, token_count, fnv_ids(ids)


def bench_hf_python(model_id: str, corpora: dict[str, str], repeats: int, target_bytes: int) -> dict[str, dict[str, float | int]]:
    from tokenizers import Tokenizer  # type: ignore

    tokenizer = Tokenizer.from_pretrained(model_id)
    results = {}
    for name, text in corpora.items():
        mib_s, tokens, token_hash = median_mib_s(lambda t: tokenizer.encode(t).ids, text, repeats, target_bytes)
        results[name] = {"mib_s": mib_s, "tokens": tokens, "hash": token_hash}
    return results


def bench_tiktoken_gpt2(corpora: dict[str, str], repeats: int, target_bytes: int) -> dict[str, dict[str, float | int]] | None:
    try:
        import tiktoken  # type: ignore
    except ImportError:
        return None

    encoding = tiktoken.get_encoding("gpt2")
    results = {}
    for name, text in corpora.items():
        mib_s, tokens, token_hash = median_mib_s(encoding.encode, text, repeats, target_bytes)
        results[name] = {"mib_s": mib_s, "tokens": tokens, "hash": token_hash}
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
            """
            [package]
            name = "hf_tokenizers_probe"
            version = "0.1.0"
            edition = "2021"

            [dependencies]
            tokenizers = "0.23.1"
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

            fn median(mut values: Vec<f64>) -> f64 {
                values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                values[values.len() / 2]
            }

            fn main() {
                let args: Vec<String> = env::args().collect();
                if args.len() < 5 {
                    eprintln!("usage: hf_tokenizers_probe TOKENIZER_JSON REPEATS TARGET_BYTES CORPUS_FILE...");
                    std::process::exit(2);
                }
                let tokenizer = Tokenizer::from_file(&args[1]).unwrap();
                let repeats: usize = args[2].parse().unwrap();
                let target_bytes: usize = args[3].parse().unwrap();

                for path in &args[4..] {
                    let text = fs::read_to_string(path).unwrap();
                    let n_bytes = text.as_bytes().len();
                    let batch = std::cmp::max(1, target_bytes / std::cmp::max(n_bytes, 1));
                    for _ in 0..std::cmp::min(batch, 8) {
                        let enc = tokenizer.encode(text.as_str(), false).unwrap();
                        black_box(enc.get_ids());
                    }

                    let mut durations = Vec::with_capacity(repeats);
                    let mut token_count = 0usize;
                    for _ in 0..repeats {
                        let start = Instant::now();
                        let mut local_tokens = 0usize;
                        for _ in 0..batch {
                            let enc = tokenizer.encode(text.as_str(), false).unwrap();
                            local_tokens += enc.get_ids().len();
                            black_box(enc.get_ids());
                        }
                        durations.push(start.elapsed().as_secs_f64());
                        token_count = local_tokens / batch;
                    }

                    let enc = tokenizer.encode(text.as_str(), false).unwrap();
                    let mib_s = (n_bytes as f64 * batch as f64) / median(durations) / (1024.0 * 1024.0);
                    let name = std::path::Path::new(path).file_stem().unwrap().to_str().unwrap();
                    println!("{}\t{}\t{}\t{}", name, mib_s, token_count, hash_ids(enc.get_ids()));
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
        name, mib_s, tokens, token_hash = line.split("\t")
        results[name] = {"mib_s": float(mib_s), "tokens": int(tokens), "hash": int(token_hash)}
    return results


def bench_hf_rust(
    tokenizer_json: Path,
    corpus_paths: dict[str, Path],
    repeats: int,
    target_bytes: int,
    cache_dir: Path,
) -> dict[str, dict[str, float | int]] | None:
    probe = ensure_hf_rust_probe(cache_dir)
    if probe is None:
        return None
    cmd = [str(probe), str(tokenizer_json), str(repeats), str(target_bytes)]
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
    run([
        "c++", "-std=c++20", "-O3", "-DNDEBUG",
        "-I", str(REPO_ROOT / "include"),
        "-I", str(json_include),
        str(REPO_ROOT / "scripts" / "cpp_encode_probe.cpp"),
        str(build_dir / "libbpe_tokenizer.a"),
        "-o", str(probe),
    ])
    return probe


def bench_cpp(
    model_dir: Path,
    corpus_paths: dict[str, Path],
    repeats: int,
    target_bytes: int,
    build_dir: Path,
) -> dict[str, dict[str, float | int]] | None:
    if not model_dir.exists():
        return None
    probe = ensure_cpp_probe(build_dir)
    cmd = [str(probe), str(model_dir), str(repeats), str(target_bytes)]
    cmd.extend(str(path) for path in corpus_paths.values())
    output = subprocess.check_output(cmd, cwd=REPO_ROOT, text=True)
    return parse_probe_output(output)


def parity_status(a: dict[str, dict[str, float | int]] | None, b: dict[str, dict[str, float | int]] | None) -> dict[str, object]:
    if a is None or b is None:
        return {"status": "missing_baseline"}
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
    return {"status": "pass" if not mismatches else "fail", "mismatches": mismatches}


def machine() -> dict[str, str]:
    data = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
    }
    if platform.system() == "Darwin":
        try:
            brand = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
            if brand:
                data["cpu_brand"] = brand
        except Exception:
            pass
    return data


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
    parser.add_argument("--skip-rust", action="store_true")
    parser.add_argument("--skip-cpp", action="store_true")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    corpus_paths = write_corpora(load_corpora(), cache_dir / "corpora")
    corpora = {name: path.read_text(encoding="utf-8") for name, path in corpus_paths.items()}

    print(f"Downloading tokenizer files for {args.qwen_model} and {args.gpt2_model}")
    qwen_json = download_tokenizer_json(args.qwen_model)
    gpt2_json = download_tokenizer_json(args.gpt2_model)

    results: dict[str, dict[str, dict[str, float | int]] | None] = {}
    print("\n[HF tokenizers Python: Qwen3]")
    results["hf_python_qwen3"] = bench_hf_python(args.qwen_model, corpora, args.repeats, args.target_bytes)

    print("\n[HF tokenizers Python: GPT-2]")
    results["hf_python_gpt2"] = bench_hf_python(args.gpt2_model, corpora, args.repeats, args.target_bytes)

    print("\n[tiktoken Python: GPT-2]")
    results["tiktoken_gpt2"] = bench_tiktoken_gpt2(corpora, args.repeats, args.target_bytes)

    if args.skip_rust:
        results["hf_rust_qwen3"] = None
        results["hf_rust_gpt2"] = None
    else:
        print("\n[HF tokenizers Rust: Qwen3]")
        results["hf_rust_qwen3"] = bench_hf_rust(qwen_json, corpus_paths, args.repeats, args.target_bytes, cache_dir)
        print("\n[HF tokenizers Rust: GPT-2]")
        results["hf_rust_gpt2"] = bench_hf_rust(gpt2_json, corpus_paths, args.repeats, args.target_bytes, cache_dir)

    if args.skip_cpp:
        results["cpp_gpt2"] = None
    else:
        print("\n[C++ tokenizer: GPT-2 fixture]")
        results["cpp_gpt2"] = bench_cpp(Path(args.cpp_gpt2_dir), corpus_paths, args.repeats, args.target_bytes, Path(args.build_dir))

    parity = {
        "qwen3_hf_rust_vs_python": parity_status(results.get("hf_rust_qwen3"), results.get("hf_python_qwen3")),
        "gpt2_hf_rust_vs_python": parity_status(results.get("hf_rust_gpt2"), results.get("hf_python_gpt2")),
        "gpt2_cpp_vs_hf_python": parity_status(results.get("cpp_gpt2"), results.get("hf_python_gpt2")),
        "gpt2_cpp_vs_tiktoken": parity_status(results.get("cpp_gpt2"), results.get("tiktoken_gpt2")),
        "qwen3_cpp_vs_hf": {
            "status": "unsupported",
            "reason": (
                "The C++ library currently loads native/GPT-2-style vocab+merges. "
                "Qwen3 needs tokenizer.json support, NFC normalization, ByteLevel vocabulary decoding, "
                "and the Qwen regex pre-tokenizer before an apples-to-apples C++ Qwen3 SOTA claim is possible."
            ),
        },
    }

    payload = {
        "scope": {
            "claim": "encode-only tokenizer throughput",
            "qwen_model": args.qwen_model,
            "full_parity_required": True,
            "sota_claim_status": "not_met_until_qwen3_cpp_parity_and_native_baselines_pass",
        },
        "methodology_gates": {
            "1_define_category": "encode-only",
            "2_lock_correctness": "token-id hash parity recorded per corpus",
            "3_real_baselines": "HF tokenizers Qwen3 and GPT-2, Python binding plus Rust crate when cargo is available, plus tiktoken GPT-2 when installed",
            "4_neutral_harness": "same NFC-normalized corpus files, warmup, repeated median throughput",
            "5_serious_corpora": sorted(corpora),
            "6_metrics": "MiB/s, token count, token-id hash, machine metadata",
            "7_control_hardware": "machine metadata recorded; caller should run on pinned/quiet host for publishable numbers",
            "8_publish_artifacts": "this script, C++ probe source, and JSON output are reproducible artifacts",
            "9_adversarial_review": "parity failures and unsupported Qwen3 C++ gate are explicit in JSON",
        },
        "machine": machine(),
        "versions": {
            "python_executable": sys.executable,
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
        "parity": parity,
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
            print(f"  {corpus:10s} {row['mib_s']:9.2f} MiB/s tokens={row['tokens']} hash={row['hash']}")
    print("parity")
    for name, row in parity.items():
        print(f"  {name}: {row['status']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
