#!/usr/bin/env python3
"""Download Gemma 4 GGUF weights and optional speech models into ./models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

E2B_REPO = "unsloth/gemma-4-E2B-it-GGUF"
E4B_REPO = "unsloth/gemma-4-E4B-it-GGUF"
E2B_FILE = "gemma-4-E2B-it-Q4_K_M.gguf"
E4B_FILE = "gemma-4-E4B-it-Q4_K_M.gguf"
E2B_MMPROJ_REPO = "ggml-org/gemma-4-E2B-it-GGUF"
E4B_MMPROJ_REPO = "ggml-org/gemma-4-E4B-it-GGUF"
E2B_MMPROJ = "mmproj-gemma-4-E2B-it-Q8_0.gguf"
E4B_MMPROJ = "mmproj-gemma-4-E4B-it-Q8_0.gguf"


def fetch(repo: str, filename: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / filename
    if target.is_file() and target.stat().st_size > 0:
        print(f"skip (exists): {target}", flush=True)
        return
    print(f"downloading {repo}/{filename} -> {dest}", flush=True)
    hf_hub_download(repo_id=repo, filename=filename, local_dir=str(dest))
    print(f"ready: {target}", flush=True)


def ensure_gemma4() -> None:
    llm = MODELS / "llm"
    mm = MODELS / "mmproj"
    fetch(E2B_REPO, E2B_FILE, llm)
    fetch(E4B_REPO, E4B_FILE, llm)
    fetch(E2B_MMPROJ_REPO, E2B_MMPROJ, mm)
    fetch(E4B_MMPROJ_REPO, E4B_MMPROJ, mm)


def ensure_sense_voice() -> None:
    dest = MODELS / "asr" / "sense-voice"
    repo = "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09"
    for fname in ("model.int8.onnx", "tokens.txt"):
        fetch(repo, fname, dest)


def ensure_kokoro() -> None:
    dest = MODELS / "tts" / "kokoro-multi-lang-v1_0"
    if (dest / "model.onnx").is_file():
        print(f"skip tts (exists): {dest}", flush=True)
        return
    dest.mkdir(parents=True, exist_ok=True)
    print("downloading kokoro-multi-lang-v1_0 snapshot …", flush=True)
    snapshot_download(repo_id="csukuangfj/kokoro-multi-lang-v1_0", local_dir=str(dest))
    print(f"tts -> {dest}", flush=True)


PROFILES = {
    "gemma4": (ensure_gemma4,),
    "gemma4-voice": (ensure_gemma4, ensure_sense_voice, ensure_kokoro),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES.keys()), default="gemma4")
    args = parser.parse_args()
    MODELS.mkdir(parents=True, exist_ok=True)
    for step in PROFILES[args.profile]:
        step()
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
