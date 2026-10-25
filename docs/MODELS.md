# Model files

## LLM — Gemma 4 E2B and E4B

| File | Size | Source |
|------|------|--------|
| `llm/gemma-4-E2B-it-Q4_K_M.gguf` | ~3.1 GB | [unsloth/gemma-4-E2B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF) |
| `llm/gemma-4-E4B-it-Q4_K_M.gguf` | ~5.0 GB | [unsloth/gemma-4-E4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF) |
| `mmproj/mmproj-gemma-4-E2B-it-Q8_0.gguf` | ~0.56 GB | [ggml-org/gemma-4-E2B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF) |
| `mmproj/mmproj-gemma-4-E4B-it-Q8_0.gguf` | ~0.56 GB | [ggml-org/gemma-4-E4B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF) |

The app settings switch between E2B and E4B, and between **SenseVoice + Kokoro** and **native mmproj** (audio clip saved on device; decode with `scripts/test_gemma4.sh --audio`).

**Faster / smaller:** `google/gemma-3-1b-it` GGUF Q4_K_M (~0.8 GB).

## STT — SenseVoice (default)

Folder: `models/asr/sense-voice/`

| File | Notes |
|------|--------|
| `model.int8.onnx` | From [sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models) |
| `tokens.txt` | Same bundle |

Hugging Face mirror: `pengzhendong/sherpa-onnx-sense-voice-zh-en-ja-ko-yue` (int8 files).

## STT — Whisper tiny.en (optional)

Folder: `models/asr/whisper-tiny.en/`

| File |
|------|
| `tiny.en-encoder.int8.onnx` |
| `tiny.en-decoder.int8.onnx` |
| `tiny.en-tokens.txt` |

See [sherpa Whisper tiny.en docs](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/whisper/tiny.en.html).

## TTS — Kokoro multi-lang v1.0

Folder: `models/tts/kokoro-multi-lang-v1_0/`

Typical contents (from sherpa TTS model pack):

- `model.onnx`
- `voices.bin`
- `tokens.txt`
- `lexicon-us-en.txt`, `lexicon-zh.txt` (and related FSTs if bundled)

See [Kokoro pretrained models](https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/kokoro.html).

Speaker id **0–52** (English voices often in low ids; try `af_sarah` mapping in Kokoro docs → numeric sid in app settings).

## Detector — YOLO26n (walk to object)

Folder: `models/litert/` (copied to `files/models/litert/` on the phone)

| File | Size | Notes |
|------|------|--------|
| `yolo26n_w8a32.tflite` | ~2.9 MB | Ultralytics YOLO26 nano, 8-bit weights, 80 COCO classes, 640 input. The app tiles a 1280×960 frame. No `robot` or `can` class. |

## Total storage budget (OnePlus 12)

| Profile | Approx. |
|---------|---------|
| Gemma 2 Q4 + SenseVoice + Kokoro | ~4.5 GB |
| Gemma 3 1B + Whisper tiny + Kokoro | ~2.5 GB |

Keep models on **internal** storage; SD cards add latency and can cause load failures.
