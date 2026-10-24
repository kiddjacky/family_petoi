# Gemma Voice Chat (Android)

On-device voice and text chat for **OnePlus 12** (Snapdragon 8 Gen 3, 16 GB RAM) using:

| Component | Stack | Default model |
|-----------|--------|----------------|
| LLM | [llama.cpp](https://github.com/ggml-org/llama.cpp) Android binding (KleidiAI on arm64) | **Gemma 4 E2B or E4B** `Q4_K_M` |
| STT | [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) | **SenseVoice** int8 (multilingual) or **Whisper** tiny.en |
| TTS | sherpa-onnx Kokoro | **kokoro-multi-lang-v1_0** |

Everything runs offline after models are copied to the phone. No cloud API keys.

## Why these choices on OnePlus 12

- **Gemma 2 2B Q4_K_M**: Good quality/speed balance; expect roughly **8–20 tok/s** on SD 8 Gen 3 (varies with context length and thermal state).
- **SenseVoice int8**: Strong zh/en/ja/ko/yue ASR in one small model; good for mixed-language speech.
- **Whisper tiny.en**: Lighter English-only option if you do not need SenseVoice languages.
- **Kokoro**: Fast non-autoregressive TTS; multiple English voices via speaker id.

Alternatives that also fit 16 GB RAM: **Gemma 3 1B** (faster, smaller), **Qwen2.5-1.5B-Instruct** (multilingual chat).

## Quick start

### 1. Bootstrap native dependencies (Mac/Linux)

```bash
cd /Users/robotdepot/work/gemma_proj
./scripts/bootstrap.sh
```

This clones `llama.cpp`, downloads sherpa-onnx Android JNI libraries, and copies the Kotlin API into `:sherpa`.

### 2. Download models (optional on desktop, or on phone)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r scripts/requirements.txt
python scripts/download_models.py --profile oneplus12
```

Models land in `models/` (~4–5 GB total). Copy that folder to the phone:

`Android/data/com.gemmaproj.voicechat/files/models/`

Or push from your computer after building the tree above:

Models must live in the app **internal** folder `files/models/` (see `ModelPaths.kt`). Easiest: Android Studio **Device File Explorer** → `data/data/com.gemmaproj.voicechat/files/models/`. Or:

```bash
adb push models /data/local/tmp/gemma-models
adb shell run-as com.gemmaproj.voicechat cp -r /data/local/tmp/gemma-models files/models
```

See [docs/MODELS.md](docs/MODELS.md) for exact filenames.

### 3. Build and install

Open **`android/`** in Android Studio (Ladybug or newer), install NDK **29.0.13113456** (same as upstream llama.android), sync Gradle, run on device.

```bash
cd android
./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

First native build of `:lib` (llama.cpp) can take **15–30 minutes**; later builds are incremental.

### 4. Accept Gemma license

Gemma weights require accepting [Google’s Gemma Terms of Use](https://ai.google.dev/gemma/terms) on Hugging Face before download.

## App usage

- **Text mode**: Type messages; replies stream from Gemma.
- **Voice mode**: Hold the microphone button, speak, release; STT → Gemma → Kokoro playback.
- **Settings**: Switch STT engine (SenseVoice / Whisper), TTS on/off, system prompt, max tokens.

## Project layout

```
gemma_proj/
  android/           # Gemma Voice Chat app (:app), sherpa (:sherpa), llama (:lib)
  scripts/           # bootstrap, model download
  third_party/       # llama.cpp (created by bootstrap)
  models/            # downloaded GGUF + ONNX (gitignored)
  docs/ARCHITECTURE.md
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Slow first token | Use Q4_K_M not Q8; reduce `context` to 2048; close background apps |
| OOM on load | Use Gemma 3 1B or Q3_K_M; ensure models are on internal storage |
| STT empty | Check `models/asr/...` paths; 16 kHz mono PCM; grant mic permission |
| TTS silent | Install Kokoro assets under `models/tts/kokoro-multi-lang-v1_0/`; try speaker id `0` |
| Build fails on `:lib` | Match NDK version in Android Studio SDK Manager |

## License

Application code: MIT. Third-party models and libraries follow their respective licenses (Gemma Terms, sherpa-onnx Apache 2.0, llama.cpp MIT).
