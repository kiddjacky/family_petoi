# Family Petoi — Gemma Voice

On-device Android app (`com.gemmaproj.voicechat`) that talks with **Gemma 4** on the phone and walks a **Petoi robot dog** (Quaddle / Bittle) to objects you name. Built and tested on a **OnePlus 12** (Snapdragon 8 Gen 3, 16 GB RAM). Everything after the first model copy is offline. No cloud API keys.

| Piece | What runs on the phone | Role |
|-------|------------------------|------|
| Chat + orders | **Gemma 4 E2B** LiteRT GPU (`gemma-4-E2B-it-gpu.litertlm`) | Answers you, reads walk orders, writes a short play tip |
| Hear | **SenseVoice** int8 (sherpa-onnx) | English and Chinese speech |
| Speak | **Kokoro** multi-lang (sherpa-onnx) | Phone speaker; the robot has none |
| See | **YOLO26n** LiteRT (`yolo26n_w8a32.tflite`) | Finds objects and the robot in the camera |
| Walk | **Petoi BLE** Nordic UART | Sends `kwkF` / `kwkL` / `kwkR` / `kbalance` / `khi` |

The last committed baseline was chat + a BLE walk pad. This version adds voice “go to …” walking, YOLO tracking, auto-reconnect, and Gemma’s arrival greeting.

## What you can do

**Hold to talk** on the camera screen (English or Chinese):

- “go to the water bottle” / “走到那个红色的杯子那里” — Gemma names a YOLO class (and a color if you said one). The dog walks there.
- “go to the yellow can” — both the can and the water bottle are `bottle` to YOLO; the spoken color picks the yellow one.
- “now the cup” / “去找那只小熊” — switches target mid-walk.
- “stop” / “停” — stands still, no Gemma wait.

On the way, Gemma answers in the language you used. When the boxes touch, the phone says **Hello!** plus a short parent–child game for that object, and the robot waves (`khi`).

**Buttons:** chips for bottle / cup / ball / bowl / teddy, **Go to …** / **Stop**, **Restart** (clears the walk and target; the BLE link stays up), **Live**, **Find**. On the Dog tab, **Switch robot** forgets the saved name and connects to the nearest Petoi.

## How the walk works

1. YOLO26n runs on the 1280×960 camera frame, tiled (whole frame plus overlapping 640 px tiles), confidence 0.12, up to 32 boxes. NMS is per class so a “person” box on the robot does not delete a “motorcycle” box on the same spot.
2. The **target** is the box with that COCO class. A spoken color samples the middle of each same-class box (HSV) and keeps the one that matches. Once chosen, identity stays locked; each look snaps the box to the newest match so a moved object is followed.
3. The **robot** has no COCO class (YOLO calls it horse, person, motorcycle, dog…). A short probe step (`kwkF 2`) finds which remaining small box moved. After that it is followed by position and size, not by label.
4. Heading is measured from that movement. The loop is probe forward → turn toward the target → probe forward until the boxes touch.
5. If YOLO misses a frame, the last boxes stay and the dog keeps moving. If Bluetooth drops, the walk pauses and resumes after reconnect.

Only one BLE client can hold the robot. Close the Petoi phone app. The first robot this app connects to is remembered (`shared_prefs/dog.xml`); later scans ignore the others until you tap **Switch robot**.

## Limits we already hit

- YOLO26n is the nano 8-bit COCO model. It has no `robot` or `can` class. The yellow can scores as `bottle` (~0.3–0.5); robots flicker between labels. Locking + color + motion ID paper over this; they do not fix the model.
- After a wave the robot can stop advertising for about two minutes (power on the robot). The app retries every few seconds.
- Gemma 4 image answers take 6–8 s per photo, so the walk uses YOLO, not Gemma, for boxes.

## Quick start

### 1. Bootstrap native dependencies (Mac/Linux)

```bash
cd /Users/robotdepot/work/gemma_proj
./scripts/bootstrap.sh
```

Clones `llama.cpp`, downloads sherpa-onnx Android JNI, and copies the Kotlin API into `:sherpa`.

### 2. Models on the phone

Use the project venv (do not re-download GGUFs, LiteRT files, or EmbeddingGemma unless you are missing them):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r scripts/requirements.txt
python scripts/download_models.py --profile oneplus12
```

Copy into the app **internal** folder `files/models/` (`ModelPaths.kt`). Easiest: Android Studio Device File Explorer → `data/data/com.gemmaproj.voicechat/files/models/`. Or:

```bash
adb push models /data/local/tmp/gemma-models
adb shell run-as com.gemmaproj.voicechat cp -r /data/local/tmp/gemma-models files/models
```

Needed for this walk build, besides chat/voice:

| Path under `files/models/` | Use |
|----------------------------|-----|
| `litert/gemma-4-E2B-it-gpu.litertlm` | Gemma orders, replies, play tips |
| `litert/yolo26n_w8a32.tflite` | Object + robot boxes |
| SenseVoice + `tts/kokoro-multi-lang-v1_0/` | Hear / speak |

See [docs/MODELS.md](docs/MODELS.md). Gemma weights need [Google’s Gemma Terms of Use](https://ai.google.dev/gemma/terms).

### 3. Build and install

Open **`android/`** in Android Studio (Ladybug or newer), NDK **29.0.13113456**, or:

```bash
export JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home"
cd android
# wrapper lives under ~/.gradle if ./gradlew is missing
./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

First native build of `:lib` (llama.cpp) can take 15–30 minutes.

App logs do not reach logcat on the OnePlus 12. Read files instead:

```bash
adb shell "run-as com.gemmaproj.voicechat cat files/approach-log.txt"
adb shell "run-as com.gemmaproj.voicechat cat files/dog-log.txt"
```

Launch extras (force-stop first): `walkProbe` (`'|'`-separated orders, parse only, no motion), `dogAsk`, `dogPad`.

## Project layout

```
gemma_proj/
  android/app/.../engine/
    ApproachPilot.kt   # walk decisions
    YoloDetector.kt    # tiled YOLO26n
    WalkOrders.kt      # Gemma JSON → go / stop / other
    ColorMatch.kt      # spoken color among same-class boxes
    ObjectPlay.kt      # reply + play tip
    DogLink.kt         # BLE keep-alive, remembered robot
    PetoiBle.kt        # Nordic UART
  android/app/.../ui/
    SceneCamera.kt     # camera, chips, Restart
    ChatViewModel.kt   # orders, walk loop, greeting
    DogScreen.kt       # walk pad + Switch robot
  scripts/             # bootstrap, model download
  models/              # gitignored
  docs/ARCHITECTURE.md
  docs/MODELS.md
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| “No Petoi robot is advertising” | Power the dog, close the Petoi app, wait if it just waved. Tap **Switch robot** if the saved name is the old one. |
| Connected to the wrong dog | **Switch robot** on the Dog tab, then turn only the one you want on. |
| Walks to the water bottle instead of the can | Say the color: “go to the yellow can”. |
| Robot box flickers / “dog unseen” | YOLO has no robot class; the walk follows the last box and the one that moved. Keep the phone steady. |
| Gemma paused / buttons only | EmbeddingGemma and Gemma 4 both loaded; if memory is short, Gemma unloads. Restart the app. |
| STT empty | Mic permission; SenseVoice under `models/asr/...`; 16 kHz mono. |
| TTS silent | Kokoro under `models/tts/kokoro-multi-lang-v1_0/`. |
| Slow first token | Keep E2B GPU; close background apps. Do not switch chat to FP16 or turn on speculative decoding. |

## License

Application code: MIT. Third-party models and libraries follow their own licenses (Gemma Terms, sherpa-onnx Apache 2.0, llama.cpp MIT, Ultralytics YOLO).
