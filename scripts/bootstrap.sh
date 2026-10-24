#!/usr/bin/env bash
# Fetch llama.cpp, sherpa-onnx Android JNI, and Kotlin API for Gradle modules :lib and :sherpa.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SHERPA_VERSION="${SHERPA_VERSION:-v1.13.8}"
LLAMA_DIR="$ROOT/third_party/llama.cpp"
ANDROID_DIR="$ROOT/android"
SHERPA_MODULE="$ANDROID_DIR/sherpa"

echo "==> llama.cpp"
if [[ ! -d "$LLAMA_DIR/.git" ]]; then
  mkdir -p third_party
  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
else
  echo "    already cloned"
fi

echo "==> sherpa-onnx Android libs ($SHERPA_VERSION)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
ARCHIVE="sherpa-onnx-${SHERPA_VERSION}-android.tar.bz2"
URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/${SHERPA_VERSION}/${ARCHIVE}"
curl -fsSL "$URL" -o "$TMP/$ARCHIVE"
tar -xjf "$TMP/$ARCHIVE" -C "$TMP"

JNI_DEST="$SHERPA_MODULE/src/main/jniLibs/arm64-v8a"
mkdir -p "$JNI_DEST"
cp -v "$TMP"/jniLibs/arm64-v8a/* "$JNI_DEST/"

echo "==> sherpa-onnx Kotlin API"
KT_DEST="$SHERPA_MODULE/src/main/java/com/k2fsa/sherpa/onnx"
rm -rf "$SHERPA_MODULE/src/main/java"
mkdir -p "$KT_DEST"
SHERPA_SRC="$ROOT/third_party/sherpa-onnx"
if [[ ! -d "$SHERPA_SRC/.git" ]]; then
  git clone --depth 1 --branch "$SHERPA_VERSION" https://github.com/k2-fsa/sherpa-onnx.git "$SHERPA_SRC" 2>/dev/null \
    || git clone --depth 1 https://github.com/k2-fsa/sherpa-onnx.git "$SHERPA_SRC"
fi
cp -v "$SHERPA_SRC"/sherpa-onnx/kotlin-api/*.kt "$KT_DEST/"

echo "==> Gradle wrapper (from llama.android)"
if [[ ! -x "$ANDROID_DIR/gradlew" ]]; then
  mkdir -p "$ANDROID_DIR/gradle/wrapper"
  cp "$LLAMA_DIR/examples/llama.android/gradle/wrapper/"* "$ANDROID_DIR/gradle/wrapper/"
  cp "$LLAMA_DIR/examples/llama.android/gradlew" "$ANDROID_DIR/gradlew"
  if [[ -f "$LLAMA_DIR/examples/llama.android/gradlew.bat" ]]; then
    cp "$LLAMA_DIR/examples/llama.android/gradlew.bat" "$ANDROID_DIR/gradlew.bat"
  fi
  chmod +x "$ANDROID_DIR/gradlew"
fi

echo "bootstrap complete. Open android/ in Android Studio."
