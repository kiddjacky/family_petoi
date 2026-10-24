package com.gemmaproj.voicechat.data

import android.content.Context
import com.gemmaproj.voicechat.domain.GemmaVariant
import java.io.File

/**
 * On-device layout under [Context.getFilesDir]/models/.
 * Desktop downloads live in the repo `models/` tree with the same relative names.
 */
class ModelPaths(private val root: File) {

    val llmDir: File get() = root.resolve("llm")
    val mmprojDir: File get() = root.resolve("mmproj")
    val asrDir: File get() = root.resolve("asr")
    val ttsDir: File get() = root.resolve("tts")
    val clipsDir: File get() = root.resolve("clips")

    fun llmFile(variant: GemmaVariant): File = when (variant) {
        GemmaVariant.E2B -> llmDir.resolve("gemma-4-E2B-it-Q4_K_M.gguf")
        GemmaVariant.E4B -> llmDir.resolve("gemma-4-E4B-it-Q4_K_M.gguf")
    }

    fun mmprojFile(variant: GemmaVariant): File = when (variant) {
        GemmaVariant.E2B -> mmprojDir.resolve("mmproj-gemma-4-E2B-it-Q8_0.gguf")
        GemmaVariant.E4B -> mmprojDir.resolve("mmproj-gemma-4-E4B-it-Q8_0.gguf")
    }

    val senseVoiceDir: File get() = asrDir.resolve("sense-voice")
    val whisperTinyDir: File get() = asrDir.resolve("whisper-tiny.en")
    val kokoroDir: File get() = ttsDir.resolve("kokoro-multi-lang-v1_0")

    fun llmReady(variant: GemmaVariant): Boolean = llmFile(variant).isFile

    fun mmprojReady(variant: GemmaVariant): Boolean = mmprojFile(variant).isFile

    fun senseVoiceReady(): Boolean =
        senseVoiceDir.resolve("model.int8.onnx").isFile &&
            senseVoiceDir.resolve("tokens.txt").isFile

    fun whisperReady(): Boolean =
        whisperTinyDir.resolve("tiny.en-encoder.int8.onnx").isFile &&
            whisperTinyDir.resolve("tiny.en-decoder.int8.onnx").isFile &&
            whisperTinyDir.resolve("tiny.en-tokens.txt").isFile

    fun kokoroReady(): Boolean =
        kokoroDir.resolve("model.onnx").isFile &&
            kokoroDir.resolve("voices.bin").isFile &&
            kokoroDir.resolve("tokens.txt").isFile &&
            kokoroDir.resolve("espeak-ng-data").isDirectory &&
            kokoroDir.resolve("lexicon-us-en.txt").isFile

    fun litertFile(variant: GemmaVariant): File = when (variant) {
        GemmaVariant.E2B -> root.resolve("litert/gemma-4-E2B-it-gpu.litertlm")
        GemmaVariant.E4B -> root.resolve("litert/gemma-4-E4B-it-gpu.litertlm")
    }

    /**
     * Text, image, and audio bundle. The GPU-only file has no audio encoder,
     * so native speech has to load this one and run the encoder on CPU.
     */
    fun litertMultimodal(): File = root.resolve("litert/gemma-4-E2B-it.litertlm")

    /**
     * EmbeddingGemma 2 text-vision. Maps a phrase and an image crop into one
     * vector so a grid of crops can say where the dog and the target are.
     */
    fun embeddingVision(): File =
        root.resolve("litert/embeddinggemma-2-text-vision-440m.litertlm")

    /** YOLO26n detection. Boxes only; EmbeddingGemma decides which box is which. */
    fun yoloFile(): File = root.resolve("litert/yolo26n_w8a32.tflite")

    /** Optional `host` or `host:port` for the Petoi BiBoard WebSocket. */
    fun robotHostFile(): File = root.resolve("robot-host.txt")

    /** Gemma 3 1B Q4_K_M for the llama.cpp side of the NPU comparison. */
    fun gemma3Gguf(): File = llmDir.resolve("google_gemma-3-1b-it-Q4_K_M.gguf")

    /** Gemma 3 1B compiled for the Snapdragon 8 Gen 3 Hexagon NPU (SM8650). */
    fun gemma3Npu(): File = root.resolve("litert/Gemma3-1B-IT_q4_ekv1280_sm8650.litertlm")

    fun missingSummary(variant: GemmaVariant): List<String> {
        val missing = mutableListOf<String>()
        if (!llmReady(variant)) missing += "LLM: ${llmFile(variant).name}"
        if (!mmprojReady(variant)) missing += "mmproj: ${mmprojFile(variant).name} (native audio/image)"
        if (!senseVoiceReady()) missing += "ASR sense-voice/ (pipeline mode)"
        if (!kokoroReady()) missing += "TTS kokoro-multi-lang-v1_0/ (pipeline mode)"
        return missing
    }

    companion object {
        fun from(context: Context): ModelPaths =
            ModelPaths(context.filesDir.resolve("models"))
    }
}
