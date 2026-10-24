package com.gemmaproj.voicechat.engine

import com.gemmaproj.voicechat.data.ModelPaths
import com.gemmaproj.voicechat.domain.SttEngine
import com.k2fsa.sherpa.onnx.OfflineModelConfig
import com.k2fsa.sherpa.onnx.OfflineRecognizerConfig
import com.k2fsa.sherpa.onnx.OfflineSenseVoiceModelConfig
import com.k2fsa.sherpa.onnx.OfflineTtsConfig
import com.k2fsa.sherpa.onnx.OfflineTtsKokoroModelConfig
import com.k2fsa.sherpa.onnx.OfflineTtsModelConfig
import com.k2fsa.sherpa.onnx.OfflineWhisperModelConfig
import com.k2fsa.sherpa.onnx.getFeatureConfig
import java.io.File

object SherpaConfigFactory {
    private const val SAMPLE_RATE = 16_000

    fun recognizerConfig(paths: ModelPaths, engine: SttEngine): OfflineRecognizerConfig {
        val modelConfig = when (engine) {
            SttEngine.SENSE_VOICE -> senseVoiceConfig(paths.senseVoiceDir)
            SttEngine.WHISPER_TINY_EN -> whisperTinyConfig(paths.whisperTinyDir)
        }
        return OfflineRecognizerConfig(
            featConfig = getFeatureConfig(sampleRate = SAMPLE_RATE, featureDim = 80),
            modelConfig = modelConfig,
        )
    }

    private fun senseVoiceConfig(dir: File): OfflineModelConfig {
        require(dir.isDirectory) { "Missing ${dir.absolutePath}" }
        return OfflineModelConfig(
            senseVoice = OfflineSenseVoiceModelConfig(
                model = dir.resolve("model.int8.onnx").absolutePath,
            ),
            tokens = dir.resolve("tokens.txt").absolutePath,
            modelType = "sense_voice",
            numThreads = 4,
        )
    }

    private fun whisperTinyConfig(dir: File): OfflineModelConfig {
        require(dir.isDirectory) { "Missing ${dir.absolutePath}" }
        return OfflineModelConfig(
            whisper = OfflineWhisperModelConfig(
                encoder = dir.resolve("tiny.en-encoder.int8.onnx").absolutePath,
                decoder = dir.resolve("tiny.en-decoder.int8.onnx").absolutePath,
            ),
            tokens = dir.resolve("tiny.en-tokens.txt").absolutePath,
            modelType = "whisper",
            numThreads = 4,
        )
    }

    fun kokoroTtsConfig(paths: ModelPaths): OfflineTtsConfig {
        val dir = paths.kokoroDir
        val model = dir.resolve("model.onnx")
        val voices = dir.resolve("voices.bin")
        val tokens = dir.resolve("tokens.txt")
        val espeak = dir.resolve("espeak-ng-data")
        require(model.isFile) { "Missing ${model.absolutePath}" }
        require(voices.isFile) { "Missing ${voices.absolutePath}" }
        require(tokens.isFile) { "Missing ${tokens.absolutePath}" }
        require(espeak.isDirectory) { "Missing ${espeak.absolutePath}" }
        val lexicon = listOf("lexicon-us-en.txt", "lexicon-zh.txt")
            .map { dir.resolve(it) }
            .filter { it.isFile }
            .joinToString(",") { it.absolutePath }
        val ruleFsts = listOf("phone-zh.fst", "date-zh.fst", "number-zh.fst")
            .map { dir.resolve(it) }
            .filter { it.isFile }
            .joinToString(",") { it.absolutePath }
        return OfflineTtsConfig(
            model = OfflineTtsModelConfig(
                kokoro = OfflineTtsKokoroModelConfig(
                    model = model.absolutePath,
                    voices = voices.absolutePath,
                    tokens = tokens.absolutePath,
                    dataDir = espeak.absolutePath,
                    lexicon = lexicon,
                    dictDir = dir.resolve("dict").takeIf { it.isDirectory }?.absolutePath ?: "",
                ),
                numThreads = 4,
                debug = false,
                provider = "cpu",
            ),
            ruleFsts = ruleFsts,
        )
    }
}
