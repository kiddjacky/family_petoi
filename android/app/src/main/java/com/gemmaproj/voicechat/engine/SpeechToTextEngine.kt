package com.gemmaproj.voicechat.engine

import android.util.Log
import com.gemmaproj.voicechat.data.ModelPaths
import com.gemmaproj.voicechat.domain.SttEngine
import com.k2fsa.sherpa.onnx.OfflineRecognizer
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

class SpeechToTextEngine {
    private val mutex = Mutex()
    private var recognizer: OfflineRecognizer? = null
    private var activeEngine: SttEngine? = null

    suspend fun ensureLoaded(paths: ModelPaths, engine: SttEngine) = mutex.withLock {
        if (recognizer != null && activeEngine == engine) return@withLock
        recognizer?.release()
        recognizer = null
        val config = SherpaConfigFactory.recognizerConfig(paths, engine)
        Log.i(TAG, "Loading STT $engine")
        recognizer = OfflineRecognizer(config = config)
        activeEngine = engine
    }

    suspend fun transcribe(samples: FloatArray, sampleRate: Int = 16_000): String =
        withContext(Dispatchers.Default) {
            val rec = recognizer ?: error("STT not initialized")
            val started = System.nanoTime()
            val stream = rec.createStream()
            try {
                stream.acceptWaveform(samples, sampleRate)
                rec.decode(stream)
                val text = cleanTranscript(rec.getResult(stream).text)
                val ms = (System.nanoTime() - started) / 1_000_000
                Log.i(TAG, "stt_ms=$ms chars=${text.length} samples=${samples.size}")
                text
            } finally {
                stream.release()
            }
        }

    fun release() {
        recognizer?.release()
        recognizer = null
        activeEngine = null
    }

    companion object {
        private const val TAG = "SpeechToTextEngine"
        private val SENSE_TAG = Regex("<\\|[^|>]*\\|>")
        private val WHITESPACE = Regex("\\s+")

        fun cleanTranscript(raw: String): String {
            val cleaned = raw.replace(SENSE_TAG, " ").replace(WHITESPACE, " ").trim()
            val shouted = cleaned.length > 1 && cleaned == cleaned.uppercase()
            if (!shouted) return cleaned
            val sentence = cleaned.lowercase().replaceFirstChar { it.uppercase() }
            return sentence
        }
    }
}
