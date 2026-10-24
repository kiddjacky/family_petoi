package com.gemmaproj.voicechat.engine

import android.media.AudioAttributes
import android.os.Process
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.util.Log
import com.gemmaproj.voicechat.data.ModelPaths
import com.k2fsa.sherpa.onnx.OfflineTts
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

class TextToSpeechEngine {
    private val mutex = Mutex()
    private val trackLock = Any()
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private var tts: OfflineTts? = null
    private var track: AudioTrack? = null
    private var tailMs = 80L
    private var session: SpeechSession? = null

    suspend fun ensureLoaded(paths: ModelPaths) = mutex.withLock {
        if (tts != null) return@withLock
        val config = SherpaConfigFactory.kokoroTtsConfig(paths)
        Log.i(TAG, "Loading Kokoro TTS")
        val engine = OfflineTts(config = config)
        tts = engine
        initAudioTrack(engine)
        withContext(Dispatchers.Default) {
            Process.setThreadPriority(Process.THREAD_PRIORITY_URGENT_AUDIO)
            val started = System.nanoTime()
            engine.generate("Hi.", sid = 0, speed = 1f)
            Log.i(TAG, "kokoro_warmup_ms=${(System.nanoTime() - started) / 1_000_000}")
        }
    }

    /**
     * Speaks [text] in sentence-sized pieces. The next piece is synthesized
     * while the current one is still playing.
     */
    fun openSession(speakerId: Int, startedNanos: Long): SpeechSession {
        session?.stop()
        val engine = tts ?: error("TTS not initialized")
        val audioTrack = track ?: error("AudioTrack not initialized")
        val opened = SpeechSession(engine, audioTrack, speakerId, startedNanos)
        session = opened
        return opened
    }

    suspend fun speak(text: String, speakerId: Int, speed: Float = 1f) {
        val opened = openSession(speakerId, System.nanoTime())
        opened.speakPiece(text)
        opened.finish()
    }

    fun release() {
        session?.stop()
        session = null
        synchronized(trackLock) {
            track?.release()
            track = null
        }
        tts?.release()
        tts = null
    }

    private fun initAudioTrack(engine: OfflineTts) {
        val sampleRate = engine.sampleRate()
        val bufLength = AudioTrack.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_FLOAT,
        ).coerceAtLeast(sampleRate)
        tailMs = bufLength / 4 * 1000L / sampleRate.coerceAtLeast(1) + 60L
        val attr = AudioAttributes.Builder()
            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
            .setUsage(AudioAttributes.USAGE_MEDIA)
            .build()
        val format = AudioFormat.Builder()
            .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
            .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
            .setSampleRate(sampleRate)
            .build()
        track = AudioTrack(
            attr,
            format,
            bufLength,
            AudioTrack.MODE_STREAM,
            AudioManager.AUDIO_SESSION_ID_GENERATE,
        )
    }

    inner class SpeechSession(
        private val engine: OfflineTts,
        private val audioTrack: AudioTrack,
        speakerId: Int,
        private val startedNanos: Long,
    ) {
        private val sid = speakerId.coerceIn(0, (engine.numSpeakers() - 1).coerceAtLeast(0))
        private val pcm = Channel<FloatArray>(Channel.UNLIMITED)
        private val job: Job = scope.launch(Dispatchers.IO) { playQueuedAudio() }
        @Volatile private var stopped = false
        private var finished = false

        /**
         * Synthesizes [text] before returning, so Gemma's token loop waits and
         * Kokoro gets the CPU. Playback of earlier pieces continues in [job].
         */
        suspend fun speakPiece(text: String) {
            if (stopped || finished) return
            val cleaned = spokenText(text)
            if (cleaned.isEmpty()) return
            val audio = withContext(Dispatchers.Default) {
                Process.setThreadPriority(Process.THREAD_PRIORITY_URGENT_AUDIO)
                Log.i(TAG, "tts_chunk ms=${elapsedMs()} chars=${cleaned.length} text=$cleaned")
                val synthStarted = System.nanoTime()
                val generated = engine.generate(cleaned, sid = sid, speed = 1f)
                val kokoroMs = (System.nanoTime() - synthStarted) / 1_000_000
                Log.i(TAG, "kokoro_ms=$kokoroMs chars=${cleaned.length}")
                generated
            }
            if (!stopped && audio.samples.isNotEmpty()) pcm.trySend(audio.samples)
        }

        suspend fun finish() {
            if (stopped || finished) return
            finished = true
            pcm.close()
            job.join()
        }

        fun stop() {
            if (stopped) return
            stopped = true
            finished = true
            pcm.close()
            job.cancel()
            synchronized(trackLock) {
                try {
                    audioTrack.pause()
                    audioTrack.flush()
                } catch (_: IllegalStateException) {
                    // Track was already stopped.
                }
            }
        }

        private suspend fun playQueuedAudio() {
            Process.setThreadPriority(Process.THREAD_PRIORITY_AUDIO)
            var startedPlayback = false
            for (samples in pcm) {
                if (stopped || samples.isEmpty()) continue
                if (!startedPlayback) {
                    startedPlayback = true
                    synchronized(trackLock) {
                        if (!stopped) audioTrack.play()
                    }
                    Log.i(TAG, "tts_first_audio_ms=${elapsedMs()}")
                }
                var offset = 0
                while (offset < samples.size && !stopped) {
                    val count = minOf(2048, samples.size - offset)
                    val wrote = synchronized(trackLock) {
                        if (stopped) return
                        audioTrack.write(samples, offset, count, AudioTrack.WRITE_BLOCKING)
                    }
                    if (wrote <= 0) break
                    offset += wrote
                }
            }
            if (startedPlayback && !stopped) {
                delay(tailMs)
                synchronized(trackLock) {
                    if (!stopped) {
                        audioTrack.pause()
                        audioTrack.flush()
                    }
                }
            }
        }

        private fun elapsedMs(): Long = (System.nanoTime() - startedNanos) / 1_000_000
    }

    private fun spokenText(raw: String): String =
        raw.replace(TAG_PATTERN, " ")
            .replace(MARKDOWN_PATTERN, "")
            .replace(WHITESPACE, " ")
            .trim()
            .take(MAX_SPOKEN_CHARS)

    companion object {
        private const val TAG = "TextToSpeechEngine"
        private const val MAX_SPOKEN_CHARS = 240
        private val TAG_PATTERN = Regex("<\\|[^|>]*\\|>")
        private val MARKDOWN_PATTERN = Regex("[*_#`]")
        private val WHITESPACE = Regex("\\s+")
    }
}
