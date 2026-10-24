package com.gemmaproj.voicechat.engine

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder

class VoiceRecorder {
    private val lock = Any()
    private var audioRecord: AudioRecord? = null
    private val sampleRate = 16_000
    private val buffer = ArrayList<Float>(sampleRate * 30)

    @Volatile
    var isRecording: Boolean = false
        private set

    fun start(): Boolean {
        val minBuf = AudioRecord.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
        )
        if (minBuf <= 0) return false
        val record = AudioRecord(
            MediaRecorder.AudioSource.VOICE_RECOGNITION,
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            minBuf * 4,
        )
        if (record.state != AudioRecord.STATE_INITIALIZED) {
            record.release()
            return false
        }
        synchronized(lock) {
            buffer.clear()
            audioRecord = record
            isRecording = true
        }
        record.startRecording()
        if (record.recordingState == AudioRecord.RECORDSTATE_RECORDING) return true
        synchronized(lock) {
            isRecording = false
            audioRecord = null
        }
        record.release()
        return false
    }

    /** Call from a worker thread while [isRecording] is true. */
    fun pollChunk(): Int {
        val record = synchronized(lock) { audioRecord } ?: return 0
        val chunk = ShortArray(1024)
        val read = record.read(chunk, 0, chunk.size)
        if (read > 0) {
            synchronized(lock) {
                for (i in 0 until read) {
                    buffer.add(chunk[i] / 32768f)
                }
            }
        }
        return read
    }

    fun stopCapture(): FloatArray {
        val record = synchronized(lock) {
            isRecording = false
            val current = audioRecord
            audioRecord = null
            current
        } ?: return FloatArray(0)
        try {
            if (record.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
                record.stop()
            }
        } catch (_: IllegalStateException) {
            // Already stopped by a failed start.
        } finally {
            record.release()
        }
        return synchronized(lock) { buffer.toFloatArray() }
    }

    val capturedSampleRate: Int get() = sampleRate
}
