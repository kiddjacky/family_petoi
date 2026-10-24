package com.gemmaproj.voicechat

import android.app.Application
import android.util.Log
import com.gemmaproj.voicechat.engine.DogLink
import com.github.csukuangfj.piper.phonemize.piperPhonemizeInitialize

class GemmaVoiceApp : Application() {
    @Volatile
    var benchVariant: String? = null

    @Volatile
    var litertBench: Boolean = false

    @Volatile
    var voiceProbeWav: String? = null

    @Volatile
    var autoAsk: String? = null

    /** A request for the dog, asked once Gemma is ready, as if typed on the Dog tab. */
    @Volatile
    var dogAsk: String? = null

    /** Absolute path of a photo to scan for the object a person is pointing at. */
    @Volatile
    var pointPhoto: String? = null

    /** When true, the mic bypasses SenseVoice and Gemma hears the WAV itself. */
    @Volatile
    var gemmaAudio: Boolean = false

    /** Load YOLO and EmbeddingGemma once, then time repeated runs. Skips Gemma chat. */
    @Volatile
    var profileTrack: Boolean = false

    @Volatile
    var profilePhoto: String? = null

    /** Time EmbeddingGemma on the GPU instead of the CPU. */
    @Volatile
    var profileGpu: Boolean = false

    /** Connect over Bluetooth LE and record the robot's '?' reply. Sends no gait. */
    @Volatile
    var bleProbe: Boolean = false

    /** The one robot link for the process; screens come and go, this stays connected. */
    val dogLink: DogLink by lazy { DogLink(this) }

    override fun onCreate() {
        super.onCreate()
        try {
            piperPhonemizeInitialize(this)
        } catch (e: UnsatisfiedLinkError) {
            Log.w(TAG, "piper-phonemize init failed (Kokoro may need lexicon files): ${e.message}")
        }
    }

    companion object {
        private const val TAG = "GemmaVoiceApp"
    }
}
