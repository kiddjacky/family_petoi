package com.gemmaproj.voicechat.engine

import android.content.Context
import android.media.AudioAttributes
import android.media.MediaPlayer
import android.util.Log
import java.io.File

/**
 * Plays a caller-supplied recording when the dog reaches the toy.
 * Looks for files/attract.wav, then .mp3, then .m4a. Returns false when none is there
 * so the caller can speak a short line instead.
 */
internal object AttractSound {
    private val NAMES = listOf("attract.wav", "attract.mp3", "attract.m4a")

    fun play(context: Context): Boolean {
        val file = NAMES.map { File(context.filesDir, it) }.firstOrNull { it.isFile && it.length() > 0L }
            ?: return false
        val player = MediaPlayer()
        player.setAudioAttributes(
            AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_MEDIA)
                .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                .build(),
        )
        player.setOnCompletionListener {
            it.release()
        }
        player.setOnErrorListener { failed, _, _ ->
            failed.release()
            true
        }
        player.setDataSource(file.absolutePath)
        player.prepare()
        player.start()
        Log.i(TAG, "playing ${file.name}")
        return true
    }

    private const val TAG = "AttractSound"
}
