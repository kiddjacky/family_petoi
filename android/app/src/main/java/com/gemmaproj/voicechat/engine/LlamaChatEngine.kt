package com.gemmaproj.voicechat.engine

import android.content.Context
import android.util.Log
import com.arm.aichat.AiChat
import com.arm.aichat.InferenceEngine
import com.arm.aichat.UnsupportedArchitectureException
import com.gemmaproj.voicechat.data.ModelPaths
import com.gemmaproj.voicechat.domain.GemmaVariant
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout

class LlamaChatEngine(context: Context) {
    private val appContext = context.applicationContext
    private val engine: InferenceEngine = AiChat.getInferenceEngine(appContext)

    private var loadedPath: String? = null

    suspend fun load(paths: ModelPaths, variant: GemmaVariant) {
        loadFile(paths.llmFile(variant))
    }

    suspend fun loadFile(file: java.io.File) = withContext(Dispatchers.Default) {
        require(file.isFile) { "LLM not found: ${file.absolutePath}" }
        if (loadedPath == file.absolutePath && engine.state.value is InferenceEngine.State.ModelReady) {
            return@withContext
        }
        Log.i(TAG, "Loading LLM ${file.name}")
        awaitNativeLibrary()
        val state = engine.state.value
        if (loadedPath != null || state is InferenceEngine.State.Error) {
            engine.cleanUp()
        }
        try {
            engine.loadModel(file.absolutePath)
        } catch (e: UnsupportedArchitectureException) {
            throw IllegalStateException(
                "GGML CPU backend did not load. Reinstall the app so native libraries are extracted.",
                e,
            )
        }
        loadedPath = file.absolutePath
    }

    private suspend fun awaitNativeLibrary() {
        withTimeout(20_000) {
            engine.state.first { state ->
                state !is InferenceEngine.State.Uninitialized &&
                    state !is InferenceEngine.State.Initializing
            }
        }
    }

    suspend fun setSystemPrompt(prompt: String) = withContext(Dispatchers.Default) {
        engine.setSystemPrompt(prompt)
    }

    fun generate(userMessage: String, maxTokens: Int): Flow<String> =
        engine.sendUserPrompt(userMessage, predictLength = maxTokens)

    suspend fun bench(pp: Int, tg: Int, nr: Int): String = withContext(Dispatchers.Default) {
        engine.bench(pp = pp, tg = tg, pl = 1, nr = nr)
    }

    fun releaseModel() {
        if (loadedPath == null) return
        engine.cleanUp()
        loadedPath = null
    }

    fun destroy() {
        loadedPath = null
        engine.destroy()
    }

    companion object {
        private const val TAG = "LlamaChatEngine"
    }
}
