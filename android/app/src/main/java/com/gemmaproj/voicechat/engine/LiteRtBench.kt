package com.gemmaproj.voicechat.engine

import android.content.Context
import android.system.Os
import android.util.Log
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.ExperimentalApi
import com.google.ai.edge.litertlm.ExperimentalFlags
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.SamplerConfig
import com.google.ai.edge.litertlm.ThinkingConfig
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Loads a Gemma 4 `.litertlm` file and records load time, time to first token, and a short decode.
 * GPU is tried first. The GPU file is not a GGUF, so this does not touch the llama.cpp engine.
 */
class LiteRtBench(context: Context) {
    private val appContext = context.applicationContext

    @OptIn(ExperimentalApi::class)
    suspend fun run(model: File): String = withContext(Dispatchers.Default) {
        require(model.isFile) { "LiteRT model not found: ${model.absolutePath}" }
        ExperimentalFlags.enableBenchmark = true
        val attempts = listOf(
            Attempt("GPU", Backend.GPU(), speculative = false),
            Attempt("GPU+MTP", Backend.GPU(), speculative = true),
        )
        val lines = mutableListOf<String>()
        lines += "file=${model.name} bytes=${model.length()}"
        for (attempt in attempts) {
            try {
                lines += runAttempt(model, attempt)
            } catch (error: Exception) {
                Log.e(TAG, "${attempt.label} failed", error)
                lines += "${attempt.label} failed: ${error.message ?: error.javaClass.simpleName}"
            }
        }
        lines.joinToString("\n")
    }

    /**
     * Runs the SM8650 Hexagon build. [nativeLibraryDir] must contain the
     * phone's QNN HTP libraries (libQnnHtp.so, the v75 stub, system, and skel).
     * The compiled context is 1280 tokens, so this does not use the GPU file's
     * 2048-token window.
     */
    @OptIn(ExperimentalApi::class)
    suspend fun runNpu(model: File, nativeLibraryDir: File): String = withContext(Dispatchers.Default) {
        require(model.isFile) { "LiteRT model not found: ${model.absolutePath}" }
        require(nativeLibraryDir.isDirectory) { "QNN directory missing: ${nativeLibraryDir.absolutePath}" }
        Os.setenv("ADSP_LIBRARY_PATH", nativeLibraryDir.absolutePath, true)
        ExperimentalFlags.enableBenchmark = true
        val attempt = Attempt(
            label = "NPU",
            backend = Backend.NPU(nativeLibraryDir = nativeLibraryDir.absolutePath),
            speculative = false,
        )
        val header = "file=${model.name} bytes=${model.length()} qnn=${nativeLibraryDir.absolutePath}"
        val line = try {
            runAttempt(model, attempt, maxNumTokens = NPU_CONTEXT_TOKENS)
        } catch (error: Exception) {
            Log.e(TAG, "NPU failed", error)
            "NPU failed: ${error.message ?: error.javaClass.simpleName}"
        }
        "$header\n$line"
    }

    @OptIn(ExperimentalApi::class)
    private suspend fun runAttempt(
        model: File,
        attempt: Attempt,
        maxNumTokens: Int = CONTEXT_TOKENS,
    ): String {
        ExperimentalFlags.enableSpeculativeDecoding = attempt.speculative
        val config = EngineConfig(
            modelPath = model.absolutePath,
            backend = attempt.backend,
            maxNumTokens = maxNumTokens,
            cacheDir = appContext.cacheDir.absolutePath,
        )
        val engine = Engine(config)
        try {
            val loadStarted = System.nanoTime()
            engine.initialize()
            val loadMs = elapsedMs(loadStarted)
            Log.i(TAG, "${attempt.label} load_ms=$loadMs")

            val conversationConfig = ConversationConfig(
                systemInstruction = Contents.of(SYSTEM_PROMPT),
                samplerConfig = SamplerConfig(topK = 64, topP = 0.95, temperature = 1.0),
                maxOutputToken = DECODE_TOKENS,
                thinkingConfig = ThinkingConfig(enableThinking = false, thinkingTokenBudget = 0),
                enableSpeculativeDecoding = attempt.speculative,
                prefillPrefaceOnInit = true,
            )
            val prefaceStarted = System.nanoTime()
            val conversation = engine.createConversation(conversationConfig)
            try {
                val prefaceMs = elapsedMs(prefaceStarted)
                val reply = streamReply(conversation, PREFILL_PROMPT, maxTokens = DECODE_TOKENS)
                val bench = conversation.getBenchmarkInfo()
                val summary = buildString {
                    append("backend=${attempt.label} load_ms=$loadMs preface_ms=$prefaceMs ")
                    append("prefill_tokens=${bench.lastPrefillTokenCount} ")
                    append("prefill_tps=${"%.2f".format(bench.lastPrefillTokensPerSecond)} ")
                    append("decode_tokens=${bench.lastDecodeTokenCount} ")
                    append("decode_tps=${"%.2f".format(bench.lastDecodeTokensPerSecond)} ")
                    append("bench_ttft_s=${"%.3f".format(bench.timeToFirstTokenInSecond)} ")
                    append("bench_init_s=${"%.3f".format(bench.initTimeInSecond)} ")
                    append(reply)
                }
                Log.i(TAG, summary)
                return summary
            } finally {
                conversation.close()
            }
        } finally {
            engine.close()
        }
    }

    private suspend fun streamReply(
        conversation: Conversation,
        prompt: String,
        maxTokens: Int,
    ): String {
        val started = System.nanoTime()
        var firstMs = -1L
        var pieces = 0
        val text = StringBuilder()
        conversation.sendMessageAsync(
            prompt,
            maxOutputToken = maxTokens,
            thinkingConfig = ThinkingConfig(enableThinking = false, thinkingTokenBudget = 0),
        ).collect { message ->
            val chunk = messageText(message)
            if (chunk.isEmpty()) return@collect
            if (firstMs < 0) firstMs = elapsedMs(started)
            pieces += 1
            if (text.isNotEmpty() && chunk.startsWith(text.toString())) {
                text.clear()
            }
            text.append(chunk)
        }
        val totalMs = elapsedMs(started)
        val decodeMs = (totalMs - firstMs).coerceAtLeast(1)
        val afterFirst = (pieces - 1).coerceAtLeast(0)
        val piecesPerSec = afterFirst * 1000.0 / decodeMs
        return "ttft_ms=$firstMs pieces=$pieces pieces_per_s=${"%.2f".format(piecesPerSec)} " +
            "reply_ms=$totalMs text=${text.toString().replace('\n', ' ')}"
    }

    private fun messageText(message: Message): String =
        message.contents.contents.mapNotNull { content ->
            (content as? Content.Text)?.text
        }.joinToString("")

    private fun elapsedMs(startedNanos: Long): Long = (System.nanoTime() - startedNanos) / 1_000_000

    private data class Attempt(
        val label: String,
        val backend: Backend,
        val speculative: Boolean,
    )

    companion object {
        private const val TAG = "LiteRtBench"
        private const val CONTEXT_TOKENS = 2048
        private const val NPU_CONTEXT_TOKENS = 1280
        private const val DECODE_TOKENS = 128
        private const val SYSTEM_PROMPT = "Follow the user's request. Do not stop early."
        private val PREFILL_PROMPT =
            "The quick brown fox jumps over the lazy dog. ".repeat(45) +
                "Then list 70 animals separated by commas. No introduction."
    }
}
