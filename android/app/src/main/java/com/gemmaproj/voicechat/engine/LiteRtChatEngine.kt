package com.gemmaproj.voicechat.engine

import android.content.Context
import android.util.Log
import com.google.ai.edge.litertlm.ActivationDataType
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
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

/**
 * Gemma 4 chat on the Adreno GPU through LiteRT-LM.
 * One conversation is kept so later turns reuse the prefilled system prompt.
 * Speculative decoding stays off: GPU+MTP produced garbage on short replies.
 */
@OptIn(ExperimentalApi::class)
class LiteRtChatEngine(context: Context) {
    private val cacheDir = context.applicationContext.cacheDir
    private val mutex = Mutex()
    private var engine: Engine? = null
    private var conversation: Conversation? = null
    private var loadedPath: String? = null
    private var audioEnabled = false
    private var visionEnabled = false
    private var instruction: String = ""

    @OptIn(ExperimentalApi::class)
    suspend fun load(
        model: File,
        systemPrompt: String,
        listen: Boolean = false,
        vision: Boolean = false,
    ) {
        mutex.withLock {
            require(model.isFile) { "LiteRT model not found: ${model.absolutePath}" }
            if (
                loadedPath == model.absolutePath &&
                audioEnabled == listen &&
                visionEnabled == vision &&
                conversation != null
            ) {
                return@withLock
            }
            closeLocked()
            withContext(Dispatchers.Default) {
                openGpu(model, systemPrompt, listen, vision)
            }
        }
    }

    @OptIn(ExperimentalApi::class)
    private fun openGpu(model: File, systemPrompt: String, listen: Boolean, vision: Boolean) {
        val kind = when {
            vision -> "vision"
            listen -> "audio"
            else -> "GPU"
        }
        Log.i(TAG, "Loading $kind ${model.name}")
        ExperimentalFlags.enableBenchmark = true
        val started = System.nanoTime()
        var created: Engine? = null
        var lastError: Exception? = null
        for (plan in startPlans(listen, vision)) {
            try {
                created = startEngine(model, listen, plan.vision, plan.activation)
                break
            } catch (error: Exception) {
                lastError = error
                Log.w(
                    TAG,
                    "engine start failed vision=${plan.vision} activation=${plan.activation}: ${error.message}",
                )
            }
        }
        val ready = created ?: throw lastError ?: IllegalStateException("engine start failed")
        try {
            instruction = systemPrompt
            val talk = ready.createConversation(conversationConfig())
            engine = ready
            conversation = talk
            loadedPath = model.absolutePath
            audioEnabled = listen
            visionEnabled = vision
            Log.i(TAG, "GPU ready ms=${elapsedMs(started)} audio=$listen vision=$vision")
        } catch (error: Exception) {
            ready.close()
            throw error
        }
    }

    /**
     * Text stays on the GPU. FP16 activations on this Adreno turn short replies
     * into channel tokens, so float32 is first. The vision adapter in the
     * multimodal bundle is marked CPU, so a GPU vision start is allowed to fail
     * over to CPU without giving up the text GPU.
     */
    private fun startPlans(listen: Boolean, vision: Boolean): List<StartPlan> {
        val float32 = ActivationDataType.FLOAT32
        return when {
            vision -> listOf(
                StartPlan(Backend.GPU(), float32),
                StartPlan(Backend.CPU(), float32),
                StartPlan(Backend.CPU(), null),
            )
            listen -> listOf(
                StartPlan(null, float32),
                StartPlan(null, null),
            )
            else -> listOf(StartPlan(null, float32))
        }
    }

    private fun startEngine(
        model: File,
        listen: Boolean,
        vision: Backend?,
        activation: ActivationDataType?,
    ): Engine {
        val created = Engine(
            EngineConfig(
                modelPath = model.absolutePath,
                backend = Backend.GPU(),
                visionBackend = vision,
                // The published audio encoder is constrained to CPU.
                audioBackend = if (listen) Backend.CPU() else null,
                maxNumTokens = CONTEXT_TOKENS,
                // One photo per conversation. The next photo starts a fresh one.
                maxNumImages = if (vision != null) 1 else null,
                cacheDir = cacheDir.absolutePath,
                activationDataType = activation,
            ),
        )
        try {
            created.initialize()
        } catch (error: Exception) {
            created.close()
            throw error
        }
        return created
    }

    private fun conversationConfig(): ConversationConfig = ConversationConfig(
        systemInstruction = Contents.of(instruction),
        samplerConfig = SamplerConfig(topK = 40, topP = 0.9, temperature = 0.4),
        maxOutputToken = 256,
        thinkingConfig = NO_THINKING,
        enableSpeculativeDecoding = false,
        prefillPrefaceOnInit = false,
    )

    private fun recreateConversation(systemPrompt: String = instruction) {
        val current = engine ?: error("GPU model is not loaded")
        instruction = systemPrompt
        try {
            conversation?.close()
        } catch (error: Exception) {
            Log.w(TAG, "conversation reset failed: ${error.message}")
        }
        conversation = null
        conversation = current.createConversation(conversationConfig())
    }

    fun generate(userText: String, maxTokens: Int): Flow<String> =
        stream(Contents.of(userText), maxTokens)

    /**
     * One turn under a different system prompt, in a fresh conversation. The chat's own
     * prompt comes back afterwards, even on cancel; earlier chat turns are not kept.
     */
    fun generateAs(systemPrompt: String, userText: String, maxTokens: Int): Flow<String> = flow {
        val chatPrompt = mutex.withLock {
            val saved = instruction
            recreateConversation(systemPrompt)
            saved
        }
        try {
            stream(Contents.of(userText), maxTokens).collect { emit(it) }
        } finally {
            withContext(NonCancellable) {
                mutex.withLock {
                    try {
                        recreateConversation(chatPrompt)
                    } catch (error: Exception) {
                        Log.w(TAG, "chat prompt restore failed: ${error.message}")
                    }
                }
            }
        }
    }.flowOn(Dispatchers.Default)

    /** [wavPath] is a 16 kHz mono WAV. The first line of the reply is the transcript. */
    fun generateAudio(wavPath: String, maxTokens: Int): Flow<String> =
        stream(
            Contents.of(
                listOf(
                    Content.AudioFile(wavPath),
                    Content.Text(HEAR_PROMPT),
                ),
            ),
            maxTokens,
        )

    /**
     * [jpegPath] is one upright JPEG. Each call starts a new conversation so
     * earlier photos do not stay in the 2048-token context.
     */
    fun generateImage(jpegPath: String, maxTokens: Int): Flow<String> =
        askImage(jpegPath, SCENE_PROMPT, maxTokens)

    /**
     * The photo is the camera view at the moment the user started speaking.
     * [question] is the SenseVoice transcript of that utterance.
     */
    fun answerImage(jpegPath: String, question: String, maxTokens: Int): Flow<String> =
        askImage(jpegPath, questionPrompt(question), maxTokens)

    private fun askImage(jpegPath: String, prompt: String, maxTokens: Int): Flow<String> = flow {
        mutex.withLock { recreateConversation() }
        stream(
            Contents.of(
                listOf(
                    Content.ImageFile(jpegPath),
                    Content.Text(prompt),
                ),
            ),
            maxTokens,
        ).collect { emit(it) }
    }.flowOn(Dispatchers.Default)

    private fun stream(contents: Contents, maxTokens: Int): Flow<String> = flow {
        val talk = conversation ?: error("GPU model is not loaded")
        val started = System.nanoTime()
        var committed = ""
        var firstMs = -1L
        talk.sendMessageAsync(
            contents,
            maxOutputToken = maxTokens,
            thinkingConfig = NO_THINKING,
        ).collect { message ->
            val chunk = messageText(message)
            if (chunk.isEmpty()) return@collect
            val delta = when {
                chunk.startsWith(committed) -> chunk.substring(committed.length)
                else -> chunk
            }
            if (delta.isEmpty()) return@collect
            committed = if (chunk.startsWith(committed)) chunk else committed + delta
            if (firstMs < 0) {
                firstMs = elapsedMs(started)
                Log.i(TAG, "ttft_ms=$firstMs")
            }
            emit(delta)
        }
        logBenchmark(talk, firstMs, elapsedMs(started), committed.length)
    }.flowOn(Dispatchers.Default)

    fun cancel() {
        try {
            conversation?.cancelProcess()
        } catch (error: Exception) {
            Log.w(TAG, "cancel failed: ${error.message}")
        }
    }

    fun close() {
        closeLocked()
    }

    private fun closeLocked() {
        try {
            conversation?.close()
        } catch (error: Exception) {
            Log.w(TAG, "conversation close failed: ${error.message}")
        }
        conversation = null
        try {
            engine?.close()
        } catch (error: Exception) {
            Log.w(TAG, "engine close failed: ${error.message}")
        }
        engine = null
        loadedPath = null
        audioEnabled = false
        visionEnabled = false
        instruction = ""
    }

    @OptIn(ExperimentalApi::class)
    private fun logBenchmark(talk: Conversation, firstMs: Long, totalMs: Long, chars: Int) {
        try {
            val bench = talk.getBenchmarkInfo()
            val line = "reply_ms=$totalMs chars=$chars ttft_ms=$firstMs " +
                "prefill_tokens=${bench.lastPrefillTokenCount} " +
                "prefill_tps=${"%.1f".format(bench.lastPrefillTokensPerSecond)} " +
                "decode_tokens=${bench.lastDecodeTokenCount} " +
                "decode_tps=${"%.1f".format(bench.lastDecodeTokensPerSecond)} " +
                "bench_ttft_s=${"%.3f".format(bench.timeToFirstTokenInSecond)}"
            Log.i(TAG, line)
            File(cacheDir, "hear-log.txt").appendText(line + "\n")
        } catch (error: Exception) {
            val line = "reply_ms=$totalMs chars=$chars ttft_ms=$firstMs bench_unavailable=${error.message}"
            Log.i(TAG, line)
            File(cacheDir, "hear-log.txt").appendText(line + "\n")
        }
    }

    private fun messageText(message: Message): String =
        message.contents.contents.mapNotNull { content ->
            (content as? Content.Text)?.text
        }.joinToString("")

    private fun elapsedMs(startedNanos: Long): Long = (System.nanoTime() - startedNanos) / 1_000_000

    companion object {
        private const val TAG = "LiteRtChatEngine"
        private const val CONTEXT_TOKENS = 2048
        private val NO_THINKING = ThinkingConfig(enableThinking = false)
        private const val HEAR_PROMPT =
            "The audio is the user's spoken message. " +
                "First line must be exactly: Heard: <the words they said>. " +
                "Then answer in one or two short spoken sentences."
        private const val SCENE_PROMPT =
            "This is one photo from the phone camera. " +
                "In two short sentences, say what place this looks like and what is happening. " +
                "Use only what is visible."

        private fun questionPrompt(question: String): String {
            val asked = question.replace("\"", "'").take(400)
            return "The photo is what the phone camera sees right now. " +
                "The user said: \"$asked\". " +
                "Answer that question from the photo and their words. " +
                "Use their language. A few short sentences."
        }

        private data class StartPlan(
            val vision: Backend?,
            val activation: ActivationDataType?,
        )
    }
}
