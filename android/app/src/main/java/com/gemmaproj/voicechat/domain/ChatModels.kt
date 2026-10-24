package com.gemmaproj.voicechat.domain

enum class ChatMode {
    TEXT,
    VOICE,
}

/** Which Gemma 4 instruct GGUF to load. */
enum class GemmaVariant {
    E2B,
    E4B,
}

/**
 * PIPELINE: SenseVoice (or Whisper) then Kokoro.
 * NATIVE: the mic clip is a WAV passed straight into Gemma's audio encoder.
 */
enum class VoiceStack {
    PIPELINE,
    NATIVE,
}

enum class SttEngine {
    SENSE_VOICE,
    WHISPER_TINY_EN,
}

data class ChatMessage(
    val id: String,
    val role: Role,
    val text: String,
    val isStreaming: Boolean = false,
) {
    enum class Role { USER, ASSISTANT, SYSTEM }
}

data class UserPreferences(
    val chatMode: ChatMode = ChatMode.TEXT,
    val gemma: GemmaVariant = GemmaVariant.E2B,
    val voiceStack: VoiceStack = VoiceStack.PIPELINE,
    val sttEngine: SttEngine = SttEngine.SENSE_VOICE,
    val speakReplies: Boolean = true,
    val kokoroSpeakerId: Int = 0,
    val maxNewTokens: Int = 256,
    val systemPrompt: String = "You are a helpful, concise voice assistant. Reply briefly.",
)
