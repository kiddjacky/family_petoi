package com.gemmaproj.voicechat.engine

import org.json.JSONException
import org.json.JSONObject

/**
 * Gemma's spoken side of a walk order: a short answer to what the person said, and an idea
 * for how a parent and child can play with that object and the robot dog once it arrives.
 */
internal data class PlayIdea(val reply: String, val tip: String)

internal object ObjectPlay {
    private const val MAX_REPLY_CHARS = 160
    private const val MAX_TIP_CHARS = 320

    val SYSTEM_PROMPT: String = buildString {
        appendLine("You are the cheerful voice of a small robot dog that walks to objects on the floor.")
        appendLine("A person just told the dog to go to an object, and the dog is on its way.")
        appendLine("Answer with JSON only: {\"reply\": \"<answer>\", \"tip\": \"<idea>\"}")
        appendLine("<answer>: one short friendly sentence answering what they said, saying the dog is going to the object.")
        appendLine("Call the object what the person called it (they may say \"can\" where the camera says \"bottle\").")
        appendLine("<idea>: one or two short sentences on a simple game a parent and a young child can play with this object and the robot dog,")
        appendLine("  for example counting, colors, hide and seek, rolling, stacking or pretend play.")
        appendLine("The idea must be safe for small children: nothing thrown hard, nothing to put in the mouth, no drinking or opening containers, no climbing.")
        appendLine("Language: if they spoke Chinese, write both in simplified Chinese; otherwise English. No emojis, no markdown, no lists.")
        appendLine("Examples:")
        appendLine("They said: \"go to the ball\"  Camera sees: sports ball  Language: English")
        appendLine("{\"reply\": \"On my way to the ball!\", \"tip\": \"Roll the ball gently to your child and let the dog chase it; count how many rolls it takes to reach it.\"}")
        appendLine("They said: \"去那个杯子那里\"  Camera sees: cup  Language: Chinese")
        append("{\"reply\": \"好的，我这就去找杯子！\", \"tip\": \"把杯子倒扣在地上，让孩子把小玩具藏在下面，再让小狗来猜在哪个杯子里。\"}")
    }

    private val CHINESE = Regex("[\\u4e00-\\u9fff]")

    /** The user turn for [SYSTEM_PROMPT]; [color] is the spoken color, if any. */
    fun request(said: String, label: String, color: String?): String {
        val name = listOfNotNull(color, label).joinToString(" ")
        val language = if (CHINESE.containsMatchIn(said)) "Chinese" else "English"
        return "They said: \"${said.trim().take(200)}\"  Camera sees: $name  Language: $language"
    }

    /** Canned words for when Gemma's JSON is missing or broken; the walk never waits on this. */
    fun fallback(label: String, color: String?, said: String = ""): PlayIdea {
        val name = listOfNotNull(color, label).joinToString(" ")
        val reply = if (CHINESE.containsMatchIn(said)) "好的，这就过去。" else "Okay, going to the $name."
        return PlayIdea(reply = reply, tip = "")
    }

    fun parse(raw: String, label: String, color: String?, said: String = ""): PlayIdea {
        val fallback = fallback(label, color, said)
        val body = raw.trim().removePrefix("```json").removePrefix("```").removeSuffix("```").trim()
        val start = body.indexOf('{')
        val end = body.lastIndexOf('}')
        if (start < 0 || end <= start) return fallback
        val json = try {
            JSONObject(body.substring(start, end + 1))
        } catch (error: JSONException) {
            return fallback
        }
        val reply = speakable(json.optString("reply", ""), MAX_REPLY_CHARS)
        val tip = speakable(json.optString("tip", ""), MAX_TIP_CHARS)
        return PlayIdea(reply = reply.ifEmpty { fallback.reply }, tip = tip)
    }

    /** TTS reads markdown and emoji aloud, so only plain sentences pass, cut at a sentence end. */
    private fun speakable(text: String, maxChars: Int): String {
        val plain = text
            .replace(Regex("[*_#`>\\[\\]]"), "")
            .replace(Regex("[\\p{So}\\p{Cn}\\uFE0F\\u200D]"), "")
            .replace(Regex("\\s+"), " ")
            .trim()
        if (plain.length <= maxChars) return plain
        val cut = plain.take(maxChars)
        val sentenceEnd = cut.indexOfLast { it in ".!?。！？" }
        return if (sentenceEnd > 0) cut.take(sentenceEnd + 1) else cut.trimEnd() + "…"
    }
}
