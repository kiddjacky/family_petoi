package com.gemmaproj.voicechat.engine

import org.json.JSONArray
import org.json.JSONObject

/** One robot command with how long to wait for it. */
internal data class DogAction(
    val command: String,
    val timeoutMs: Long,
    val turnHoldMs: Long = 0L,
    /** Eye cues: the firmware's echo for these is unverified, so a missing echo is not an error. */
    val optional: Boolean = false,
)

internal data class DogReply(
    val say: String,
    val actions: List<DogAction>,
    /** Codes Gemma asked for that are not on the allowlist; never sent. */
    val rejected: List<String>,
)

/**
 * Gemma answers as the robot dog with `{"say": "...", "do": [...]}`. The phone speaks `say`
 * (the robot has no speaker) and the robot runs `do`. Following opencat-gemma, model output is
 * never trusted: every code is checked against [CODES] and every move is made finite here.
 */
internal object DogSpeech {
    private enum class Kind { STEPS, TIMED, ONCE, EYES }

    private class Code(val meaning: String, val kind: Kind)

    /** A curated Quaddle subset of opencat-gemma `contract/quaddle.json`. Acrobatics are left out. */
    private val CODES: Map<String, Code> = linkedMapOf(
        "kwkF" to Code("walk forward", Kind.STEPS),
        "kbkF" to Code("walk backward", Kind.STEPS),
        "kwkL" to Code("turn left", Kind.TIMED),
        "kwkR" to Code("turn right", Kind.TIMED),
        "kvtL" to Code("spin left", Kind.TIMED),
        "kvtR" to Code("spin right", Kind.TIMED),
        "ksit" to Code("sit", Kind.ONCE),
        "krest" to Code("lie down / rest", Kind.ONCE),
        "kbalance" to Code("stand up and balance", Kind.ONCE),
        "kup" to Code("stand up", Kind.ONCE),
        "khi" to Code("say hi / wave", Kind.ONCE),
        "knd" to Code("nod (yes)", Kind.ONCE),
        "kwh" to Code("shake head (no / don't know)", Kind.ONCE),
        "kstr" to Code("stretch", Kind.ONCE),
        "kpu" to Code("push-up", Kind.ONCE),
        "kfiv" to Code("high-five", Kind.ONCE),
        "khsk" to Code("handshake", Kind.ONCE),
        "ksnf" to Code("sniff", Kind.ONCE),
        "kscrh" to Code("scratch", Kind.ONCE),
        "kzz" to Code("sleepy", Kind.ONCE),
        "kang" to Code("angry", Kind.ONCE),
        "kpd" to Code("play dead", Kind.ONCE),
        "veH" to Code("happy eyes", Kind.EYES),
        "veS" to Code("sad eyes", Kind.EYES),
        "vcr" to Code("red eyes", Kind.EYES),
        "vcb" to Code("blue eyes", Kind.EYES),
        "vcg" to Code("green eyes", Kind.EYES),
        "vcy" to Code("yellow eyes", Kind.EYES),
        "vcp" to Code("pink eyes", Kind.EYES),
        "vcv" to Code("purple eyes", Kind.EYES),
    )

    const val TALK_EYES = "veTH"
    const val END_TALK_EYES = "vet"

    private const val DEFAULT_STEPS = 3
    private const val MAX_STEPS = 10
    private const val TURN_HOLD_MS = 2_000L
    private const val ONCE_TIMEOUT_MS = 15_000L
    private const val EYES_TIMEOUT_MS = 1_500L
    private const val STEP_BASE_MS = 5_000L
    private const val STEP_EACH_MS = 1_500L
    private const val MAX_ACTIONS = 6
    private const val MAX_SAY_CHARS = 400

    val SYSTEM_PROMPT: String = buildString {
        appendLine("You are Quaddle, a small friendly robot dog. A person talks to you.")
        appendLine("Answer with JSON only: {\"say\": \"...\", \"do\": [...]}")
        appendLine("- \"say\": what you say out loud, as the dog. One or two short sentences. Plain words, no emoji.")
        appendLine("- When the person asks you to say some words, \"say\" is exactly those words, unchanged.")
        appendLine("- You run fully offline: no internet, no clock, no weather, no news. If asked about those, say kindly that you can't know that offline.")
        appendLine("- \"do\": robot commands in order, only codes from this list, or [] if none fit:")
        for ((code, info) in CODES) appendLine("$code: ${info.meaning}")
        appendLine("kwkF and kbkF may be followed by a space and 1 to $MAX_STEPS steps, e.g. \"kwkF 3\".")
        appendLine("Examples:")
        appendLine("\"hi there, can you lie down?\" -> {\"say\": \"Hi! Okay, lying down now.\", \"do\": [\"khi\", \"krest\"]}")
        appendLine("\"what time is it?\" -> {\"say\": \"I don't have a clock. I run offline.\", \"do\": [\"kwh\"]}")
        appendLine("\"say \\\"nice to meet you\\\"\" -> {\"say\": \"nice to meet you\", \"do\": []}")
        append("\"come here two steps and stretch\" -> {\"say\": \"Coming!\", \"do\": [\"kwkF 2\", \"kstr\"]}")
    }

    private val QUOTED = Regex("""^\s*(?:please\s+)?say\s*[:,]?\s*["“”'‘’](.+)["“”'‘’]\s*[.!?]?\s*$""", RegexOption.IGNORE_CASE)

    /** `say "…"` requests are spoken verbatim even if Gemma rewords them. */
    fun quotedWords(request: String): String? = QUOTED.find(request)?.groupValues?.get(1)?.trim()?.ifBlank { null }

    fun parse(raw: String, request: String): DogReply {
        val body = raw.trim().removePrefix("```json").removePrefix("```").removeSuffix("```").trim()
        val start = body.indexOf('{')
        val end = body.lastIndexOf('}')
        var say = ""
        val asked = mutableListOf<String>()
        if (start >= 0 && end > start) {
            try {
                val json = JSONObject(body.substring(start, end + 1))
                say = json.optString("say", "")
                val list: JSONArray? = json.optJSONArray("do")
                if (list != null) for (index in 0 until list.length()) asked += list.optString(index, "").trim()
            } catch (error: org.json.JSONException) {
                say = ""
            }
        }
        if (say.isBlank() && asked.isEmpty()) {
            // Not JSON: speak whatever text came back, run nothing.
            say = body.replace(Regex("[{}\\[\\]\"]"), " ").replace(Regex("\\s+"), " ").trim()
        }
        quotedWords(request)?.let { say = it }
        val actions = mutableListOf<DogAction>()
        val rejected = mutableListOf<String>()
        for (command in asked.filter { it.isNotEmpty() }) {
            val action = toAction(command)
            if (action == null) rejected += command else actions += action
        }
        return DogReply(say.take(MAX_SAY_CHARS).trim(), actions.take(MAX_ACTIONS), rejected)
    }

    private fun toAction(command: String): DogAction? {
        val parts = command.split(' ').filter { it.isNotEmpty() }
        val code = CODES[parts.firstOrNull() ?: return null] ?: return null
        val name = parts[0]
        return when (code.kind) {
            Kind.STEPS -> {
                if (parts.size > 2) return null
                val steps = if (parts.size == 2) parts[1].toIntOrNull() ?: return null else DEFAULT_STEPS
                if (steps !in 1..MAX_STEPS) return null
                DogAction("$name $steps", STEP_BASE_MS + STEP_EACH_MS * steps)
            }
            Kind.TIMED -> if (parts.size == 1) DogAction(name, ONCE_TIMEOUT_MS, turnHoldMs = TURN_HOLD_MS) else null
            Kind.ONCE -> if (parts.size == 1) DogAction(name, ONCE_TIMEOUT_MS) else null
            Kind.EYES -> if (parts.size == 1) eyes(name) else null
        }
    }

    fun eyes(code: String): DogAction = DogAction(code, EYES_TIMEOUT_MS, optional = true)
}
