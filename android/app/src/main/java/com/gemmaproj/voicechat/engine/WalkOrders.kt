package com.gemmaproj.voicechat.engine

import org.json.JSONException
import org.json.JSONObject

/** What a spoken line means for the walk-to-target loop. */
internal sealed interface WalkOrder {
    /**
     * [label] is one of YOLO's COCO class names, the only things the walk can find. [color]
     * ("yellow") picks one object when several share the class.
     */
    data class GoTo(val label: String, val color: String? = null) : WalkOrder
    /** An object was named, but YOLO has no class for it. */
    data class CannotSee(val name: String) : WalkOrder
    data object Stop : WalkOrder
    /** Not about walking: a question or chat for the normal scene answer. */
    data object Other : WalkOrder
}

/**
 * Gemma reads what the person said (English or Chinese) and names the object as a YOLO class.
 * "Stop" never waits for Gemma: [isStop] catches it first.
 */
internal object WalkOrders {
    /** Classes that are never a walk target: people, furniture, and things a small robot cannot reach. */
    private val NOT_TARGETS = setOf(
        "person", "car", "bus", "train", "truck", "boat", "airplane", "motorcycle", "bicycle",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe", "dining table", "bed", "couch", "toilet",
        "refrigerator", "oven", "sink", "tv",
    )

    val TARGETS: List<String> = YoloDetector.COCO_NAMES.filter { it !in NOT_TARGETS }

    /** Everyday words YOLO's class names miss. Checked after an exact class name. */
    private val SYNONYMS = linkedMapOf(
        "water bottle" to "bottle",
        // YOLO has no can class; it calls drink cans bottles.
        "can" to "bottle",
        "soda" to "bottle",
        "ball" to "sports ball",
        "football" to "sports ball",
        "soccer" to "sports ball",
        "mug" to "cup",
        "glass" to "cup",
        "phone" to "cell phone",
        "teddy" to "teddy bear",
        "stuffed" to "teddy bear",
        "plush" to "teddy bear",
        "bag" to "handbag",
        "plant" to "potted plant",
        "computer" to "laptop",
    )

    val SYSTEM_PROMPT: String = buildString {
        appendLine("You read what a person says to a small robot dog on the floor. They may speak English or Chinese.")
        appendLine("The dog's camera only recognizes these objects:")
        appendLine(TARGETS.joinToString(", "))
        appendLine("Answer with JSON only, one of:")
        appendLine("{\"go\": \"<name>\", \"color\": \"<color>\"} when they want the dog to walk to, go to, find or approach an object.")
        appendLine("  <name> is the closest name from the list above, copied exactly. A drink can is \"bottle\". If nothing on the list fits, use the object's English name.")
        appendLine("  <color> is the object's color in English if they said one, otherwise leave \"color\" out.")
        appendLine("{\"stop\": true} when they want the dog to stop, wait or stay.")
        appendLine("{\"other\": true} for anything else: questions, chat, or no object named.")
        appendLine("Examples:")
        appendLine("\"go to the water bottle\" -> {\"go\": \"bottle\"}")
        appendLine("\"走到那个红色的杯子那里\" -> {\"go\": \"cup\", \"color\": \"red\"}")
        appendLine("\"go to the yellow can\" -> {\"go\": \"bottle\", \"color\": \"yellow\"}")
        appendLine("\"now go to the ball instead\" -> {\"go\": \"sports ball\"}")
        appendLine("\"去找那只小熊\" -> {\"go\": \"teddy bear\"}")
        appendLine("\"go to my shoe\" -> {\"go\": \"shoe\"}")
        appendLine("\"wait there\" -> {\"stop\": true}")
        appendLine("\"what do you see?\" -> {\"other\": true}")
        append("\"桌子上有什么\" -> {\"other\": true}")
    }

    private val STOP_WORDS = Regex("""\b(stop|halt|freeze)\b|停|站住|别动|别走|不要走""", RegexOption.IGNORE_CASE)

    /** Stopping must not wait for a model, so the obvious words are caught before Gemma runs. */
    fun isStop(said: String): Boolean = STOP_WORDS.containsMatchIn(said)

    /**
     * The YOLO class for a name like "a water bottle" or "Sports ball", or null if YOLO cannot see it.
     * An exact class name is always taken, even one Gemma is not offered ("tv" for a tapped box).
     */
    fun targetLabel(name: String): String? {
        val words = name.lowercase().replace(Regex("[^a-z ]"), " ").replace(Regex("\\s+"), " ").trim()
        if (words.isEmpty()) return null
        if (words in YoloDetector.COCO_NAMES) return words
        TARGETS.filter { Regex("\\b${Regex.escape(it)}\\b").containsMatchIn(words) }.maxByOrNull { it.length }?.let { return it }
        return SYNONYMS.entries.firstOrNull { Regex("\\b${Regex.escape(it.key)}\\b").containsMatchIn(words) }?.value
    }

    /** [said] is the person's own words; a color in them counts even if Gemma dropped it. */
    fun parse(raw: String, said: String = ""): WalkOrder {
        val body = raw.trim().removePrefix("```json").removePrefix("```").removeSuffix("```").trim()
        val start = body.indexOf('{')
        val end = body.lastIndexOf('}')
        if (start < 0 || end <= start) return WalkOrder.Other
        val json = try {
            JSONObject(body.substring(start, end + 1))
        } catch (error: JSONException) {
            return WalkOrder.Other
        }
        if (json.optBoolean("stop", false)) return WalkOrder.Stop
        val name = json.optString("go", "").replace(Regex("\\s+"), " ").trim()
        if (name.isEmpty()) return WalkOrder.Other
        val color = ColorMatch.find(json.optString("color", "")) ?: ColorMatch.find(said)
        return targetLabel(name)?.let { WalkOrder.GoTo(it, color) } ?: WalkOrder.CannotSee(name.take(40))
    }
}
