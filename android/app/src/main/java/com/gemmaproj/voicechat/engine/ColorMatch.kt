package com.gemmaproj.voicechat.engine

import android.graphics.Bitmap
import android.graphics.Color

/**
 * Tells same-class YOLO boxes apart by a spoken color: "the yellow can" versus a clear water
 * bottle, both "bottle" to YOLO. Only the middle of each box is sampled, so floor around the
 * object counts for little.
 */
internal object ColorMatch {
    /** English color words Gemma may return, and the Chinese ones a person may say. */
    val NAMES: Map<String, String> = linkedMapOf(
        "red" to "red", "红" to "red",
        "orange" to "orange", "橙" to "orange", "橘" to "orange",
        "yellow" to "yellow", "黄" to "yellow", "gold" to "yellow",
        "green" to "green", "绿" to "green",
        "blue" to "blue", "蓝" to "blue",
        "purple" to "purple", "紫" to "purple",
        "pink" to "pink", "粉" to "pink",
        "white" to "white", "白" to "white",
        "black" to "black", "黑" to "black",
    )

    private const val INNER = 0.6f
    private const val SAMPLES_PER_SIDE = 24

    /** The color named in [text], or null. */
    fun find(text: String): String? {
        val lower = text.lowercase()
        return NAMES.entries.firstOrNull { (word, _) ->
            if (word.first().code < 128) Regex("\\b$word\\b").containsMatchIn(lower) else lower.contains(word)
        }?.value
    }

    /** Share 0..1 of sampled pixels in the middle of [box] that look [color]. */
    fun share(bitmap: Bitmap, box: YoloBox, color: String): Float {
        val boxWidth = (box.right - box.left) * bitmap.width
        val boxHeight = (box.bottom - box.top) * bitmap.height
        val startX = box.left * bitmap.width + boxWidth * (1f - INNER) / 2f
        val startY = box.top * bitmap.height + boxHeight * (1f - INNER) / 2f
        val hsv = FloatArray(3)
        var matched = 0
        var total = 0
        for (row in 0 until SAMPLES_PER_SIDE) {
            for (column in 0 until SAMPLES_PER_SIDE) {
                val x = (startX + boxWidth * INNER * (column + 0.5f) / SAMPLES_PER_SIDE).toInt().coerceIn(0, bitmap.width - 1)
                val y = (startY + boxHeight * INNER * (row + 0.5f) / SAMPLES_PER_SIDE).toInt().coerceIn(0, bitmap.height - 1)
                Color.colorToHSV(bitmap.getPixel(x, y), hsv)
                total += 1
                if (looks(color, hsv[0], hsv[1], hsv[2])) matched += 1
            }
        }
        return if (total == 0) 0f else matched.toFloat() / total
    }

    private fun looks(color: String, hue: Float, saturation: Float, value: Float): Boolean {
        val vivid = saturation > 0.35f && value > 0.25f
        return when (color) {
            "red" -> vivid && (hue < 15f || hue >= 345f)
            "orange" -> vivid && hue >= 15f && hue < 40f
            "yellow" -> vivid && hue >= 40f && hue < 70f
            "green" -> vivid && hue >= 70f && hue < 170f
            "blue" -> vivid && hue >= 170f && hue < 260f
            "purple" -> vivid && hue >= 260f && hue < 320f
            "pink" -> saturation > 0.2f && value > 0.4f && hue >= 320f && hue < 345f
            "white" -> saturation < 0.15f && value > 0.75f
            "black" -> value < 0.2f
            else -> false
        }
    }
}
