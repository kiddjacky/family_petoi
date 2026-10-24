package com.gemmaproj.voicechat.engine

import android.graphics.Bitmap
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Once a second: YOLO finds a person holding an object, EmbeddingGemma registers that
 * crop, and the same vector has to match again before the dog is asked to walk.
 * The robot box is tracked between seconds so the loop embeds one crop, not every box.
 */
internal class PlayWatcher {
    private var registered: FloatArray? = null
    private var registeredName: String = ""
    private var streak: Int = 0
    private var mismatch: Int = 0
    private var robotLeft = 0f
    private var robotTop = 0f
    private var robotRight = 0f
    private var robotBottom = 0f
    private var robotKnown = false

    var announced: Boolean = false

    fun reset() {
        registered = null
        registeredName = ""
        streak = 0
        mismatch = 0
        robotKnown = false
        announced = false
    }

    suspend fun step(bitmap: Bitmap, yolo: YoloDetector, embed: EmbeddingLocator): PlayFrame {
        val boxes = yolo.detect(bitmap, encodeCrops = false)
        val personIndex = boxes.indices
            .filter { boxes[it].label == "person" }
            .maxByOrNull { boxes[it].score }
        if (personIndex == null) {
            streak = 0
            return frame(bitmap, boxes, personIndex = -1, heldIndex = -1, robotIndex = -1, command = null, note = "Waiting for a person.")
        }
        val person = boxes[personIndex]
        val robotIndex = robotIndex(boxes, personIndex, bitmap, embed)
        val heldIndex = heldIndex(boxes, person, personIndex, robotIndex)
        if (heldIndex < 0) {
            streak = 0
            return frame(bitmap, boxes, personIndex, heldIndex, robotIndex, command = null, note = "Waiting for a held object.")
        }
        val held = boxes[heldIndex]
        val vector = embed.embedImages(listOf(held.jpegBytes(bitmap, 70))).first()
        val memory = registered
        val similarity = if (memory == null) 1f else cosine(vector, memory)
        if (memory == null || similarity < NEW_OBJECT) {
            mismatch = if (memory == null) 0 else mismatch + 1
            streak = 0
            if (memory != null && mismatch < RETARGET) {
                return frame(
                    bitmap, boxes, personIndex, heldIndex, robotIndex, null,
                    "Held object changed (${score(similarity)}). Still the $registeredName.",
                )
            }
            registered = vector
            registeredName = toyName(embed, vector)
            mismatch = 0
            streak = 1
            announced = false
            return frame(bitmap, boxes, personIndex, heldIndex, robotIndex, null, "Registered the $registeredName.")
        }
        mismatch = 0
        streak += 1
        if (streak < PLAY_STREAK) {
            return frame(bitmap, boxes, personIndex, heldIndex, robotIndex, null, "Watching the $registeredName. $streak s.")
        }
        if (robotIndex < 0) {
            return frame(bitmap, boxes, personIndex, heldIndex, robotIndex, null, "Playing with the $registeredName. Looking for the dog.")
        }
        val robot = boxes[robotIndex]
        val dx = held.centerX - robot.centerX
        val dy = held.centerY - robot.centerY
        val command = playSkill(dx, dy)
        val note = if (command == null) {
            "The dog reached the $registeredName."
        } else {
            "Playing with the $registeredName. ${walkWords(command)}"
        }
        return frame(bitmap, boxes, personIndex, heldIndex, robotIndex, command, note)
    }

    private suspend fun robotIndex(
        boxes: List<YoloBox>,
        personIndex: Int,
        bitmap: Bitmap,
        embed: EmbeddingLocator,
    ): Int {
        val tracked = boxes.indices
            .filter { it != personIndex }
            .filter { robotKnown && boxIou(boxes[it], robotLeft, robotTop, robotRight, robotBottom) >= TRACK_IOU }
            .maxByOrNull { boxIou(boxes[it], robotLeft, robotTop, robotRight, robotBottom) }
        if (tracked != null) {
            rememberRobot(boxes[tracked])
            return tracked
        }
        robotKnown = false
        val pool = boxes.indices
            .filter { it != personIndex }
            .sortedByDescending { boxes[it].score }
            .take(ROBOT_CANDIDATES)
        if (pool.isEmpty()) return -1
        val vectors = embed.embedImages(pool.map { boxes[it].jpegBytes(bitmap, 70) })
        val scores = embed.scoreVectors(vectors, listOf(DOG_QUERY)).getValue(DOG_QUERY)
        var best = 0
        for (index in scores.indices) if (scores[index] > scores[best]) best = index
        if (scores[best] < ROBOT_FLOOR) return -1
        val chosen = pool[best]
        rememberRobot(boxes[chosen])
        return chosen
    }

    private fun rememberRobot(box: YoloBox) {
        robotLeft = box.left
        robotTop = box.top
        robotRight = box.right
        robotBottom = box.bottom
        robotKnown = true
    }

    private suspend fun toyName(embed: EmbeddingLocator, vector: FloatArray): String {
        val scores = embed.scoreVectors(listOf(vector), TOY_LABELS)
        var bestLabel = "toy"
        var bestScore = 0.3f
        for (label in TOY_LABELS) {
            val score = scores[label]?.firstOrNull() ?: continue
            if (score > bestScore) {
                bestScore = score
                bestLabel = label.removePrefix("a ").removePrefix("an ")
            }
        }
        return bestLabel
    }

    private fun frame(
        bitmap: Bitmap,
        boxes: List<YoloBox>,
        personIndex: Int,
        heldIndex: Int,
        robotIndex: Int,
        command: String?,
        note: String,
    ): PlayFrame {
        val drawn = boxes.mapIndexed { index, box ->
            val role = when (index) {
                robotIndex -> "dog"
                personIndex -> "person"
                heldIndex -> "target"
                else -> "other"
            }
            val label = when (role) {
                "dog" -> "robot"
                "person" -> "person"
                "target" -> registeredName.ifBlank { "toy" }
                else -> box.label
            }
            PlayBox(label, role, box.left, box.top, box.right, box.bottom, box.score)
        }
        return PlayFrame(
            caption = note,
            boxes = drawn,
            width = bitmap.width,
            height = bitmap.height,
            command = command,
            arrived = command == null && streak >= PLAY_STREAK && robotIndex >= 0 && heldIndex >= 0,
            playing = streak >= PLAY_STREAK,
            log = "streak=$streak mismatch=$mismatch person=$personIndex held=$heldIndex robot=$robotIndex " +
                "name=$registeredName cmd=${command ?: "-"} $note",
        )
    }

    companion object {
        private const val DOG_QUERY = "a robot dog"
        private const val PLAY_STREAK = 2
        private const val RETARGET = 2
        private const val SAME_OBJECT = 0.78f
        private const val NEW_OBJECT = SAME_OBJECT
        private const val ROBOT_FLOOR = 0.35f
        private const val TRACK_IOU = 0.30f
        private const val ROBOT_CANDIDATES = 3
        private val TOY_LABELS = listOf(
            "a water bottle",
            "a cup",
            "a ball",
            "a shoe",
            "a backpack",
            "a chair",
            "a phone",
            "a book",
            "a toy",
        )

        private fun heldIndex(boxes: List<YoloBox>, person: YoloBox, personIndex: Int, robotIndex: Int): Int {
            return boxes.indices
                .filter { it != personIndex && it != robotIndex && isHeld(person, boxes[it]) }
                .maxByOrNull { overlap(person, boxes[it]) }
                ?: -1
        }

        private fun isHeld(person: YoloBox, box: YoloBox): Boolean {
            val inside = box.centerX in person.left..person.right && box.centerY in person.top..person.bottom
            return inside || boxIou(box, person.left, person.top, person.right, person.bottom) >= 0.08f
        }

        private fun overlap(person: YoloBox, box: YoloBox): Float {
            return boxIou(box, person.left, person.top, person.right, person.bottom) +
                if (box.centerX in person.left..person.right && box.centerY in person.top..person.bottom) 1f else 0f
        }

        private fun boxIou(box: YoloBox, left: Float, top: Float, right: Float, bottom: Float): Float {
            val overlapW = max(0f, min(box.right, right) - max(box.left, left))
            val overlapH = max(0f, min(box.bottom, bottom) - max(box.top, top))
            val overlapArea = overlapW * overlapH
            val area = (box.right - box.left) * (box.bottom - box.top) +
                (right - left) * (bottom - top) - overlapArea
            if (area <= 0f) return 0f
            return overlapArea / area
        }

        private fun score(value: Float): String = String.format(java.util.Locale.US, "%.2f", value)

        private fun cosine(left: FloatArray, right: FloatArray): Float {
            val count = minOf(left.size, right.size)
            if (count == 0) return 0f
            var dot = 0.0
            var leftNorm = 0.0
            var rightNorm = 0.0
            for (index in 0 until count) {
                val a = left[index].toDouble()
                val b = right[index].toDouble()
                dot += a * b
                leftNorm += a * a
                rightNorm += b * b
            }
            val denom = sqrt(leftNorm) * sqrt(rightNorm)
            if (denom < 1e-8) return 0f
            return (dot / denom).toFloat()
        }

        internal fun playSkill(dx: Float, dy: Float): String? {
            if (hypot(dx.toDouble(), dy.toDouble()) < 0.12) return null
            return when {
                kotlin.math.abs(dx) >= kotlin.math.abs(dy) && dx > 0f -> "kwkR"
                kotlin.math.abs(dx) >= kotlin.math.abs(dy) -> "kwkL"
                dy < 0f -> "kwkF 1"
                else -> "kbkF 1"
            }
        }

        internal fun walkWords(command: String): String = when (command) {
            "kwkR" -> "Go right."
            "kwkL" -> "Go left."
            "kwkF 1" -> "Go forward."
            "kbkF 1" -> "Go back."
            else -> command
        }
    }
}

internal data class PlayBox(
    val label: String,
    val role: String,
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float,
    val score: Float,
)

internal data class PlayFrame(
    val caption: String,
    val boxes: List<PlayBox>,
    val width: Int,
    val height: Int,
    val command: String?,
    val arrived: Boolean,
    val playing: Boolean,
    val log: String,
)
