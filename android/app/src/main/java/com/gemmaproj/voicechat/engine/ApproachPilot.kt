package com.gemmaproj.voicechat.engine

import android.graphics.Bitmap
import java.util.Locale
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

/**
 * Walks the robot to a target object, one decision per camera frame.
 *
 * The camera cannot tell which way the robot faces, so heading is measured: after a forward
 * step, the robot box's movement on screen is its heading. A turn makes the heading unknown
 * again, so the next forward step re-measures it. The loop is therefore
 * "probe forward → turn toward the target → probe forward → …" until the boxes touch.
 *
 * YOLO proposes boxes. EmbeddingGemma decides which box is the robot and which is the target,
 * only when tracking by overlap is lost, so most frames cost one YOLO pass and no embedding.
 *
 * Screen geometry assumes the rear camera looks down at the floor from above the robot
 * (not mirrored), so clockwise on screen is a right turn for the robot.
 */
internal class ApproachPilot {
    private var robot: Rect? = null
    private var target: Rect? = null
    private var targetMissed = 0
    private var robotMissed = 0
    private var heading: Vec? = null
    private var probeFrom: Vec? = null
    private var stalls = 0
    private var actions = 0

    fun reset() {
        robot = null
        target = null
        targetMissed = 0
        robotMissed = 0
        heading = null
        probeFrom = null
        stalls = 0
        actions = 0
    }

    suspend fun step(bitmap: Bitmap, yolo: YoloDetector, embed: EmbeddingLocator, targetQuery: String): ApproachFrame {
        val width = bitmap.width
        val height = bitmap.height
        val boxes = yolo.detect(bitmap, encodeCrops = false)
        val picked = identify(bitmap, boxes, embed, targetQuery)
        val robotBox = picked.robot
        val targetBox = picked.target
        val targetName = targetQuery.removePrefix("a ").removePrefix("an ")

        if (robotBox == null) {
            robotMissed += 1
            probeFrom = null
            return frame(
                width, height, boxes, picked,
                decision = null,
                note = "Can't see the dog.",
                gaveUp = robotMissed >= GIVE_UP_FRAMES,
            )
        }
        robotMissed = 0
        val robotCenter = center(robotBox, width, height)
        measureHeading(robotCenter, robotBox, width, height)

        if (targetBox == null) {
            return frame(width, height, boxes, picked, decision = null, note = "Can't see the $targetName.")
        }
        val targetCenter = center(targetBox, width, height)
        if (touching(robotBox, targetBox)) {
            return frame(width, height, boxes, picked, Decision(STOP, 0L), "The dog reached the $targetName.", arrived = true)
        }
        if (actions >= MAX_ACTIONS) {
            return frame(width, height, boxes, picked, Decision(STOP, 0L), "Gave up after $MAX_ACTIONS moves.", gaveUp = true)
        }

        val toTarget = Vec(targetCenter.x - robotCenter.x, targetCenter.y - robotCenter.y)
        val robotSize = max(robotBox.width * width, robotBox.height * height)
        val gapInBodies = toTarget.length / max(robotSize, 1f)
        val known = heading
        val decision = if (known == null) {
            probeFrom = robotCenter
            Decision(forward(gapInBodies, probing = true), 0L)
        } else {
            val angle = signedAngleDegrees(known, toTarget)
            when {
                abs(angle) > TURN_ABOVE_DEGREES -> {
                    heading = null
                    probeFrom = null
                    Decision(if (angle > 0f) RIGHT else LEFT, turnHold(abs(angle)))
                }
                else -> {
                    probeFrom = robotCenter
                    Decision(forward(gapInBodies, probing = false), 0L)
                }
            }
        }
        actions += 1
        val angleText = known?.let { "angle ${signedAngleDegrees(it, toTarget).toInt()}°" } ?: "heading unknown"
        val note = "${describe(decision)} ${String.format(Locale.US, "%.1f", gapInBodies)} bodies to the $targetName, $angleText."
        return frame(width, height, boxes, picked, decision, note)
    }

    /** After a forward step, the robot's movement on screen is the direction it faces. */
    private fun measureHeading(now: Vec, robotBox: Rect, width: Int, height: Int) {
        val start = probeFrom ?: return
        probeFrom = null
        val moved = Vec(now.x - start.x, now.y - start.y)
        val robotSize = max(robotBox.width * width, robotBox.height * height)
        if (moved.length < max(MIN_MOVE_PX, robotSize * MIN_MOVE_BODY)) {
            stalls += 1
            return
        }
        stalls = 0
        val unit = moved.unit()
        val previous = heading
        heading = if (previous == null) unit else Vec(previous.x + unit.x, previous.y + unit.y).unit()
    }

    private fun forward(gapInBodies: Float, probing: Boolean): String {
        // Short probes so a wrong guess costs little; longer strides when far and aimed.
        if (probing || stalls >= STALLS_FOR_LONG) return if (stalls >= STALLS_FOR_LONG) "kwkF 3" else "kwkF 2"
        return when {
            gapInBodies > 4f -> "kwkF 4"
            gapInBodies > 2f -> "kwkF 3"
            gapInBodies > 1f -> "kwkF 2"
            else -> "kwkF 1"
        }
    }

    private fun turnHold(angle: Float): Long = when {
        angle > 120f -> 2_000L
        angle > 75f -> 1_400L
        else -> 800L
    }

    private suspend fun identify(bitmap: Bitmap, boxes: List<YoloBox>, embed: EmbeddingLocator, query: String): Picked {
        val candidates = boxes.indices.filter { boxes[it].label != "person" }
        var robotIndex = track(boxes, candidates, robot)
        var targetIndex = track(boxes, candidates.filter { it != robotIndex }, target)
        if (robotIndex < 0 || targetIndex < 0) {
            val pool = candidates
                .filter { it != robotIndex && it != targetIndex }
                .sortedByDescending { boxes[it].score }
                .take(CANDIDATES)
            if (pool.isNotEmpty()) {
                val vectors = embed.embedImages(pool.map { boxes[it].jpegBytes(bitmap, 70) })
                val scores = embed.scoreVectors(vectors, listOf(DOG_QUERY, query))
                val dogScores = scores.getValue(DOG_QUERY)
                val targetScores = scores.getValue(query)
                if (robotIndex < 0) {
                    val best = dogScores.indices.maxByOrNull { dogScores[it] - targetScores[it] }
                    if (best != null && dogScores[best] >= ROBOT_FLOOR && dogScores[best] > targetScores[best]) {
                        robotIndex = pool[best]
                    }
                }
                if (targetIndex < 0) {
                    val best = targetScores.indices
                        .filter { pool[it] != robotIndex }
                        .maxByOrNull { targetScores[it] }
                    if (best != null && targetScores[best] >= TARGET_FLOOR) targetIndex = pool[best]
                }
            }
        }
        robot = if (robotIndex >= 0) boxes[robotIndex].rect() else null
        if (targetIndex >= 0) {
            target = boxes[targetIndex].rect()
            targetMissed = 0
        } else if (target != null) {
            // The target does not move; keep its last box through a few missed detections.
            targetMissed += 1
            if (targetMissed > TARGET_MEMORY_FRAMES) target = null
        }
        return Picked(robotIndex, targetIndex, robot, target)
    }

    private fun track(boxes: List<YoloBox>, indices: List<Int>, last: Rect?): Int {
        if (last == null) return -1
        return indices
            .filter { iou(boxes[it].rect(), last) >= TRACK_IOU }
            .maxByOrNull { iou(boxes[it].rect(), last) }
            ?: -1
    }

    private fun frame(
        width: Int,
        height: Int,
        boxes: List<YoloBox>,
        picked: Picked,
        decision: Decision?,
        note: String,
        arrived: Boolean = false,
        gaveUp: Boolean = false,
    ): ApproachFrame {
        val drawn = boxes.mapIndexed { index, box ->
            val role = when (index) {
                picked.robotIndex -> "dog"
                picked.targetIndex -> "target"
                else -> "other"
            }
            PlayBox(if (role == "dog") "robot" else box.label, role, box.left, box.top, box.right, box.bottom, box.score)
        }.toMutableList()
        val remembered = picked.target
        if (picked.targetIndex < 0 && remembered != null) {
            drawn += PlayBox("target (last seen)", "target", remembered.left, remembered.top, remembered.right, remembered.bottom, 0f)
        }
        val robotBox = picked.robot
        val arrow = heading
        val headingLine = if (robotBox != null && arrow != null) {
            val start = center(robotBox, width, height)
            val reach = max(robotBox.width * width, robotBox.height * height)
            listOf(
                start.x / width, start.y / height,
                (start.x + arrow.x * reach) / width, (start.y + arrow.y * reach) / height,
            )
        } else {
            emptyList()
        }
        return ApproachFrame(
            caption = note,
            boxes = drawn,
            width = width,
            height = height,
            command = decision?.command,
            turnHoldMs = decision?.turnHoldMs ?: 0L,
            arrived = arrived,
            gaveUp = gaveUp,
            headingLine = headingLine,
            log = "robot=${picked.robotIndex} target=${picked.targetIndex} heading=${heading?.text() ?: "-"} " +
                "stalls=$stalls actions=$actions cmd=${decision?.command ?: "-"} hold=${decision?.turnHoldMs ?: 0} $note",
        )
    }

    private data class Decision(val command: String, val turnHoldMs: Long)

    private data class Picked(val robotIndex: Int, val targetIndex: Int, val robot: Rect?, val target: Rect?)

    private data class Rect(val left: Float, val top: Float, val right: Float, val bottom: Float) {
        val width: Float get() = right - left
        val height: Float get() = bottom - top
    }

    private data class Vec(val x: Float, val y: Float) {
        val length: Float get() = hypot(x, y)
        fun unit(): Vec = if (length < 1e-6f) this else Vec(x / length, y / length)
        fun text(): String = String.format(Locale.US, "%.2f,%.2f", x, y)
    }

    private fun YoloBox.rect(): Rect = Rect(left, top, right, bottom)

    companion object {
        const val STOP = "kbalance"
        private const val LEFT = "kwkL"
        private const val RIGHT = "kwkR"
        private const val DOG_QUERY = "a robot dog"
        private const val ROBOT_FLOOR = 0.35f
        private const val TARGET_FLOOR = 0.30f
        private const val CANDIDATES = 4
        private const val TRACK_IOU = 0.30f
        private const val TARGET_MEMORY_FRAMES = 5
        private const val GIVE_UP_FRAMES = 20
        private const val MAX_ACTIONS = 80
        private const val TURN_ABOVE_DEGREES = 35f
        private const val MIN_MOVE_PX = 6f
        private const val MIN_MOVE_BODY = 0.08f
        private const val STALLS_FOR_LONG = 2
        /** Robot box grown by this share of its size; touching the target counts as arrived. */
        private const val REACH = 0.25f

        private fun describe(decision: Decision): String {
            val command = decision.command
            return when {
                command == LEFT -> "Turn left ${decision.turnHoldMs} ms."
                command == RIGHT -> "Turn right ${decision.turnHoldMs} ms."
                command.startsWith("kwkF") -> "Forward ×${command.substringAfter(' ', "1")}."
                else -> command
            }
        }

        private fun center(box: Rect, width: Int, height: Int): Vec =
            Vec((box.left + box.right) / 2f * width, (box.top + box.bottom) / 2f * height)

        private fun touching(robot: Rect, target: Rect): Boolean {
            val padX = robot.width * REACH
            val padY = robot.height * REACH
            return robot.left - padX < target.right && robot.right + padX > target.left &&
                robot.top - padY < target.bottom && robot.bottom + padY > target.top
        }

        /** Positive when the target is clockwise from the heading on screen (y points down). */
        private fun signedAngleDegrees(heading: Vec, toTarget: Vec): Float {
            val cross = heading.x * toTarget.y - heading.y * toTarget.x
            val dot = heading.x * toTarget.x + heading.y * toTarget.y
            return Math.toDegrees(atan2(cross, dot).toDouble()).toFloat()
        }

        private fun iou(a: Rect, b: Rect): Float {
            val overlapW = max(0f, min(a.right, b.right) - max(a.left, b.left))
            val overlapH = max(0f, min(a.bottom, b.bottom) - max(a.top, b.top))
            val overlap = overlapW * overlapH
            val union = a.width * a.height + b.width * b.height - overlap
            return if (union <= 0f) 0f else overlap / union
        }
    }
}

internal data class ApproachFrame(
    val caption: String,
    val boxes: List<PlayBox>,
    val width: Int,
    val height: Int,
    val command: String?,
    val turnHoldMs: Long,
    val arrived: Boolean,
    val gaveUp: Boolean,
    /** Normalized x1, y1, x2, y2 of the measured heading arrow, or empty. */
    val headingLine: List<Float>,
    val log: String,
)
