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
 * YOLO alone finds both. The target is the box with the target's COCO class. The robot has no
 * COCO class (YOLO calls it a dog, a cat, even a motorcycle), so it is the remaining small box,
 * and once found it is followed by position, not by whatever label YOLO gives it next.
 *
 * Screen geometry assumes the rear camera looks down at the floor from above the robot
 * (not mirrored), so clockwise on screen is a right turn for the robot.
 */
internal class ApproachPilot {
    private var robot: Rect? = null
    private var target: Rect? = null
    private var targetMissed = 0
    private var robotMissed = 0
    /** Frames the locked robot has been unseen; it is dropped after [ROBOT_MEMORY_FRAMES]. */
    private var robotLost = 0
    private var heading: Vec? = null
    private var probeFrom: Vec? = null
    private var stalls = 0
    private var actions = 0
    /** Every possible robot box just before a probe step; the one that moves is the robot. */
    private var identifyFrom: List<Rect>? = null
    private var identifyTries = 0

    fun reset() {
        robot = null
        target = null
        targetMissed = 0
        robotMissed = 0
        robotLost = 0
        heading = null
        probeFrom = null
        stalls = 0
        actions = 0
        identifyFrom = null
        identifyTries = 0
    }

    /** A new target mid-walk. The robot box and its measured heading are still true, so they stay. */
    fun retarget() {
        target = null
        targetMissed = 0
        actions = 0
    }

    /** [targetLabel] is a COCO class name such as "bottle"; [color] (e.g. "yellow") picks among several. */
    fun step(bitmap: Bitmap, yolo: YoloDetector, targetLabel: String, color: String? = null): ApproachFrame {
        val width = bitmap.width
        val height = bitmap.height
        val boxes = yolo.detectTiled(bitmap)
        var picked = identify(bitmap, boxes, targetLabel, color)
        val targetName = if (color != null) "$color $targetLabel" else targetLabel

        // A locked robot YOLO missed this frame keeps walking from its last box, so the robot
        // is never left idle waiting for the detector; it is matched again a frame or two later.
        val remembered = if (picked.robot == null) robot else null
        if (picked.robot == null && remembered == null) {
            val moved = whichOneMoved(boxes, picked.pool, width, height)
                ?: return lookForRobot(width, height, boxes, picked)
            robot = boxes[moved].rect()
            picked = picked.copy(robotIndex = moved, robot = robot)
        }
        val seen = picked.robot != null
        val robotBox = picked.robot ?: remembered ?: return lookForRobot(width, height, boxes, picked)
        val targetBox = picked.target
        robotMissed = 0
        val robotCenter = center(robotBox, width, height)
        if (seen) measureHeading(robotCenter, robotBox, width, height)
        val unseenNote = if (seen) "" else " (dog unseen, using its last place)"

        if (targetBox == null) {
            return frame(width, height, boxes, picked, decision = null, note = "Can't see the $targetName.")
        }
        val targetCenter = center(targetBox, width, height)
        if (seen && touching(robotBox, targetBox)) {
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
            if (seen) probeFrom = robotCenter
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
                    // Unseen: keep the older start point so the next sighting still measures heading.
                    if (seen) probeFrom = robotCenter
                    Decision(forward(gapInBodies, probing = false), 0L)
                }
            }
        }
        actions += 1
        val angleText = known?.let { "angle ${signedAngleDegrees(it, toTarget).toInt()}°" } ?: "heading unknown"
        val note = "${describe(decision)} ${String.format(Locale.US, "%.1f", gapInBodies)} bodies to the $targetName, $angleText.$unseenNote"
        return frame(width, height, boxes, picked, decision, note)
    }

    /**
     * Several robot dogs look alike, so ours is the box that moved after
     * our probe step. Every box shifts the same way when the phone moves, so the median shift is
     * taken out first. The winner's movement is also its heading.
     */
    private fun whichOneMoved(boxes: List<YoloBox>, pool: List<Int>, width: Int, height: Int): Int? {
        val before = identifyFrom ?: return null
        identifyFrom = null
        val shifts = before.mapNotNull { old ->
            val oldX = (old.left + old.right) / 2f
            val oldY = (old.top + old.bottom) / 2f
            val reach = max(old.width, old.height) * FOLLOW_BODIES
            pool
                .map { it to Vec((boxes[it].centerX - oldX) * width, (boxes[it].centerY - oldY) * height) }
                .filter { hypot(it.second.x / width, it.second.y / height) <= reach }
                .minByOrNull { it.second.length }
                ?.let { (index, shift) -> Triple(index, shift, max(old.width * width, old.height * height)) }
        }
        if (shifts.isEmpty()) return null
        val camera = if (shifts.size >= MIN_BOXES_FOR_CAMERA_SHIFT) {
            Vec(median(shifts.map { it.second.x }), median(shifts.map { it.second.y }))
        } else {
            Vec(0f, 0f)
        }
        val best = shifts
            .map { (index, shift, size) -> Triple(index, Vec(shift.x - camera.x, shift.y - camera.y), size) }
            .filter { (_, own, size) -> own.length >= max(MIN_MOVE_PX, size * IDENTIFY_MOVE_BODY) }
            .maxByOrNull { it.second.length }
        if (best == null) {
            stalls += 1
            return null
        }
        stalls = 0
        identifyTries = 0
        heading = best.second.unit()
        probeFrom = null
        return best.first
    }

    /** No robot yet: step forward and see which box moves, or report what is missing. */
    private fun lookForRobot(width: Int, height: Int, boxes: List<YoloBox>, picked: Picked): ApproachFrame {
        probeFrom = null
        if (picked.pool.isEmpty()) {
            robotMissed += 1
            identifyFrom = null
            return frame(width, height, boxes, picked, null, "Can't see the dog.", gaveUp = robotMissed >= GIVE_UP_FRAMES)
        }
        if (identifyTries >= MAX_IDENTIFY_TRIES) {
            return frame(
                width, height, boxes, picked, Decision(STOP, 0L),
                "Couldn't tell which one is my robot. Is it connected and in view?", gaveUp = true,
            )
        }
        identifyTries += 1
        actions += 1
        identifyFrom = picked.pool.map { boxes[it].rect() }
        val probe = if (identifyTries > 2) "kwkF 3" else "kwkF 2"
        return frame(
            width, height, boxes, picked, Decision(probe, 0L),
            "Taking a step to see which of ${picked.pool.size} is my robot.",
        )
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

    /**
     * YOLO flickers: a box vanishes for a frame, or the other same-class object scores higher.
     * So the target and the robot are locked once chosen. A locked one is only matched near
     * where it was, and a missed frame keeps its last box instead of choosing again.
     */
    private fun identify(bitmap: Bitmap, boxes: List<YoloBox>, targetLabel: String, color: String?): Picked {
        val targets = boxes.indices.filter { boxes[it].label == targetLabel }
        val lockedTarget = target
        var targetIndex = if (lockedTarget != null) {
            // Prefer the same object, but take the newest box so a moved can or bottle is followed.
            stayOn(boxes, targets, lockedTarget)
                .takeIf { it >= 0 }
                ?: rechoose(bitmap, boxes, targets, color, lockedTarget)
                ?: overlapAny(boxes, lockedTarget, color, bitmap)
        } else {
            choose(bitmap, boxes, targets, color)
        }
        if (targetIndex >= 0) {
            target = boxes[targetIndex].rect()
            targetMissed = 0
        } else if (lockedTarget != null) {
            targetMissed += 1
            if (targetMissed > TARGET_MEMORY_FRAMES) target = null
        }

        val heldTarget = target
        val robotPool = boxes.indices.filter { index ->
            index != targetIndex && boxes[index].label !in NEVER_ROBOT && isRobotSized(boxes[index]) &&
                (heldTarget == null || iou(boxes[index].rect(), heldTarget) < TARGET_LOOKALIKE_IOU)
        }
        val robotIndex = follow(boxes, robotPool, robot).takeIf { it >= 0 }
            ?: follow(boxes, boxes.indices.filter { it != targetIndex && isRobotSized(boxes[it]) }, robot)
        if (robotIndex >= 0) {
            robot = boxes[robotIndex].rect()
            robotLost = 0
        } else if (robot != null) {
            robotLost += 1
            if (robotLost > ROBOT_MEMORY_FRAMES) {
                robot = null
                heading = null
                probeFrom = null
            }
        }
        val visibleRobot = if (robotIndex >= 0) robot else null
        return Picked(robotIndex, targetIndex, visibleRobot, target, robotPool)
    }

    /**
     * First pick of the target. Same class, different object (a yellow can and a clear water
     * bottle are both "bottle"): the spoken color decides, and a box without enough of that
     * color is never taken, even when it is the only one. Without a color, YOLO's surest box.
     */
    private fun choose(bitmap: Bitmap, boxes: List<YoloBox>, targets: List<Int>, color: String?): Int {
        if (color == null) return targets.maxByOrNull { boxes[it].score } ?: -1
        return targets
            .map { it to ColorMatch.share(bitmap, boxes[it], color) }
            .filter { it.second >= MIN_COLOR_SHARE }
            .maxByOrNull { it.second }
            ?.first ?: -1
    }

    private fun rechoose(bitmap: Bitmap, boxes: List<YoloBox>, targets: List<Int>, color: String?, near: Rect): Int {
        val nearX = (near.left + near.right) / 2f
        val nearY = (near.top + near.bottom) / 2f
        return targets
            .filter { color == null || ColorMatch.share(bitmap, boxes[it], color) >= MIN_COLOR_SHARE }
            .minByOrNull { hypot(boxes[it].centerX - nearX, boxes[it].centerY - nearY) }
            ?: -1
    }

    /**
     * YOLO often relabels the same object (can → cup, robot → person). A box sitting on the
     * locked place, with the spoken color if any, is still that object.
     */
    private fun overlapAny(boxes: List<YoloBox>, locked: Rect, color: String?, bitmap: Bitmap): Int {
        return boxes.indices
            .filter { iou(boxes[it].rect(), locked) >= TRACK_IOU }
            .filter { color == null || ColorMatch.share(bitmap, boxes[it], color) >= MIN_COLOR_SHARE }
            .maxByOrNull { iou(boxes[it].rect(), locked) }
            ?: -1
    }

    /** The locked target: overlap first, else a same-class box whose center barely moved (phone shake). */
    private fun stayOn(boxes: List<YoloBox>, targets: List<Int>, locked: Rect): Int {
        val overlapping = track(boxes, targets, locked)
        if (overlapping >= 0) return overlapping
        val lockedX = (locked.left + locked.right) / 2f
        val lockedY = (locked.top + locked.bottom) / 2f
        val reach = max(locked.width, locked.height) * TARGET_SHIFT_SIZES
        return targets
            .map { it to hypot(boxes[it].centerX - lockedX, boxes[it].centerY - lockedY) }
            .filter { it.second <= reach }
            .minByOrNull { it.second }
            ?.first ?: -1
    }

    /**
     * The robot moves up to a stride between looks, so overlap alone loses it. Overlap wins;
     * otherwise the nearest box within [FOLLOW_BODIES] body lengths of where it was.
     */
    private fun follow(boxes: List<YoloBox>, indices: List<Int>, last: Rect?): Int {
        if (last == null) return -1
        val overlapping = track(boxes, indices, last)
        if (overlapping >= 0) return overlapping
        val lastX = (last.left + last.right) / 2f
        val lastY = (last.top + last.bottom) / 2f
        // Each unseen frame the robot may have walked on, so the search widens.
        val reach = max(last.width, last.height) * (FOLLOW_BODIES + robotLost * LOST_REACH_BODIES)
        val lastArea = last.width * last.height
        return indices
            .filter { index ->
                // Another robot or a can passing close by is a different size; ours barely changes.
                val ratio = (boxes[index].right - boxes[index].left) * (boxes[index].bottom - boxes[index].top) / max(lastArea, 1e-6f)
                ratio in MIN_ROBOT_AREA_RATIO..MAX_ROBOT_AREA_RATIO
            }
            .map { it to hypot(boxes[it].centerX - lastX, boxes[it].centerY - lastY) }
            .filter { it.second <= reach }
            .minByOrNull { it.second }
            ?.first ?: -1
    }

    private fun isRobotSized(box: YoloBox): Boolean {
        val area = (box.right - box.left) * (box.bottom - box.top)
        return area <= MAX_ROBOT_AREA
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
            PlayBox(if (role == "dog") "robot (${box.label})" else box.label, role, box.left, box.top, box.right, box.bottom, box.score)
        }.toMutableList()
        val remembered = picked.target
        if (picked.targetIndex < 0 && remembered != null) {
            drawn += PlayBox("target (last seen)", "target", remembered.left, remembered.top, remembered.right, remembered.bottom, 0f)
        }
        val lostRobot = robot
        if (picked.robotIndex < 0 && lostRobot != null) {
            drawn += PlayBox("robot (last seen)", "dog", lostRobot.left, lostRobot.top, lostRobot.right, lostRobot.bottom, 0f)
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
                "stalls=$stalls actions=$actions cmd=${decision?.command ?: "-"} hold=${decision?.turnHoldMs ?: 0} $note " +
                "boxes=${boxes.joinToString(",") { "${it.label}:${String.format(Locale.US, "%.2f", it.score)}" }}",
        )
    }

    private data class Decision(val command: String, val turnHoldMs: Long)

    /** [pool] is every box that could be the robot: not the target, not a person or furniture. */
    private data class Picked(val robotIndex: Int, val targetIndex: Int, val robot: Rect?, val target: Rect?, val pool: List<Int>)

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
        /**
         * Furniture is in view but is never the robot. "person" stays possible: YOLO has called
         * the Quaddle a person, and the motion check rejects real feet that stand still.
         */
        private val NEVER_ROBOT = setOf(
            "chair", "couch", "bed", "dining table", "tv", "potted plant", "refrigerator",
            "oven", "sink", "toilet", "laptop", "bench",
        )
        /** The probe step must move a box this share of its size to count, beyond the phone's own shake. */
        private const val IDENTIFY_MOVE_BODY = 0.25f
        private const val MAX_IDENTIFY_TRIES = 4
        private const val MIN_BOXES_FOR_CAMERA_SHIFT = 3
        /** A share of the frame; anything bigger is furniture or floor, not the robot. */
        private const val MAX_ROBOT_AREA = 0.35f
        /** A stride plus some slack: the robot walks between looks, so its box must still match. */
        private const val FOLLOW_BODIES = 3.0f
        private const val TRACK_IOU = 0.15f
        /** A locked target may drift this share of its size and still be the same object. */
        private const val TARGET_SHIFT_SIZES = 1.5f
        /** About 30 s of looks; the target stands still, so its last box stays good that long. */
        private const val TARGET_MEMORY_FRAMES = 30
        /** A pool box overlapping the locked target this much is the target under another label. */
        private const val TARGET_LOOKALIKE_IOU = 0.5f
        /** The robot is held through this many missed looks before it must be found again. */
        private const val ROBOT_MEMORY_FRAMES = 4
        /** Extra search radius, in body lengths, per frame the robot went unseen. */
        private const val LOST_REACH_BODIES = 1.0f
        /** Following by distance: the next robot box must be this close to its last size. */
        private const val MIN_ROBOT_AREA_RATIO = 0.3f
        private const val MAX_ROBOT_AREA_RATIO = 3.5f
        /** Share of a box's middle that must be the spoken color before it can be the target. */
        private const val MIN_COLOR_SHARE = 0.12f
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

        private fun median(values: List<Float>): Float {
            val sorted = values.sorted()
            val middle = sorted.size / 2
            return if (sorted.size % 2 == 1) sorted[middle] else (sorted[middle - 1] + sorted[middle]) / 2f
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
