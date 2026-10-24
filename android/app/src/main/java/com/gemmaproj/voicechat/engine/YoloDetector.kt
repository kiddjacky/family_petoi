package com.gemmaproj.voicechat.engine

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min
import org.tensorflow.lite.Interpreter

/**
 * YOLO26n boxes for one photo. The class name is only a hint. EmbeddingGemma
 * decides which box is the robot and which box is the object to walk toward.
 *
 * The LiteRT export is NCHW float [1, 3, 640, 640] in and [1, 84, 8400] out.
 * The 84 channels are cx, cy, w, h in 0..1 of the letterboxed canvas, then 80
 * class scores. Android runs the NMS.
 */
internal class YoloDetector {
    private var interpreter: Interpreter? = null
    private var loadedPath: String? = null
    private var inputBuffer: ByteBuffer? = null
    private var pixels: IntArray? = null
    private var letterCanvas: Bitmap? = null
    private var output: Array<Array<FloatArray>>? = null

    fun ensure(model: File) = synchronized(this) {
        if (interpreter != null && loadedPath == model.absolutePath) return
        if (!model.isFile) error("Missing ${model.name}. YOLO26 is not on this phone.")
        interpreter?.close()
        val options = Interpreter.Options().apply { numThreads = 4 }
        interpreter = Interpreter(model, options)
        loadedPath = model.absolutePath
    }

    fun detect(source: Bitmap, encodeCrops: Boolean = true): List<YoloBox> = synchronized(this) {
        val engine = interpreter ?: error("YOLO is not loaded")
        val inputShape = engine.getInputTensor(0).shape()
        val outputShape = engine.getOutputTensor(0).shape()
        if (inputShape.size != 4 || inputShape[1] != 3 || inputShape[2] != INPUT || inputShape[3] != INPUT) {
            error("Unexpected YOLO input ${inputShape.contentToString()}")
        }
        if (outputShape.size != 3 || outputShape[1] != 4 + COCO_NAMES.size || outputShape[2] != ANCHORS) {
            error("Unexpected YOLO output ${outputShape.contentToString()}")
        }
        val letter = letterbox(source)
        val input = scratchInput()
        val pixelBuffer = scratchPixels()
        letter.bitmap.getPixels(pixelBuffer, 0, INPUT, 0, 0, INPUT, INPUT)
        for (channel in 0 until 3) {
            val shift = when (channel) {
                0 -> 16
                1 -> 8
                else -> 0
            }
            for (pixel in pixelBuffer) {
                input.putFloat(((pixel shr shift) and 0xFF) / 255f)
            }
        }
        input.rewind()
        val raw = scratchOutput(outputShape[1], outputShape[2])
        engine.run(input, raw)
        val candidates = decode(raw[0], letter, source.width, source.height)
        nms(candidates).take(MAX_BOXES).map { box ->
            if (encodeCrops) box.withCrop(source) else box
        }
    }

    fun close() = synchronized(this) {
        interpreter?.close()
        interpreter = null
        loadedPath = null
        letterCanvas?.recycle()
        letterCanvas = null
        inputBuffer = null
        pixels = null
        output = null
    }

    private fun scratchInput(): ByteBuffer {
        val existing = inputBuffer
        if (existing != null && existing.capacity() == 3 * INPUT * INPUT * 4) {
            existing.clear()
            return existing
        }
        val created = ByteBuffer.allocateDirect(3 * INPUT * INPUT * 4).order(ByteOrder.nativeOrder())
        inputBuffer = created
        return created
    }

    private fun scratchPixels(): IntArray {
        val existing = pixels
        if (existing != null && existing.size == INPUT * INPUT) return existing
        val created = IntArray(INPUT * INPUT)
        pixels = created
        return created
    }

    private fun scratchOutput(channels: Int, anchors: Int): Array<Array<FloatArray>> {
        val existing = output
        if (existing != null && existing.size == 1 && existing[0].size == channels && existing[0][0].size == anchors) {
            return existing
        }
        val created = Array(1) { Array(channels) { FloatArray(anchors) } }
        output = created
        return created
    }

    private fun decode(output: Array<FloatArray>, letter: Letterbox, imageWidth: Int, imageHeight: Int): List<YoloBox> {
        val found = ArrayList<YoloBox>()
        for (anchor in 0 until ANCHORS) {
            var bestScore = 0f
            var bestClass = 0
            for (classIndex in COCO_NAMES.indices) {
                val score = output[4 + classIndex][anchor]
                if (score > bestScore) {
                    bestScore = score
                    bestClass = classIndex
                }
            }
            if (bestScore < CONFIDENCE) continue
            val centerX = (output[0][anchor] * INPUT - letter.padX) / letter.scale
            val centerY = (output[1][anchor] * INPUT - letter.padY) / letter.scale
            val boxWidth = output[2][anchor] * INPUT / letter.scale
            val boxHeight = output[3][anchor] * INPUT / letter.scale
            val left = ((centerX - boxWidth / 2f) / imageWidth).coerceIn(0f, 1f)
            val top = ((centerY - boxHeight / 2f) / imageHeight).coerceIn(0f, 1f)
            val right = ((centerX + boxWidth / 2f) / imageWidth).coerceIn(0f, 1f)
            val bottom = ((centerY + boxHeight / 2f) / imageHeight).coerceIn(0f, 1f)
            if (right - left < 0.02f || bottom - top < 0.02f) continue
            found += YoloBox(
                label = COCO_NAMES[bestClass],
                score = bestScore,
                left = left,
                top = top,
                right = right,
                bottom = bottom,
                jpeg = ByteArray(0),
            )
        }
        return found
    }

    private fun nms(boxes: List<YoloBox>): List<YoloBox> {
        val ordered = boxes.sortedByDescending { it.score }
        val kept = ArrayList<YoloBox>(ordered.size)
        for (box in ordered) {
            if (kept.none { intersectionOverUnion(it, box) > IOU }) kept += box
        }
        return kept
    }

    private fun intersectionOverUnion(left: YoloBox, right: YoloBox): Float {
        val overlapW = max(0f, min(left.right, right.right) - max(left.left, right.left))
        val overlapH = max(0f, min(left.bottom, right.bottom) - max(left.top, right.top))
        val overlap = overlapW * overlapH
        val area = (left.right - left.left) * (left.bottom - left.top) +
            (right.right - right.left) * (right.bottom - right.top) - overlap
        if (area <= 0f) return 0f
        return overlap / area
    }

    private fun letterbox(source: Bitmap): Letterbox {
        val scale = min(INPUT.toFloat() / source.width, INPUT.toFloat() / source.height)
        val width = (source.width * scale).toInt().coerceAtLeast(1)
        val height = (source.height * scale).toInt().coerceAtLeast(1)
        val padX = (INPUT - width) / 2f
        val padY = (INPUT - height) / 2f
        val canvas = letterCanvas?.takeUnless { it.isRecycled }
            ?: Bitmap.createBitmap(INPUT, INPUT, Bitmap.Config.ARGB_8888).also { letterCanvas = it }
        val drawing = Canvas(canvas)
        drawing.drawColor(Color.rgb(114, 114, 114))
        val scaled = Bitmap.createScaledBitmap(source, width, height, true)
        drawing.drawBitmap(scaled, padX, padY, null)
        if (scaled !== source) scaled.recycle()
        return Letterbox(canvas, scale, padX, padY)
    }

    private data class Letterbox(val bitmap: Bitmap, val scale: Float, val padX: Float, val padY: Float)

    companion object {
        private const val INPUT = 640
        private const val ANCHORS = 8400
        private const val CONFIDENCE = 0.25f
        private const val IOU = 0.45f
        private const val MAX_BOXES = 8

        private val COCO_NAMES = arrayOf(
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
            "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
        )
    }
}

internal data class YoloBox(
    val label: String,
    val score: Float,
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float,
    val jpeg: ByteArray,
) {
    val centerX: Float get() = (left + right) / 2f
    val centerY: Float get() = (top + bottom) / 2f

    fun withCrop(source: Bitmap): YoloBox = copy(jpeg = jpegBytes(source))

    fun jpegBytes(source: Bitmap, quality: Int = 80): ByteArray {
        val padX = (right - left) * 0.08f
        val padY = (bottom - top) * 0.08f
        val cropLeft = ((left - padX) * source.width).toInt().coerceIn(0, source.width - 1)
        val cropTop = ((top - padY) * source.height).toInt().coerceIn(0, source.height - 1)
        val cropRight = ((right + padX) * source.width).toInt().coerceIn(cropLeft + 1, source.width)
        val cropBottom = ((bottom + padY) * source.height).toInt().coerceIn(cropTop + 1, source.height)
        val tile = Bitmap.createBitmap(source, cropLeft, cropTop, cropRight - cropLeft, cropBottom - cropTop)
        val encoded = ByteArrayOutputStream()
        try {
            if (!tile.compress(Bitmap.CompressFormat.JPEG, quality, encoded)) error("Could not encode a detection")
        } finally {
            tile.recycle()
        }
        return encoded.toByteArray()
    }
}
