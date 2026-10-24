package com.gemmaproj.voicechat.engine

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import com.google.ai.edge.litertlm.ActivationDataType
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.EmbeddingEngine
import com.google.ai.edge.litertlm.EmbeddingEngineConfig
import com.google.ai.edge.litertlm.EmbeddingOptions
import com.google.ai.edge.litertlm.InputData
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock
import kotlin.math.sqrt
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Finds a robot dog and a target in one photo.
 *
 * EmbeddingGemma maps text and images into the same vector, but it does not
 * return boxes. The photo is cut into a 3×3 grid, each crop is embedded, and
 * the crop closest to "a robot dog" and the crop closest to the target phrase
 * are the two locations. The approach command is the step from the dog's cell
 * to the target's cell. Higher in the frame is treated as farther away, which
 * is the usual ground-plane view from a camera looking forward.
 *
 * Gemma E2B is not called here. A full image answer takes a few seconds and
 * still would not give a position, so the dog's direction comes from this grid.
 */
internal class EmbeddingLocator(context: Context) {
    private val cacheDir = File(context.cacheDir, "embed").apply { mkdirs() }
    private val gate = ReentrantLock()
    private var engine: EmbeddingEngine? = null
    private var options = imageOptions(outputSize = OUTPUT_SIZE)
    private val textVectors = HashMap<String, FloatArray>()

    val isReady: Boolean get() = engine?.isInitialized() == true

    suspend fun open(
        model: File,
        useGpu: Boolean,
        visionGpu: Boolean = useGpu,
        activation: ActivationDataType? = null,
    ) = withContext(Dispatchers.IO) {
        gate.withLock {
            closeLocked()
            val created = EmbeddingEngine(
                EmbeddingEngineConfig(
                    modelPath = model.absolutePath,
                    backend = if (useGpu) Backend.GPU() else Backend.CPU(),
                    visionBackend = if (visionGpu) Backend.GPU() else Backend.CPU(),
                    cacheDir = cacheDir.absolutePath,
                    maxInputLength = MAX_TEXT_TOKENS,
                    visionTokensPerImage = VISION_TOKENS,
                    activationDataType = activation,
                ),
            )
            try {
                created.initialize()
            } catch (error: Exception) {
                created.close()
                throw error
            }
            engine = created
            textVectors.clear()
            options = imageOptions(outputSize = OUTPUT_SIZE)
            Log.i(TAG, "ready gpu=$useGpu visionGpu=$visionGpu activation=$activation")
        }
    }

    /**
     * One photo. The person is located, then the object list is scored on the
     * other cells. The winning object is what the dog is told to walk toward.
     * Crops are embedded once and compared with every phrase.
     */
    suspend fun locatePointed(
        jpeg: File,
        dogQuery: String,
        personQuery: String,
        objectQueries: List<String>,
    ): PointResult = withContext(Dispatchers.IO) {
        gate.withLock {
            val current = engine ?: error("EmbeddingGemma is not loaded")
            if (objectQueries.isEmpty()) error("No objects to search for")
            val started = System.nanoTime()
            val bitmap = BitmapFactory.decodeFile(jpeg.absolutePath)
                ?: error("Could not read the photo")
            try {
                val crops = cropsOf(bitmap)
                val vectors = embedCrops(current, crops)
                fun scoresFor(label: String): List<CropScore> {
                    val text = textVector(current, label)
                    return crops.zip(vectors) { crop, vector -> crop.toScore(cosine(vector, text)) }
                }
                val personScores = scoresFor(personQuery)
                val dogScores = scoresFor(dogQuery)
                val person = best(personScores)
                val dog = best(dogScores)
                val objects = objectQueries.map { name -> name to best(scoresFor(name)) }
                val awayFromPerson = objects.filter { (_, cell) ->
                    cell.col != person.col || cell.row != person.row
                }
                val pool = if (awayFromPerson.isNotEmpty()) awayFromPerson else objects
                val (targetName, target) = pool.maxBy { it.second.score }
                val targetScores = scoresFor(targetName)
                val unclear = margin(personScores, person) < CLEAR_MARGIN ||
                    margin(targetScores, target) < CLEAR_MARGIN
                val spoken = targetName.removePrefix("a ").removePrefix("an ")
                val command = when {
                    dog.col == target.col && dog.row == target.row -> "Stop. The dog is at the $spoken."
                    else -> {
                        val caution = if (unclear) "Unclear. " else ""
                        caution + "Walk to the $spoken. " + approachCommand(dog.col, dog.row, target.col, target.row)
                    }
                }
                PointResult(
                    person = person,
                    dog = dog,
                    target = target,
                    targetName = targetName,
                    command = command,
                    width = bitmap.width,
                    height = bitmap.height,
                    elapsedMs = (System.nanoTime() - started) / 1_000_000,
                    scores = objects.joinToString(" ") { (name, cell) ->
                        "${name.removePrefix("a ")}@${cell.col},${cell.row}=${"%.2f".format(cell.score)}"
                    },
                )
            } finally {
                bitmap.recycle()
            }
        }
    }

    /** Image vectors only. Text phrases are scored separately so a live loop can reuse a vector. */
    suspend fun embedImages(jpegs: List<ByteArray>): List<FloatArray> = withContext(Dispatchers.IO) {
        gate.withLock {
            val current = engine ?: error("EmbeddingGemma is not loaded")
            if (jpegs.isEmpty()) return@withLock emptyList()
            val crops = jpegs.map { bytes ->
                Crop(col = 0, row = 0, left = 0f, top = 0f, right = 1f, bottom = 1f, bytes = bytes)
            }
            embedCrops(current, crops)
        }
    }

    /** Cosine of each image vector against each label. Text vectors stay cached. */
    suspend fun scoreVectors(vectors: List<FloatArray>, labels: List<String>): Map<String, FloatArray> =
        withContext(Dispatchers.IO) {
            gate.withLock {
                val current = engine ?: error("EmbeddingGemma is not loaded")
                if (vectors.isEmpty() || labels.isEmpty()) return@withLock emptyMap()
                labels.associateWith { label ->
                    val text = textVector(current, label)
                    FloatArray(vectors.size) { index -> cosine(vectors[index], text) }
                }
            }
        }

    /** One vector per label. Each vector has one cosine score per crop, in crop order. */
    suspend fun scoreCrops(jpegs: List<ByteArray>, labels: List<String>): Map<String, FloatArray> =
        withContext(Dispatchers.IO) {
            gate.withLock {
                val current = engine ?: error("EmbeddingGemma is not loaded")
                if (jpegs.isEmpty() || labels.isEmpty()) return@withLock emptyMap()
                val crops = jpegs.map { bytes ->
                    Crop(col = 0, row = 0, left = 0f, top = 0f, right = 1f, bottom = 1f, bytes = bytes)
                }
                val vectors = embedCrops(current, crops)
                labels.associateWith { label ->
                    val text = textVector(current, label)
                    FloatArray(vectors.size) { index -> cosine(vectors[index], text) }
                }
            }
        }

    suspend fun locate(jpeg: File, dogQuery: String, targetQuery: String): LocateResult =
        withContext(Dispatchers.IO) {
            gate.withLock {
                val current = engine ?: error("EmbeddingGemma is not loaded")
                val started = System.nanoTime()
                val bitmap = BitmapFactory.decodeFile(jpeg.absolutePath)
                    ?: error("Could not read the photo")
                try {
                    val dogVector = textVector(current, dogQuery)
                    val targetVector = textVector(current, targetQuery)
                    val crops = cropsOf(bitmap)
                    val vectors = embedCrops(current, crops)
                    val dogScores = crops.zip(vectors) { crop, vector -> crop.toScore(cosine(vector, dogVector)) }
                    val targetScores = crops.zip(vectors) { crop, vector -> crop.toScore(cosine(vector, targetVector)) }
                    val dog = best(dogScores)
                    val target = best(targetScores)
                    val caution = if (margin(dogScores, dog) < CLEAR_MARGIN || margin(targetScores, target) < CLEAR_MARGIN) {
                        "Unclear. "
                    } else {
                        ""
                    }
                    val command = caution + approachCommand(dog.col, dog.row, target.col, target.row)
                    LocateResult(
                        dog = dog,
                        target = target,
                        command = command,
                        width = bitmap.width,
                        height = bitmap.height,
                        elapsedMs = (System.nanoTime() - started) / 1_000_000,
                    )
                } finally {
                    bitmap.recycle()
                }
            }
        }

    fun close() {
        gate.withLock { closeLocked() }
    }

    private fun embedCrops(current: EmbeddingEngine, crops: List<Crop>): List<FloatArray> {
        val batch = crops.map { listOf(InputData.Image(it.bytes)) }
        val responses = try {
            current.computeEmbeddingBatch(batch, options)
        } catch (batchError: Exception) {
            Log.w(TAG, "batch embed failed, retrying one crop at a time", batchError)
            if (options.outputSize != null) {
                options = imageOptions(outputSize = null)
            }
            crops.map { crop ->
                current.computeEmbedding(listOf(InputData.Image(crop.bytes)), options)
            }
        }
        if (responses.size != crops.size) error("EmbeddingGemma returned ${responses.size} crops, expected ${crops.size}")
        return responses.map { it.embedding.copyOf() }
    }

    private fun textVector(current: EmbeddingEngine, label: String): FloatArray {
        textVectors[cacheKey(label)]?.let { return it }
        val text = "task: search query | text: $label"
        val vector = try {
            current.computeEmbedding(listOf(InputData.Text(text)), options).embedding.copyOf()
        } catch (error: Exception) {
            if (options.outputSize == null) throw error
            Log.w(TAG, "sized text embed failed, dropping output size", error)
            options = imageOptions(outputSize = null)
            textVectors.clear()
            current.computeEmbedding(listOf(InputData.Text(text)), options).embedding.copyOf()
        }
        textVectors[cacheKey(label)] = vector
        return vector
    }

    private fun cacheKey(label: String): String = "$label|${options.outputSize}"

    private fun closeLocked() {
        textVectors.clear()
        try {
            engine?.close()
        } catch (error: Exception) {
            Log.w(TAG, "close failed: ${error.message}")
        }
        engine = null
    }

    private data class Crop(
        val col: Int,
        val row: Int,
        val left: Float,
        val top: Float,
        val right: Float,
        val bottom: Float,
        val bytes: ByteArray,
    ) {
        fun toScore(score: Float) = CropScore(col, row, left, top, right, bottom, score)
    }

    companion object {
        private const val TAG = "EmbeddingLocator"
        private const val COLS = 3
        private const val ROWS = 3
        private const val PAD = 0.08f
        private const val VISION_TOKENS = 70
        private const val MAX_TEXT_TOKENS = 128
        private const val OUTPUT_SIZE = 256
        private const val CLEAR_MARGIN = 0.015f

        private fun imageOptions(outputSize: Int?) = EmbeddingOptions(
            normalize = true,
            outputSize = outputSize,
            visionTokensPerImage = VISION_TOKENS,
        )

        private fun cropsOf(bitmap: Bitmap): List<Crop> {
            val cellW = bitmap.width / COLS.toFloat()
            val cellH = bitmap.height / ROWS.toFloat()
            val crops = ArrayList<Crop>(COLS * ROWS)
            for (row in 0 until ROWS) {
                for (col in 0 until COLS) {
                    val padX = cellW * PAD
                    val padY = cellH * PAD
                    val leftPx = ((col * cellW) - padX).toInt().coerceIn(0, bitmap.width - 1)
                    val topPx = ((row * cellH) - padY).toInt().coerceIn(0, bitmap.height - 1)
                    val rightPx = (((col + 1) * cellW) + padX).toInt().coerceIn(leftPx + 1, bitmap.width)
                    val bottomPx = (((row + 1) * cellH) + padY).toInt().coerceIn(topPx + 1, bitmap.height)
                    val tile = Bitmap.createBitmap(bitmap, leftPx, topPx, rightPx - leftPx, bottomPx - topPx)
                    try {
                        val encoded = ByteArrayOutputStream()
                        if (!tile.compress(Bitmap.CompressFormat.JPEG, 80, encoded)) {
                            error("Could not encode a crop")
                        }
                        crops += Crop(
                            col = col,
                            row = row,
                            left = col / COLS.toFloat(),
                            top = row / ROWS.toFloat(),
                            right = (col + 1) / COLS.toFloat(),
                            bottom = (row + 1) / ROWS.toFloat(),
                            bytes = encoded.toByteArray(),
                        )
                    } finally {
                        tile.recycle()
                    }
                }
            }
            return crops
        }

        private fun best(scores: List<CropScore>): CropScore {
            return scores.maxWith(
                compareBy<CropScore> { it.score }.thenByDescending { centerBias(it) },
            )
        }

        private fun centerBias(score: CropScore): Int {
            return -(kotlin.math.abs(score.col - 1) + kotlin.math.abs(score.row - 1))
        }

        private fun margin(scores: List<CropScore>, winner: CropScore): Float {
            val second = scores
                .filter { it.col != winner.col || it.row != winner.row }
                .maxOfOrNull { it.score }
                ?: winner.score
            return winner.score - second
        }

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

        /**
         * Camera-frame step from the dog cell to the target cell.
         * Column increases to the right. Row increases downward, so a target
         * in a smaller row is higher in the picture and counts as forward.
         */
        internal fun approachCommand(dogCol: Int, dogRow: Int, targetCol: Int, targetRow: Int): String {
            val dx = targetCol - dogCol
            val dy = targetRow - dogRow
            if (dx == 0 && dy == 0) return "Stop. The target is on the dog."
            val moves = ArrayList<String>(2)
            if (dx < 0) moves += "left"
            if (dx > 0) moves += "right"
            if (dy < 0) moves += "forward"
            if (dy > 0) moves += "back"
            return "Go ${moves.joinToString(" and ")}."
        }
    }
}

internal data class CropScore(
    val col: Int,
    val row: Int,
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float,
    val score: Float,
)

internal data class PointResult(
    val person: CropScore,
    val dog: CropScore,
    val target: CropScore,
    val targetName: String,
    val command: String,
    val width: Int,
    val height: Int,
    val elapsedMs: Long,
    val scores: String,
)

internal data class LocateResult(
    val dog: CropScore,
    val target: CropScore,
    val command: String,
    val width: Int,
    val height: Int,
    val elapsedMs: Long,
)
