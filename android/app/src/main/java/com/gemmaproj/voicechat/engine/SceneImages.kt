package com.gemmaproj.voicechat.engine

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.media.ExifInterface
import java.io.File
import java.io.FileOutputStream
import kotlin.math.max

/**
 * Camera JPEGs arrive with the rotation stored in EXIF, and the sensor frame
 * can be far larger than Gemma's vision encoder uses. This writes one upright
 * JPEG whose long edge is at most [LONG_EDGE].
 */
internal object SceneImages {
    const val LONG_EDGE = 1024

    fun normalize(source: File): File {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(source.absolutePath, bounds)
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) {
            error("Could not read the photo")
        }
        var sample = 1
        val sourceLongEdge = max(bounds.outWidth, bounds.outHeight)
        while (sourceLongEdge / sample > LONG_EDGE * 2) sample *= 2
        val decoded = BitmapFactory.decodeFile(
            source.absolutePath,
            BitmapFactory.Options().apply { inSampleSize = sample },
        ) ?: error("Could not read the photo")
        val upright = applyExif(source, decoded)
        val scaled = scaleToLongEdge(upright, LONG_EDGE)
        val dest = File(source.parentFile, "scene-${System.currentTimeMillis()}.jpg")
        FileOutputStream(dest).use { out ->
            if (!scaled.compress(Bitmap.CompressFormat.JPEG, 85, out)) {
                error("Could not save the photo")
            }
        }
        if (scaled !== upright) scaled.recycle()
        if (upright !== decoded) upright.recycle()
        decoded.recycle()
        if (source.absolutePath != dest.absolutePath) source.delete()
        trimOld(dest.parentFile, dest)
        return dest
    }

    private fun applyExif(source: File, bitmap: Bitmap): Bitmap {
        val orientation = ExifInterface(source.absolutePath).getAttributeInt(
            ExifInterface.TAG_ORIENTATION,
            ExifInterface.ORIENTATION_NORMAL,
        )
        val matrix = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.preScale(-1f, 1f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> matrix.preScale(1f, -1f)
            ExifInterface.ORIENTATION_TRANSPOSE -> {
                matrix.postRotate(90f)
                matrix.preScale(-1f, 1f)
            }
            ExifInterface.ORIENTATION_TRANSVERSE -> {
                matrix.postRotate(270f)
                matrix.preScale(-1f, 1f)
            }
            else -> return bitmap
        }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    private fun scaleToLongEdge(bitmap: Bitmap, longEdge: Int): Bitmap {
        val current = max(bitmap.width, bitmap.height)
        if (current <= longEdge) return bitmap
        val scale = longEdge.toFloat() / current.toFloat()
        val width = (bitmap.width * scale).toInt().coerceAtLeast(1)
        val height = (bitmap.height * scale).toInt().coerceAtLeast(1)
        return Bitmap.createScaledBitmap(bitmap, width, height, true)
    }

    private fun trimOld(dir: File?, keep: File) {
        val shots = dir?.listFiles { file ->
            file.isFile && file.name.startsWith("scene-") && file.name.endsWith(".jpg")
        } ?: return
        shots.sortedByDescending { it.lastModified() }
            .drop(4)
            .filter { it.absolutePath != keep.absolutePath }
            .forEach { it.delete() }
    }
}
