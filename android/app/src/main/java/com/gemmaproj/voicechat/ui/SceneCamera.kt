package com.gemmaproj.voicechat.ui

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.view.WindowManager
import com.gemmaproj.voicechat.engine.WalkOrders
import androidx.activity.ComponentActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.view.CameraController
import androidx.camera.view.LifecycleCameraController
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilterChipDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.max
import kotlinx.coroutines.CompletableDeferred

@Composable
fun SceneCamera(state: ChatUiState, viewModel: ChatViewModel) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val mainExecutor = remember(context) { ContextCompat.getMainExecutor(context) }
    val pendingPhoto = remember { AtomicReference<CompletableDeferred<File>?>(null) }
    var cameraError by remember { mutableStateOf<String?>(null) }
    var previewStreaming by remember { mutableStateOf(false) }
    val analysisExecutor = remember { Executors.newSingleThreadExecutor() }
    val controller = remember {
        LifecycleCameraController(context).apply {
            setEnabledUseCases(CameraController.IMAGE_CAPTURE or CameraController.IMAGE_ANALYSIS)
            imageCaptureMode = ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY
            imageAnalysisBackpressureStrategy = ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST
            // The walk tiles each frame for YOLO; 640×480 leaves objects across the room too small to find.
            setImageAnalysisTargetSize(CameraController.OutputSize(android.util.Size(1280, 960)))
            cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
        }
    }
    val captured = remember(state.scenePhotoPath) {
        val path = state.scenePhotoPath
        if (path.isBlank()) null else BitmapFactory.decodeFile(path)?.asImageBitmap()
    }

    DisposableEffect(lifecycleOwner, controller) {
        val activity = context as? ComponentActivity
        activity?.window?.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        try {
            controller.bindToLifecycle(lifecycleOwner)
        } catch (error: Exception) {
            try {
                controller.cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA
                controller.bindToLifecycle(lifecycleOwner)
            } catch (front: Exception) {
                cameraError = front.message ?: "Camera failed to open"
            }
        }
        onDispose {
            activity?.window?.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            pendingPhoto.getAndSet(null)?.cancel()
            controller.clearImageAnalysisAnalyzer()
            controller.unbind()
            analysisExecutor.shutdown()
        }
    }

    val analyzing = state.trackLive || state.approachLive
    LaunchedEffect(analyzing) {
        if (!analyzing) {
            controller.clearImageAnalysisAnalyzer()
            return@LaunchedEffect
        }
        controller.setImageAnalysisAnalyzer(analysisExecutor) { image ->
            try {
                viewModel.offerLiveFrame(uprightFrame(image))
            } catch (error: Exception) {
                viewModel.noteSceneError(error.message ?: "Camera frame failed")
            } finally {
                image.close()
            }
        }
    }

    fun pressTalk() {
        val deferred = CompletableDeferred<File>()
        pendingPhoto.getAndSet(deferred)?.cancel()
        val dir = File(context.cacheDir, "scene").apply { mkdirs() }
        val file = File(dir, "shot-${System.currentTimeMillis()}.jpg")
        controller.takePicture(
            ImageCapture.OutputFileOptions.Builder(file).build(),
            mainExecutor,
            object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                    if (!deferred.isCompleted) deferred.complete(file)
                }

                override fun onError(exception: ImageCaptureException) {
                    if (!deferred.isCompleted) deferred.completeExceptionally(exception)
                }
            },
        )
        viewModel.onDirectionPress()
    }

    fun releaseTalk() {
        viewModel.onDirectionRelease(pendingPhoto.getAndSet(null))
    }

    var capturing by remember { mutableStateOf(false) }

    fun captureForLocate() {
        if (capturing || state.isBusy || state.isRecording || cameraError != null) return
        capturing = true
        val dir = File(context.cacheDir, "scene").apply { mkdirs() }
        val file = File(dir, "find-${System.currentTimeMillis()}.jpg")
        controller.takePicture(
            ImageCapture.OutputFileOptions.Builder(file).build(),
            mainExecutor,
            object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                    capturing = false
                    viewModel.locatePhoto(file)
                }

                override fun onError(exception: ImageCaptureException) {
                    capturing = false
                    file.delete()
                    viewModel.noteSceneError(exception.message ?: "Photo failed")
                }
            },
        )
    }

    Box(modifier = Modifier.fillMaxSize().background(Color.Black)) {
        AndroidView(
            modifier = Modifier.fillMaxSize(),
            factory = { ctx ->
                PreviewView(ctx).apply {
                    // TextureView. The default SurfaceView draws behind Compose and looks blank.
                    implementationMode = PreviewView.ImplementationMode.COMPATIBLE
                    scaleType = PreviewView.ScaleType.FILL_CENTER
                    this.controller = controller
                    previewStreamState.observe(lifecycleOwner) { stream ->
                        previewStreaming = stream == PreviewView.StreamState.STREAMING
                    }
                }
            },
        )
        TrackOverlay(state)
        if (!previewStreaming && cameraError == null) {
            Text(
                text = "Starting camera…",
                color = Color.White,
                modifier = Modifier.align(Alignment.Center),
                style = MaterialTheme.typography.titleMedium,
            )
        }
        if (captured != null) {
            Image(
                bitmap = captured,
                contentDescription = "Photo sent to Gemma",
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .statusBarsPadding()
                    .padding(16.dp)
                    .size(108.dp)
                    .clip(RoundedCornerShape(12.dp)),
                contentScale = ContentScale.Crop,
            )
        }
        IconButton(
            onClick = viewModel::closeCamera,
            modifier = Modifier
                .align(Alignment.TopStart)
                .statusBarsPadding()
                .padding(12.dp),
        ) {
            Icon(Icons.Default.Close, contentDescription = "Close camera", tint = Color.White)
        }
        Column(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .background(Color(0xB3000000))
                .navigationBarsPadding()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            if (cameraError != null) {
                Text(
                    cameraError ?: "",
                    color = Color.White,
                    style = MaterialTheme.typography.bodyLarge,
                    textAlign = TextAlign.Center,
                    modifier = Modifier.fillMaxWidth(),
                )
            }
            if (state.sceneQuestion.isNotBlank()) {
                Text(
                    text = state.sceneQuestion,
                    color = Color(0xFFB0BEC5),
                    style = MaterialTheme.typography.bodyMedium,
                    textAlign = TextAlign.Center,
                    modifier = Modifier.fillMaxWidth(),
                )
            }
            Text(
                text = state.sceneCaption.ifBlank {
                    if (state.visionReady || state.trackReady) {
                        "Hold to talk: \"go to the cup\", \"now the shoe\", \"stop\". Keep the dog and the target in view."
                    } else {
                        state.status
                    }
                },
                color = Color.White,
                style = MaterialTheme.typography.headlineSmall,
                textAlign = TextAlign.Center,
                modifier = Modifier.fillMaxWidth(),
            )
            if (state.trackBoxes.isNotEmpty() || state.isBusy || state.isRecording) {
                Text(
                    text = state.status,
                    color = Color(0xFFB0BEC5),
                    style = MaterialTheme.typography.bodySmall,
                    textAlign = TextAlign.Center,
                    modifier = Modifier.fillMaxWidth(),
                )
            }
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .horizontalScroll(rememberScrollState()),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                for ((label, query) in APPROACH_TARGETS) {
                    TargetChip(
                        label = label,
                        selected = WalkOrders.targetLabel(state.trackTarget) == query,
                        enabled = true,
                        onClick = { viewModel.setTrackTarget(query) },
                    )
                }
            }
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .horizontalScroll(rememberScrollState()),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Button(
                    onClick = { viewModel.setApproach(!state.approachLive) },
                    enabled = state.approachLive ||
                        (previewStreaming && cameraError == null && !state.isRecording && !state.isBusy),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = if (state.approachLive) Color(0xFFC62828) else Color(0xFF2E7D32),
                    ),
                ) {
                    Text(
                        if (state.approachLive) {
                            "Stop"
                        } else {
                            "Go to ${listOfNotNull(state.trackColor, targetLabel(state.trackTarget)).joinToString(" ")}"
                        },
                    )
                }
                OutlinedButton(
                    onClick = viewModel::restartDemo,
                    enabled = !state.isRecording,
                    colors = ButtonDefaults.outlinedButtonColors(contentColor = Color.White),
                ) {
                    Text("Restart")
                }
                TargetChip(
                    label = "Live",
                    selected = state.trackLive,
                    enabled = previewStreaming && cameraError == null && !state.isRecording,
                    onClick = { viewModel.setTrackLive(!state.trackLive) },
                )
                Button(
                    onClick = ::captureForLocate,
                    enabled = previewStreaming && !state.isBusy && !state.isRecording && cameraError == null,
                ) {
                    Text("Find")
                }
            }
            if (!state.visionReady && !state.isBusy) {
                Button(onClick = viewModel::openCamera, modifier = Modifier.fillMaxWidth()) {
                    Text("Retry image model")
                }
            }
            HoldMicButton(
                enabled = state.visionReady && state.voiceReady && !state.isBusy,
                isRecording = state.isRecording,
                onPress = ::pressTalk,
                onRelease = ::releaseTalk,
                diameter = 136.dp,
                prominent = true,
            )
            Text(
                text = "Hold to talk",
                color = Color.White,
                style = MaterialTheme.typography.labelLarge,
            )
        }
    }
}

private const val MAX_FRAME_EDGE = 1280

private fun uprightFrame(image: ImageProxy): Bitmap {
    val raw = image.toBitmap()
    val rotation = image.imageInfo.rotationDegrees
    val rotated = if (rotation == 0) {
        raw
    } else {
        val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
        val turned = Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, matrix, true)
        if (turned !== raw) raw.recycle()
        turned
    }
    val longEdge = max(rotated.width, rotated.height)
    if (longEdge <= MAX_FRAME_EDGE) return rotated
    val scale = MAX_FRAME_EDGE.toFloat() / longEdge
    val width = (rotated.width * scale).toInt().coerceAtLeast(1)
    val height = (rotated.height * scale).toInt().coerceAtLeast(1)
    val scaled = Bitmap.createScaledBitmap(rotated, width, height, true)
    if (scaled !== rotated) rotated.recycle()
    return scaled
}

/** YOLO class names; the walk finds targets by YOLO class only. */
private val APPROACH_TARGETS = listOf(
    "Bottle" to "bottle",
    "Cup" to "cup",
    "Ball" to "sports ball",
    "Bowl" to "bowl",
    "Teddy" to "teddy bear",
)

private fun targetLabel(query: String): String =
    WalkOrders.targetLabel(query) ?: query.removePrefix("a ")

@Composable
private fun TargetChip(label: String, selected: Boolean, enabled: Boolean, onClick: () -> Unit) {
    FilterChip(
        selected = selected,
        onClick = onClick,
        enabled = enabled,
        label = { Text(label) },
        colors = FilterChipDefaults.filterChipColors(
            containerColor = Color(0xFF263238),
            labelColor = Color.White,
            selectedContainerColor = Color(0xFFFFC107),
            selectedLabelColor = Color.Black,
            disabledContainerColor = Color(0xFF1C2529),
            disabledLabelColor = Color(0xFF78909C),
        ),
    )
}

@Composable
private fun TrackOverlay(state: ChatUiState) {
    if (state.trackImageWidth <= 0 || state.trackImageHeight <= 0) return
    if (state.trackBoxes.isEmpty() && state.headingLine.size != 4) return
    Canvas(modifier = Modifier.fillMaxSize()) {
        val imageWidth = state.trackImageWidth.toFloat()
        val imageHeight = state.trackImageHeight.toFloat()
        val scale = max(size.width / imageWidth, size.height / imageHeight)
        val displayedWidth = imageWidth * scale
        val displayedHeight = imageHeight * scale
        val originX = (size.width - displayedWidth) / 2f
        val originY = (size.height - displayedHeight) / 2f
        val labelPaint = android.graphics.Paint().apply {
            color = android.graphics.Color.WHITE
            textSize = 16.dp.toPx()
            isAntiAlias = true
        }
        for (box in state.trackBoxes) {
            val color = when (box.role) {
                "dog" -> Color(0xFF66BB6A)
                "person" -> Color(0xFF42A5F5)
                "other" -> Color(0x66FFFFFF)
                else -> Color(0xFFFFC107)
            }
            val left = originX + box.left * displayedWidth
            val top = originY + box.top * displayedHeight
            val boxSize = Size(
                (box.right - box.left) * displayedWidth,
                (box.bottom - box.top) * displayedHeight,
            )
            drawRect(color = color, topLeft = Offset(left, top), size = boxSize, style = Stroke(width = 6.dp.toPx()))
            drawContext.canvas.nativeCanvas.drawText(box.label, left + 8.dp.toPx(), top + 22.dp.toPx(), labelPaint)
        }
        val line = state.headingLine
        if (line.size == 4) {
            drawLine(
                color = Color(0xFF00E5FF),
                start = Offset(originX + line[0] * displayedWidth, originY + line[1] * displayedHeight),
                end = Offset(originX + line[2] * displayedWidth, originY + line[3] * displayedHeight),
                strokeWidth = 6.dp.toPx(),
            )
        }
    }
}
