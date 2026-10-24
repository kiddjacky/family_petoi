package com.gemmaproj.voicechat.ui

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.ime
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBars
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.PhotoCamera
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.gestures.waitForUpOrCancellation
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.gemmaproj.voicechat.domain.ChatMessage
import com.gemmaproj.voicechat.domain.GemmaVariant
import com.gemmaproj.voicechat.domain.SttEngine
import com.gemmaproj.voicechat.domain.VoiceStack

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(viewModel: ChatViewModel) {
    val state by viewModel.uiState.collectAsStateWithLifecycle()
    var draft by remember { mutableStateOf("") }
    var showSettings by remember { mutableStateOf(false) }
    val listState = rememberLazyListState()
    val context = LocalContext.current
    val cameraPermission = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted ->
        if (granted) viewModel.openCamera() else viewModel.noteSceneError("Camera permission is required")
    }
    val density = LocalDensity.current
    val imeBottom = WindowInsets.ime.getBottom(density)
    // OnePlus reports the IME inset 1 status-bar short of the drawn keyboard.
    val keyboardCorrection = if (imeBottom > 0) {
        with(density) { WindowInsets.statusBars.getTop(density).toDp() }
    } else {
        0.dp
    }

    LaunchedEffect(state.messages.size) {
        if (state.messages.isNotEmpty()) {
            listState.animateScrollToItem(state.messages.lastIndex)
        }
    }

    Box(modifier = Modifier.fillMaxSize()) {
    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text("Gemma Voice")
                        Text(
                            text = subtitle(state),
                            style = MaterialTheme.typography.labelMedium,
                        )
                    }
                },
                actions = {
                    IconButton(
                        onClick = {
                            val granted = ContextCompat.checkSelfPermission(
                                context,
                                Manifest.permission.CAMERA,
                            ) == PackageManager.PERMISSION_GRANTED
                            if (granted) viewModel.openCamera() else cameraPermission.launch(Manifest.permission.CAMERA)
                        },
                        enabled = state.isReady && !state.isBusy,
                    ) {
                        Icon(Icons.Default.PhotoCamera, contentDescription = "Camera")
                    }
                    IconButton(onClick = { showSettings = !showSettings }) {
                        Icon(Icons.Default.Settings, contentDescription = "Settings")
                    }
                },
            )
        },
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .imePadding()
                .padding(bottom = keyboardCorrection)
                .padding(horizontal = 12.dp),
        ) {
            if (!state.isReady) {
                if (state.isBusy) {
                    LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                }
                Text(
                    text = state.status,
                    style = MaterialTheme.typography.bodyMedium,
                    modifier = Modifier.padding(vertical = 8.dp),
                )
                if (!state.isBusy) {
                    Button(
                        onClick = viewModel::loadModels,
                        modifier = Modifier.fillMaxWidth(),
                    ) {
                        Text("Retry")
                    }
                }
            } else if (state.status != "Ready") {
                Text(
                    text = state.status,
                    style = MaterialTheme.typography.bodySmall,
                    modifier = Modifier.padding(bottom = 8.dp),
                )
            }
            if (showSettings) {
                SettingsPanel(state, viewModel)
            }
            LazyColumn(
                state = listState,
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth(),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                items(state.messages, key = { it.id }) { msg ->
                    MessageBubble(msg)
                }
            }
            if (state.isRecording) {
                Text(
                    text = "Listening — release to send",
                    color = MaterialTheme.colorScheme.onErrorContainer,
                    modifier = Modifier
                        .fillMaxWidth()
                        .background(MaterialTheme.colorScheme.errorContainer)
                        .padding(horizontal = 12.dp, vertical = 8.dp),
                    style = MaterialTheme.typography.bodyMedium,
                )
            }
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 8.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                HoldMicButton(
                    enabled = state.isReady && !state.isBusy &&
                        (state.voiceReady || state.prefs.voiceStack == VoiceStack.NATIVE),
                    isRecording = state.isRecording,
                    diameter = 84.dp,
                    onPress = viewModel::onMicPress,
                    onRelease = viewModel::onMicRelease,
                )
                OutlinedTextField(
                    value = draft,
                    onValueChange = { draft = it },
                    modifier = Modifier.weight(1f),
                    enabled = state.isReady && !state.isBusy && !state.isRecording,
                    placeholder = {
                        Text(
                            if (state.voiceReady || state.prefs.voiceStack == VoiceStack.NATIVE) {
                                "Message or hold the mic"
                            } else {
                                "Message Gemma…"
                            },
                        )
                    },
                    singleLine = false,
                    maxLines = 4,
                )
                IconButton(
                    onClick = {
                        viewModel.sendText(draft)
                        draft = ""
                    },
                    enabled = state.isReady && !state.isBusy && !state.isRecording && draft.isNotBlank(),
                ) {
                    Icon(Icons.AutoMirrored.Filled.Send, contentDescription = "Send")
                }
            }
        }
    }
        if (state.cameraOpen) {
            SceneCamera(state, viewModel)
        }
    }
}

@Composable
private fun SettingsPanel(state: ChatUiState, viewModel: ChatViewModel) {
    Column(modifier = Modifier.padding(bottom = 8.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            FilterChip(
                selected = state.prefs.gemma == GemmaVariant.E2B,
                onClick = { viewModel.setGemma(GemmaVariant.E2B) },
                label = { Text("Gemma 4 E2B") },
            )
            FilterChip(
                selected = state.prefs.gemma == GemmaVariant.E4B,
                onClick = { viewModel.setGemma(GemmaVariant.E4B) },
                label = { Text("Gemma 4 E4B") },
            )
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            FilterChip(
                selected = state.prefs.voiceStack == VoiceStack.PIPELINE,
                onClick = { viewModel.setVoiceStack(VoiceStack.PIPELINE) },
                label = { Text("SenseVoice + Kokoro") },
            )
            FilterChip(
                selected = state.prefs.voiceStack == VoiceStack.NATIVE,
                onClick = { viewModel.setVoiceStack(VoiceStack.NATIVE) },
                label = { Text("Gemma audio") },
            )
        }
        if (state.prefs.voiceStack == VoiceStack.PIPELINE) {
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                FilterChip(
                    selected = state.prefs.sttEngine == SttEngine.SENSE_VOICE,
                    onClick = { viewModel.setSttEngine(SttEngine.SENSE_VOICE) },
                    label = { Text("SenseVoice") },
                )
                FilterChip(
                    selected = state.prefs.sttEngine == SttEngine.WHISPER_TINY_EN,
                    onClick = { viewModel.setSttEngine(SttEngine.WHISPER_TINY_EN) },
                    label = { Text("Whisper") },
                )
            }
            FilterChip(
                selected = state.prefs.speakReplies,
                onClick = { viewModel.setSpeakReplies(!state.prefs.speakReplies) },
                label = { Text("Speak replies (Kokoro)") },
            )
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("Voice ${state.prefs.kokoroSpeakerId}", style = MaterialTheme.typography.bodyMedium)
                TextButton(onClick = { viewModel.setKokoroSpeaker(state.prefs.kokoroSpeakerId - 1) }) {
                    Text("−")
                }
                TextButton(onClick = { viewModel.setKokoroSpeaker(state.prefs.kokoroSpeakerId + 1) }) {
                    Text("+")
                }
            }
        } else {
            Text(
                "Hold the mic. Gemma hears the recording itself, without SenseVoice. " +
                    "The first line of the reply is what it heard.",
                style = MaterialTheme.typography.bodySmall,
            )
        }
    }
}

@Composable
private fun subtitle(state: ChatUiState): String {
    if (!state.isReady) return state.status
    val model = state.modelLabel
    return when {
        state.prefs.voiceStack == VoiceStack.NATIVE -> "$model · Gemma audio + Kokoro"
        state.visionReady && state.voiceReady -> "$model · SenseVoice + camera"
        state.visionReady -> "$model · camera"
        state.voiceReady -> "$model · SenseVoice + Kokoro"
        else -> "$model · text"
    }
}

@Composable
private fun MessageBubble(message: ChatMessage) {
    val isUser = message.role == ChatMessage.Role.USER
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = if (isUser) Arrangement.End else Arrangement.Start,
    ) {
        Card(
            modifier = Modifier.fillMaxWidth(0.86f),
            colors = CardDefaults.cardColors(
                containerColor = if (isUser) {
                    MaterialTheme.colorScheme.primaryContainer
                } else {
                    MaterialTheme.colorScheme.surfaceVariant
                },
            ),
        ) {
            Text(
                text = message.text.ifEmpty { "…" },
                modifier = Modifier.padding(12.dp),
                style = MaterialTheme.typography.bodyMedium,
            )
        }
    }
}

@Composable
internal fun HoldMicButton(
    enabled: Boolean,
    isRecording: Boolean,
    onPress: () -> Unit,
    onRelease: () -> Unit,
    diameter: androidx.compose.ui.unit.Dp = 84.dp,
    prominent: Boolean = false,
) {
    val color = when {
        isRecording -> Color(0xFFC62828)
        !enabled -> MaterialTheme.colorScheme.surfaceVariant
        prominent -> Color.White
        else -> MaterialTheme.colorScheme.primary
    }
    val iconTint = when {
        isRecording -> Color.White
        !enabled -> MaterialTheme.colorScheme.onSurfaceVariant
        prominent -> Color.Black
        else -> Color.White
    }
    Box(
        modifier = Modifier
            .size(diameter)
            .clip(CircleShape)
            .background(color)
            .pointerInput(enabled) {
                if (!enabled) return@pointerInput
                awaitEachGesture {
                    awaitFirstDown(requireUnconsumed = false)
                    try {
                        onPress()
                        waitForUpOrCancellation()
                    } finally {
                        onRelease()
                    }
                }
            },
        contentAlignment = Alignment.Center,
    ) {
        Icon(
            Icons.Default.Mic,
            contentDescription = "Hold to talk",
            tint = iconTint,
            modifier = Modifier.size(diameter * 0.42f),
        )
    }
}
