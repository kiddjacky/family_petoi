package com.gemmaproj.voicechat.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.KeyboardArrowLeft
import androidx.compose.material.icons.automirrored.filled.KeyboardArrowRight
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material.icons.filled.KeyboardArrowUp
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.gemmaproj.voicechat.engine.DogLink

private const val FORWARD = "kwkF 1"
private const val BACK = "kbkF 1"
private const val LEFT = "kwkL"
private const val RIGHT = "kwkR"
private const val STOP = "kbalance"

/** Remote control tab. The link lives in the application; Gemma and the voice live in [chat]. */
@Composable
fun DogScreen(link: DogLink, chat: ChatViewModel) {
    val state by link.state.collectAsStateWithLifecycle()
    val talk by chat.uiState.collectAsStateWithLifecycle()
    val canWalk = state.connected && !state.busy
    Column(
        modifier = Modifier
            .fillMaxSize()
            .statusBarsPadding()
            .imePadding()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 16.dp, vertical = 12.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        LinkHeader(state.robotName, state.connected, onSwitch = link::forgetRobot)
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(24.dp))
                .background(MaterialTheme.colorScheme.surfaceVariant)
                .padding(vertical = 14.dp, horizontal = 12.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            WalkKey("Forward", Icons.Default.KeyboardArrowUp, canWalk) { link.walk(FORWARD) }
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                WalkKey("Left", Icons.AutoMirrored.Filled.KeyboardArrowLeft, canWalk) { link.walk(LEFT) }
                WalkKey("Stop", Icons.Default.Stop, canWalk, isStop = true) { link.walk(STOP) }
                WalkKey("Right", Icons.AutoMirrored.Filled.KeyboardArrowRight, canWalk) { link.walk(RIGHT) }
            }
            WalkKey("Back", Icons.Default.KeyboardArrowDown, canWalk) { link.walk(BACK) }
            Text(
                text = state.note,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                textAlign = TextAlign.Center,
                maxLines = 2,
                modifier = Modifier.fillMaxWidth(),
            )
        }
        SpeakPanel(talk, chat)
    }
}

@Composable
private fun LinkHeader(robotName: String, connected: Boolean, onSwitch: () -> Unit) {
    Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.fillMaxWidth()) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(
                modifier = Modifier
                    .size(12.dp)
                    .clip(CircleShape)
                    .background(if (connected) Color(0xFF2E7D32) else Color(0xFFF9A825)),
            )
            Text(
                text = robotName.ifBlank { "Robot dog" },
                style = MaterialTheme.typography.headlineSmall,
                modifier = Modifier.padding(start = 10.dp),
            )
        }
        Text(
            text = if (connected) "Connected" else "Not connected",
            style = MaterialTheme.typography.labelLarge,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        if (!connected) {
            LinearProgressIndicator(modifier = Modifier.fillMaxWidth().padding(top = 8.dp))
        }
        TextButton(onClick = onSwitch) {
            Text("Switch robot (connect to the nearest)")
        }
    }
}

/** Hold Speak (SenseVoice) or type; Gemma answers as the dog and the phone says it. */
@Composable
private fun SpeakPanel(state: ChatUiState, chat: ChatViewModel) {
    var draft by remember { mutableStateOf("") }
    val canTalk = state.isReady && !state.isBusy
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(24.dp))
            .background(MaterialTheme.colorScheme.secondaryContainer)
            .padding(14.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        if (state.dogHeard.isNotBlank()) TalkLine("You", state.dogHeard)
        if (state.dogSaid.isNotBlank()) TalkLine("Dog", state.dogSaid)
        if (state.dogDid.isNotBlank()) TalkLine("Does", state.dogDid)
        Text(
            text = if (!state.isReady && state.dogHeard.isBlank()) state.status else state.dogTalkNote,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSecondaryContainer,
        )
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                HoldMicButton(
                    enabled = canTalk && state.voiceReady,
                    isRecording = state.isRecording,
                    diameter = 72.dp,
                    onPress = chat::onDogMicPress,
                    onRelease = chat::onDogMicRelease,
                )
                Text("Speak", style = MaterialTheme.typography.labelLarge)
            }
            OutlinedTextField(
                value = draft,
                onValueChange = { draft = it },
                modifier = Modifier.weight(1f),
                enabled = canTalk && !state.isRecording,
                placeholder = { Text("hello, can you sit?") },
                maxLines = 3,
            )
            IconButton(
                onClick = {
                    chat.askDog(draft)
                    draft = ""
                },
                enabled = canTalk && !state.isRecording && draft.isNotBlank(),
            ) {
                Icon(Icons.AutoMirrored.Filled.Send, contentDescription = "Ask the dog")
            }
        }
    }
}

@Composable
private fun TalkLine(who: String, text: String) {
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        Text(
            text = who,
            style = MaterialTheme.typography.labelLarge,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.onSecondaryContainer,
        )
        Text(
            text = text,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSecondaryContainer,
        )
    }
}

@Composable
private fun WalkKey(
    label: String,
    icon: ImageVector,
    enabled: Boolean,
    isStop: Boolean = false,
    onClick: () -> Unit,
) {
    Button(
        onClick = onClick,
        enabled = enabled,
        modifier = Modifier.size(width = 100.dp, height = 72.dp),
        shape = RoundedCornerShape(18.dp),
        contentPadding = PaddingValues(horizontal = 4.dp, vertical = 6.dp),
        colors = ButtonDefaults.buttonColors(
            containerColor = if (isStop) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.primary,
            contentColor = if (isStop) MaterialTheme.colorScheme.onError else MaterialTheme.colorScheme.onPrimary,
        ),
    ) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Icon(icon, contentDescription = label, modifier = Modifier.size(28.dp))
            Text(label, maxLines = 1, softWrap = false, style = MaterialTheme.typography.labelLarge)
        }
    }
}
