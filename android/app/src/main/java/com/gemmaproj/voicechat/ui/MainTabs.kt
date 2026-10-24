package com.gemmaproj.voicechat.ui

import android.content.Context
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.consumeWindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.ime
import androidx.compose.foundation.layout.navigationBars
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Chat
import androidx.compose.material.icons.filled.Pets
import androidx.compose.material3.Icon
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.gemmaproj.voicechat.engine.DogLink

enum class MainTab { CHAT, DOG }

private const val UI_PREFS = "ui"
private const val TAB_KEY = "tab"

/** Remembers the last tab so reopening the app lands where it was left. */
object TabMemory {
    fun load(context: Context): MainTab {
        val saved = context.getSharedPreferences(UI_PREFS, Context.MODE_PRIVATE).getString(TAB_KEY, null)
        return MainTab.entries.firstOrNull { it.name == saved } ?: MainTab.CHAT
    }

    fun save(context: Context, tab: MainTab) {
        context.getSharedPreferences(UI_PREFS, Context.MODE_PRIVATE).edit().putString(TAB_KEY, tab.name).apply()
    }
}

@Composable
fun MainTabs(chat: ChatViewModel, dog: DogLink) {
    val context = LocalContext.current
    var tab by remember { mutableStateOf(TabMemory.load(context)) }
    val chatState by chat.uiState.collectAsStateWithLifecycle()
    val dogState by dog.state.collectAsStateWithLifecycle()
    val keyboardUp = WindowInsets.ime.getBottom(LocalDensity.current) > 0
    // The camera is a full-screen overlay and the keyboard sits where the bar would be.
    val showBar = !keyboardUp && !(tab == MainTab.CHAT && chatState.cameraOpen)

    Column(modifier = Modifier.fillMaxSize()) {
        Box(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
                .let { if (showBar) it.consumeWindowInsets(WindowInsets.navigationBars) else it },
        ) {
            when (tab) {
                MainTab.CHAT -> ChatScreen(chat)
                MainTab.DOG -> DogScreen(dog, chat)
            }
        }
        if (showBar) {
            NavigationBar {
                NavigationBarItem(
                    selected = tab == MainTab.CHAT,
                    onClick = {
                        tab = MainTab.CHAT
                        TabMemory.save(context, tab)
                    },
                    icon = { Icon(Icons.AutoMirrored.Filled.Chat, contentDescription = null) },
                    label = { Text("Chat") },
                )
                NavigationBarItem(
                    selected = tab == MainTab.DOG,
                    onClick = {
                        tab = MainTab.DOG
                        TabMemory.save(context, tab)
                    },
                    icon = { Icon(Icons.Default.Pets, contentDescription = null) },
                    label = { Text(if (dogState.connected) "Dog ●" else "Dog") },
                )
            }
        }
    }
}
