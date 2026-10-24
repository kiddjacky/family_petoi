package com.gemmaproj.voicechat

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.PowerManager
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.core.content.ContextCompat
import androidx.lifecycle.viewmodel.compose.viewModel
import com.gemmaproj.voicechat.ui.ChatViewModel
import com.gemmaproj.voicechat.ui.MainTab
import com.gemmaproj.voicechat.ui.MainTabs
import com.gemmaproj.voicechat.ui.TabMemory

class MainActivity : ComponentActivity() {

    private var wantsDogLink = false

    private val startupPermissions = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions(),
    ) {
        if (wantsDogLink) DogLinkService.start(this)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        val bleProbe = intent.getBooleanExtra("bleProbe", false)
        val profiling = intent.getBooleanExtra("profileTrack", false) || bleProbe
        val benching = profiling || intent.getBooleanExtra("bench", false) || intent.getBooleanExtra("litert", false)
        // Benchmarks time the GPU and CPU; keep the robot link out of them unless it is the subject.
        wantsDogLink = !benching || bleProbe
        if (benching) {
            val app = application as GemmaVoiceApp
            app.profileTrack = profiling
            app.profilePhoto = intent.getStringExtra("profilePhoto")
            app.profileGpu = intent.getBooleanExtra("profileGpu", false)
            app.bleProbe = bleProbe
            app.benchVariant = intent.getStringExtra("gemma") ?: "E2B"
            app.litertBench = intent.getBooleanExtra("litert", false)
            app.voiceProbeWav = intent.getStringExtra("voiceProbe")
            window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            getSystemService(PowerManager::class.java)
                .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "gemma:bench")
                .acquire(8 * 60 * 1000L)
        } else {
            val app = application as GemmaVoiceApp
            app.voiceProbeWav = intent.getStringExtra("voiceProbe")
            app.autoAsk = intent.getStringExtra("ask")
            app.pointPhoto = intent.getStringExtra("pointPhoto")
            app.gemmaAudio = intent.getBooleanExtra("gemmaAudio", false)
            app.dogAsk = intent.getStringExtra("dogAsk")
            val dogPad = intent.getBooleanExtra("dogPad", false) || app.dogAsk != null
            if (dogPad) TabMemory.save(this, MainTab.DOG)
            if (app.autoAsk != null || app.voiceProbeWav != null || app.gemmaAudio || app.pointPhoto != null || dogPad) {
                setShowWhenLocked(true)
                setTurnScreenOn(true)
                window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            }
        }
        val needed = listOf(
            Manifest.permission.RECORD_AUDIO,
            Manifest.permission.CAMERA,
            Manifest.permission.BLUETOOTH_SCAN,
            Manifest.permission.BLUETOOTH_CONNECT,
            Manifest.permission.POST_NOTIFICATIONS,
        ).filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }
        if (needed.isNotEmpty()) startupPermissions.launch(needed.toTypedArray())
        if (wantsDogLink) DogLinkService.start(this)
        val dogLink = (application as GemmaVoiceApp).dogLink
        setContent {
            MaterialTheme {
                Surface {
                    val vm: ChatViewModel = viewModel()
                    MainTabs(vm, dogLink)
                }
            }
        }
    }
}
