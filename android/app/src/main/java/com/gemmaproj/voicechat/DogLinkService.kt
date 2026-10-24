package com.gemmaproj.voicechat

import android.Manifest
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.os.IBinder
import android.util.Log
import androidx.core.content.ContextCompat
import com.gemmaproj.voicechat.engine.DogLinkState
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.launch

/**
 * Keeps the process, and with it [GemmaVoiceApp.dogLink], alive after the app is left, so the
 * robot stays connected. Android only allows a connected-device foreground service once
 * Bluetooth connect permission is granted, so [start] checks that first.
 */
class DogLinkService : Service() {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main)
    private var watching = false

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val link = (application as GemmaVoiceApp).dogLink
        try {
            startForeground(
                NOTIFICATION_ID,
                buildNotification(link.state.value),
                ServiceInfo.FOREGROUND_SERVICE_TYPE_CONNECTED_DEVICE,
            )
        } catch (error: Exception) {
            Log.w(TAG, "foreground refused: ${error.message}")
            stopSelf()
            return START_NOT_STICKY
        }
        link.keepConnected()
        if (!watching) {
            watching = true
            val manager = getSystemService(NotificationManager::class.java)
            scope.launch {
                link.state
                    .map { it.copy(note = "") }
                    .distinctUntilChanged()
                    .collect { manager.notify(NOTIFICATION_ID, buildNotification(it)) }
            }
        }
        return START_STICKY
    }

    override fun onDestroy() {
        scope.cancel()
        super.onDestroy()
    }

    private fun buildNotification(state: DogLinkState): Notification {
        val open = PendingIntent.getActivity(
            this,
            0,
            Intent(this, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )
        val text = when {
            state.connected -> "Connected to ${state.robotName}"
            else -> "Looking for the robot…"
        }
        return Notification.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.stat_sys_data_bluetooth)
            .setContentTitle("Robot dog")
            .setContentText(text)
            .setContentIntent(open)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .build()
    }

    companion object {
        private const val TAG = "DogLinkService"
        private const val CHANNEL_ID = "dog-link"
        private const val NOTIFICATION_ID = 41

        /** Starts the service if Bluetooth permission is granted. Safe to call repeatedly. */
        fun start(context: Context) {
            val granted = listOf(Manifest.permission.BLUETOOTH_SCAN, Manifest.permission.BLUETOOTH_CONNECT).all {
                ContextCompat.checkSelfPermission(context, it) == PackageManager.PERMISSION_GRANTED
            }
            if (!granted) return
            val manager = context.getSystemService(NotificationManager::class.java)
            manager.createNotificationChannel(
                NotificationChannel(CHANNEL_ID, "Robot dog link", NotificationManager.IMPORTANCE_LOW),
            )
            try {
                ContextCompat.startForegroundService(context, Intent(context, DogLinkService::class.java))
            } catch (error: Exception) {
                Log.w(TAG, "could not start: ${error.message}")
            }
        }
    }
}
