package com.gemmaproj.voicechat.engine

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.util.Log
import androidx.core.content.ContextCompat
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

data class DogLinkState(
    val robotName: String = "",
    val connected: Boolean = false,
    val busy: Boolean = false,
    val note: String = "",
)

/**
 * The one Bluetooth link to the Petoi robot for the whole process. It outlives screens and
 * view models, so leaving the app does not drop the robot. [keepConnected] reconnects on its
 * own whenever the link falls. Connecting only sends `?`; no gait is sent unless [walk] or
 * [move] is called.
 */
class DogLink(context: Context) {
    private val appContext = context.applicationContext
    private val ble = PetoiBle(appContext)
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val moving = AtomicBoolean(false)
    @Volatile private var silentMoves = 0
    private var keeper: Job? = null

    private val _state = MutableStateFlow(DogLinkState(note = "Looking for the robot…"))
    val state: StateFlow<DogLinkState> = _state.asStateFlow()

    val isConnected: Boolean get() = ble.isConnected

    @Synchronized
    fun keepConnected() {
        if (keeper?.isActive == true) return
        keeper = scope.launch {
            var lastPingNs = 0L
            var missedPings = 0
            while (isActive) {
                if (ble.isConnected) {
                    delay(WATCH_MS)
                    // A robot that browned out can stay "connected" on the phone side forever.
                    val idle = !moving.get() && ble.quietMs > PING_QUIET_MS
                    if (idle && (System.nanoTime() - lastPingNs) / 1_000_000 > PING_QUIET_MS) {
                        lastPingNs = System.nanoTime()
                        missedPings = if (ble.ping(PING_REPLY_MS)) 0 else missedPings + 1
                        if (missedPings >= PINGS_BEFORE_DROP) {
                            missedPings = 0
                            log("silent for ${ble.quietMs / 1000}s, reconnecting")
                            ble.close()
                        }
                    }
                    continue
                }
                missedPings = 0
                if (_state.value.connected) {
                    log("dropped ${_state.value.robotName}")
                    _state.update { it.copy(connected = false, busy = false, note = "Lost the robot. Reconnecting…") }
                }
                if (!hasBluetoothPermission()) {
                    _state.update { it.copy(note = "Allow Nearby devices so the app can find the robot.") }
                    delay(RETRY_MS)
                    continue
                }
                try {
                    if (!_state.value.note.startsWith("Lost")) {
                        _state.update { it.copy(note = "Looking for the robot…") }
                    }
                    val identity = ble.connect(onlyName = rememberedRobot())
                    if (rememberedRobot() == null) rememberRobot(ble.robotName)
                    log("connected ${ble.robotName} $identity")
                    _state.update {
                        DogLinkState(robotName = ble.robotName, connected = true, note = "Ready. One tap is one short step.")
                    }
                } catch (cancelled: CancellationException) {
                    throw cancelled
                } catch (error: Exception) {
                    val message = error.message ?: "No robot"
                    log("failed $message")
                    _state.update { it.copy(connected = false, note = "$message Retrying…") }
                    delay(RETRY_MS)
                }
            }
        }
    }

    /**
     * The first robot connected becomes this app's robot; later scans ignore every other Petoi
     * robot nearby. Forgetting it lets the next scan take the strongest one and remember that.
     */
    fun forgetRobot() {
        appContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().remove(ROBOT_KEY).apply()
        log("forgot ${ble.robotName}")
        _state.update { DogLinkState(note = "Looking for the nearest robot…") }
        scope.launch { ble.close() }
    }

    private fun rememberedRobot(): String? =
        appContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getString(ROBOT_KEY, null)

    private fun rememberRobot(name: String) {
        if (name.isBlank()) return
        appContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().putString(ROBOT_KEY, name).apply()
    }

    /** Tap from the walk pad. Ignored while another step is still running. */
    fun walk(command: String) {
        if (!ble.isConnected || moving.get()) return
        scope.launch {
            try {
                move(command)
            } catch (error: Exception) {
                Log.w(TAG, "walk $command failed: ${error.message}")
            }
        }
    }

    /** Blocks until the robot echoes the command. Throws if another step is still running. */
    fun move(command: String, turnHoldMs: Long = PetoiBle.TURN_HOLD_MS, timeoutMs: Long = MOVE_TIMEOUT_MS): String {
        if (!moving.compareAndSet(false, true)) error("The robot is still moving")
        _state.update { it.copy(busy = true, note = stepWords(command)) }
        try {
            val reply = ble.move(command, turnHoldMs, timeoutMs)
            silentMoves = 0
            _state.update { it.copy(busy = false, note = reply.take(NOTE_CHARS)) }
            return reply
        } catch (error: Exception) {
            _state.update { it.copy(busy = false, note = error.message ?: "Move failed") }
            silentMoves += 1
            if (silentMoves >= SILENT_MOVES_BEFORE_DROP && ble.isConnected) {
                // The keeper loop sees the closed link and connects again.
                silentMoves = 0
                log("$SILENT_MOVES_BEFORE_DROP moves unanswered, reconnecting")
                ble.close()
            }
            throw error
        } finally {
            moving.set(false)
        }
    }

    /**
     * Runs [actions] in order, blocking. A failed move stops the sequence and stands the robot
     * up; a missing echo on an optional eye cue is only logged. Returns one line per action.
     */
    internal fun perform(actions: List<DogAction>): List<String> {
        val lines = mutableListOf<String>()
        for (action in actions) {
            try {
                val reply = move(action.command, action.turnHoldMs.takeIf { it > 0 } ?: PetoiBle.TURN_HOLD_MS, action.timeoutMs)
                lines += "${action.command} → $reply"
            } catch (error: IllegalStateException) {
                lines += "${action.command} failed: ${error.message}"
                if (action.optional) continue
                log("perform stopped at ${action.command}: ${error.message}")
                try {
                    move(STOP, timeoutMs = MOVE_TIMEOUT_MS)
                } catch (stopError: IllegalStateException) {
                    lines += "stop failed: ${stopError.message}"
                }
                break
            }
        }
        return lines
    }

    private fun hasBluetoothPermission(): Boolean {
        return listOf(Manifest.permission.BLUETOOTH_SCAN, Manifest.permission.BLUETOOTH_CONNECT).all {
            ContextCompat.checkSelfPermission(appContext, it) == PackageManager.PERMISSION_GRANTED
        }
    }

    private fun log(line: String) {
        Log.i(TAG, line)
        try {
            val file = File(appContext.filesDir, "dog-log.txt")
            if (file.length() > LOG_MAX_BYTES) file.writeText("")
            file.appendText("${System.currentTimeMillis()} $line\n")
        } catch (error: Exception) {
            Log.w(TAG, "dog log: ${error.message}")
        }
    }

    companion object {
        private const val TAG = "DogLink"
        private const val WATCH_MS = 1_000L
        private const val RETRY_MS = 3_000L
        private const val NOTE_CHARS = 140
        private const val LOG_MAX_BYTES = 64 * 1024L
        private const val MOVE_TIMEOUT_MS = 8_000L
        private const val STOP = "kbalance"
        private const val PREFS = "dog"
        private const val ROBOT_KEY = "robot"
        private const val PING_QUIET_MS = 20_000L
        private const val PING_REPLY_MS = 3_000L
        private const val PINGS_BEFORE_DROP = 2
        private const val SILENT_MOVES_BEFORE_DROP = 2

        fun stepWords(command: String): String =
            if (command == "kbalance") "Stop." else PlayWatcher.walkWords(command)
    }
}
