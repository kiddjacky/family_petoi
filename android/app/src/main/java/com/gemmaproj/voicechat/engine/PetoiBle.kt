package com.gemmaproj.voicechat.engine

import android.annotation.SuppressLint
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothGatt
import android.bluetooth.BluetoothGattCallback
import android.bluetooth.BluetoothGattCharacteristic
import android.bluetooth.BluetoothGattDescriptor
import android.bluetooth.BluetoothManager
import android.bluetooth.BluetoothProfile
import android.bluetooth.BluetoothStatusCodes
import android.bluetooth.le.ScanCallback
import android.bluetooth.le.ScanResult
import android.bluetooth.le.ScanSettings
import android.content.Context
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import android.util.Log
import java.util.UUID
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

/**
 * Petoi Bluetooth LE, the same Nordic UART link as OpenCat Gemma (`opencat_gemma/ble.py`).
 * Commands are `text\n` written with no response. The robot prints the command's first
 * letter on its own line when it is done. No pairing step.
 */
@SuppressLint("MissingPermission")
internal class PetoiBle(context: Context) {
    private val appContext = context.applicationContext
    private val callLock = ReentrantLock()
    private val bufferLock = Object()
    private val buffer = StringBuilder()
    private val gattThread = HandlerThread("petoi-ble").apply { start() }
    private val gattHandler = Handler(gattThread.looper)

    @Volatile private var gatt: BluetoothGatt? = null
    @Volatile private var lastHeardNs = 0L

    /** Milliseconds since the robot last sent anything. */
    val quietMs: Long get() = (System.nanoTime() - lastHeardNs) / 1_000_000
    private var rx: BluetoothGattCharacteristic? = null
    private var ready = false
    private var closed = false
    private var failMessage: String? = null
    private val sessionLock = Object()

    val isConnected: Boolean get() = ready && !closed

    var robotName: String = ""
        private set

    /** With [onlyName], other Petoi robots nearby are ignored even when their signal is stronger. */
    fun connect(onlyName: String? = null): String = callLock.withLock {
        closed = false
        closeGatt()
        val adapter = appContext.getSystemService(BluetoothManager::class.java)?.adapter
            ?: error("This phone has no Bluetooth")
        if (!adapter.isEnabled) error("Bluetooth is off")
        val (name, device) = scan(adapter, onlyName)
        robotName = name
        openGatt(device)
        handshake()
    }

    /**
     * Turns and other looping gaits ([TIMED_CODES]) never stop on their own, so they are held for
     * [turnHoldMs] and then balanced. Everything else waits for its echo up to [timeoutMs].
     */
    fun move(command: String, turnHoldMs: Long = TURN_HOLD_MS, timeoutMs: Long = WALK_MS): String = callLock.withLock {
        if (command.substringBefore(' ') in TIMED_CODES) {
            val turn = send(command, TURN_ECHO_MS)
            Thread.sleep(turnHoldMs.coerceIn(MIN_TURN_HOLD_MS, MAX_TURN_HOLD_MS))
            val stop = send("kbalance", BALANCE_MS)
            return "$turn | $stop"
        }
        send(command, timeoutMs)
    }

    /**
     * Sends `?` (prints the model and version; moves nothing) and reports whether any reply came.
     * Skipped, as alive, while a command holds the link: that command will notice a dead robot.
     */
    fun ping(timeoutMs: Long): Boolean {
        if (!callLock.tryLock()) return true
        try {
            if (!ready || closed) return false
            synchronized(bufferLock) { buffer.setLength(0) }
            writeBlocking("?\n")
            return waitForText(timeoutMs).isNotEmpty()
        } catch (error: IllegalStateException) {
            Log.w(TAG, "ping: ${error.message}")
            return false
        } finally {
            callLock.unlock()
        }
    }

    fun close() {
        closed = true
        synchronized(bufferLock) { bufferLock.notifyAll() }
        synchronized(sessionLock) { sessionLock.notifyAll() }
        callLock.withLock { closeGatt() }
    }

    private fun scan(adapter: android.bluetooth.BluetoothAdapter, onlyName: String?): Pair<String, BluetoothDevice> {
        val scanner = adapter.bluetoothLeScanner ?: error("Bluetooth LE is not available")
        val found = ArrayList<Triple<Int, String, BluetoothDevice>>()
        val nearby = linkedSetOf<String>()
        val callback = object : ScanCallback() {
            override fun onScanResult(callbackType: Int, result: ScanResult) {
                val record = result.scanRecord
                val name = record?.deviceName ?: result.device.name
                val hasNus = record?.serviceUuids.orEmpty().any { it.uuid == NUS_SERVICE }
                if (!name.isNullOrBlank()) synchronized(nearby) { nearby.add("$name ${result.rssi}") }
                val label = when {
                    !name.isNullOrBlank() && isPetoiName(name) -> name
                    hasNus -> name?.takeIf { it.isNotBlank() } ?: "Petoi"
                    else -> return
                }
                synchronized(found) {
                    found.removeAll { it.third.address == result.device.address }
                    found.add(Triple(result.rssi, label, result.device))
                }
            }

            override fun onScanFailed(errorCode: Int) {
                Log.e(TAG, "scan failed $errorCode")
            }
        }
        val settings = ScanSettings.Builder().setScanMode(ScanSettings.SCAN_MODE_LOW_LATENCY).build()
        try {
            scanner.startScan(null, settings, callback)
            Thread.sleep(SCAN_MS)
        } finally {
            try {
                scanner.stopScan(callback)
            } catch (error: Exception) {
                Log.w(TAG, "stopScan: ${error.message}")
            }
        }
        val best = synchronized(found) { found.filter { onlyName == null || it.second == onlyName }.maxByOrNull { it.first } }
            ?: if (onlyName != null) {
                error("$onlyName is not advertising. Turn it on and close the Petoi phone app.")
            } else {
                val others = synchronized(nearby) { nearby.joinToString(", ") }
                error(
                    if (others.isEmpty()) {
                        "No Petoi robot is advertising. Turn it on and close the Petoi phone app."
                    } else {
                        "No Petoi robot is advertising (nearby: $others). Turn it on and close the Petoi phone app."
                    },
                )
            }
        Log.i(TAG, "bluetooth ${best.second} rssi=${best.first}")
        return best.second to best.third
    }

    private fun openGatt(device: BluetoothDevice) {
        ready = false
        failMessage = null
        rx = null
        val latch = CountDownLatch(1)
        val callback = object : BluetoothGattCallback() {
            override fun onConnectionStateChange(gatt: BluetoothGatt, status: Int, newState: Int) {
                if (newState == BluetoothProfile.STATE_CONNECTED && status == BluetoothGatt.GATT_SUCCESS) {
                    gatt.discoverServices()
                    return
                }
                if (newState == BluetoothProfile.STATE_DISCONNECTED) {
                    ready = false
                    if (failMessage == null) failMessage = "Bluetooth disconnected ($status)"
                    // A dead GATT left open makes Android refuse the next connect (status 133).
                    gatt.close()
                    if (this@PetoiBle.gatt === gatt) this@PetoiBle.gatt = null
                    latch.countDown()
                    synchronized(sessionLock) { sessionLock.notifyAll() }
                    synchronized(bufferLock) { bufferLock.notifyAll() }
                }
            }

            override fun onServicesDiscovered(gatt: BluetoothGatt, status: Int) {
                if (status != BluetoothGatt.GATT_SUCCESS) {
                    failMessage = "Bluetooth services failed ($status)"
                    latch.countDown()
                    return
                }
                val service = gatt.getService(NUS_SERVICE)
                val transmit = service?.getCharacteristic(NUS_TX)
                val receive = service?.getCharacteristic(NUS_RX)
                if (transmit == null || receive == null) {
                    failMessage = "This device is not a Petoi Nordic UART"
                    latch.countDown()
                    return
                }
                rx = receive
                gatt.setCharacteristicNotification(transmit, true)
                val cccd = transmit.getDescriptor(CCCD)
                if (cccd == null) {
                    failMessage = "The robot did not offer notify"
                    latch.countDown()
                    return
                }
                val written = gatt.writeDescriptor(cccd, BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE)
                if (written != BluetoothStatusCodes.SUCCESS) {
                    failMessage = "Could not subscribe to the robot ($written)"
                    latch.countDown()
                }
            }

            override fun onDescriptorWrite(gatt: BluetoothGatt, descriptor: BluetoothGattDescriptor, status: Int) {
                if (status != BluetoothGatt.GATT_SUCCESS) {
                    failMessage = "Notify subscribe failed ($status)"
                } else {
                    ready = true
                }
                latch.countDown()
            }

            override fun onCharacteristicChanged(
                gatt: BluetoothGatt,
                characteristic: BluetoothGattCharacteristic,
                value: ByteArray,
            ) {
                val text = value.toString(Charsets.ISO_8859_1)
                lastHeardNs = System.nanoTime()
                synchronized(bufferLock) {
                    buffer.append(text)
                    bufferLock.notifyAll()
                }
            }
        }
        val opened = device.connectGatt(appContext, false, callback, BluetoothDevice.TRANSPORT_LE, BluetoothDevice.PHY_LE_1M, gattHandler)
        gatt = opened
        if (!latch.await(CONNECT_MS, TimeUnit.MILLISECONDS) || !ready) {
            val why = failMessage ?: "The robot did not finish connecting"
            closeGatt()
            error(why)
        }
    }

    private fun handshake(): String {
        val pushed = waitForText(PUSH_MS)
        if (isPetoiBanner(pushed)) {
            synchronized(bufferLock) { buffer.setLength(0) }
            return pushed.trim()
        }
        if (pushed.isNotEmpty()) {
            try {
                waitForLine("d", GREETING_MS, "the connect greeting")
            } catch (error: IllegalStateException) {
                Log.w(TAG, "no forced rest: ${error.message}")
            }
        }
        synchronized(bufferLock) { buffer.setLength(0) }
        writeBlocking("?\n")
        val reply = waitForQuery(QUERY_MS)
        synchronized(bufferLock) { buffer.setLength(0) }
        Log.i(TAG, "robot ? $reply")
        return reply
    }

    private fun isPetoiBanner(text: String): Boolean {
        return text.contains("Petoi") && text.trimEnd().endsWith("?")
    }

    private fun send(command: String, timeoutMs: Long): String {
        if (!ready || closed) error("Bluetooth robot is not connected")
        synchronized(bufferLock) { buffer.setLength(0) }
        writeBlocking(command + "\n")
        val lines = waitForLine(command.first().toString(), timeoutMs, command)
        return lines.joinToString(" | ")
    }

    private fun writeBlocking(payload: String) {
        val done = CountDownLatch(1)
        var error: Exception? = null
        val task = Runnable {
            try {
                val receive = rx ?: error("Bluetooth is not connected")
                val link = gatt ?: error("Bluetooth is not connected")
                val status = link.writeCharacteristic(
                    receive,
                    payload.toByteArray(Charsets.ISO_8859_1),
                    BluetoothGattCharacteristic.WRITE_TYPE_NO_RESPONSE,
                )
                if (status != BluetoothStatusCodes.SUCCESS) error("Bluetooth write failed ($status)")
            } catch (thrown: Exception) {
                error = thrown
            } finally {
                done.countDown()
            }
        }
        if (Looper.myLooper() == gattHandler.looper) task.run() else gattHandler.post(task)
        if (!done.await(WRITE_MS, TimeUnit.MILLISECONDS)) error("Bluetooth write timed out")
        error?.let { throw it }
    }

    private fun waitForText(timeoutMs: Long): String {
        val end = System.nanoTime() + timeoutMs * 1_000_000
        synchronized(bufferLock) {
            while (buffer.isEmpty() && !closed && System.nanoTime() < end) {
                val left = end - System.nanoTime()
                if (left <= 0) break
                bufferLock.wait(waitMillis(left))
            }
            return buffer.toString()
        }
    }

    private fun waitForLine(token: String, timeoutMs: Long, what: String): List<String> {
        val end = System.nanoTime() + timeoutMs * 1_000_000
        synchronized(bufferLock) {
            while (!closed && System.nanoTime() < end) {
                val lines = completeLines()
                if (lines.any { it.equals(token, ignoreCase = true) }) return lines
                if (!ready) error("Bluetooth disconnected during $what")
                val left = end - System.nanoTime()
                if (left <= 0) break
                bufferLock.wait(waitMillis(left))
            }
        }
        if (closed) error("Bluetooth closed")
        error("No '$token' from the robot for $what")
    }

    private fun waitForQuery(timeoutMs: Long): String {
        val end = System.nanoTime() + timeoutMs * 1_000_000
        synchronized(bufferLock) {
            while (!closed && System.nanoTime() < end) {
                val text = buffer.toString()
                val lines = text.split('\n').map { it.trim() }
                val petoiHandshake = text.contains("Petoi") && text.trimEnd().endsWith("?")
                if ("?" in lines.dropLast(1) || petoiHandshake) {
                    return lines.filter { it.isNotEmpty() }.joinToString(" | ")
                }
                val left = end - System.nanoTime()
                if (left <= 0) break
                bufferLock.wait(waitMillis(left))
            }
        }
        error("The robot did not answer '?'")
    }

    /** Object.wait(0) waits forever. A sub-millisecond remainder must not become that. */
    private fun waitMillis(remainingNs: Long): Long {
        val millis = remainingNs / 1_000_000L
        return if (millis < 1L) 1L else millis
    }

    private fun completeLines(): List<String> {
        return buffer.toString().split('\n').dropLast(1).map { it.trim() }.filter { it.isNotEmpty() }
    }

    private fun closeGatt() {
        ready = false
        rx = null
        try {
            gatt?.disconnect()
            gatt?.close()
        } catch (error: Exception) {
            Log.w(TAG, "close: ${error.message}")
        }
        gatt = null
        synchronized(bufferLock) { buffer.setLength(0) }
    }

    companion object {
        private const val TAG = "PetoiBle"
        private const val SCAN_MS = 5_000L
        private const val CONNECT_MS = 15_000L
        private const val PUSH_MS = 2_000L
        private const val GREETING_MS = 8_000L
        private const val QUERY_MS = 5_000L
        private const val WRITE_MS = 3_000L
        private const val TURN_ECHO_MS = 3_000L
        const val TURN_HOLD_MS = 700L
        private const val MIN_TURN_HOLD_MS = 300L
        private const val MAX_TURN_HOLD_MS = 3_000L
        private const val BALANCE_MS = 4_000L
        private const val WALK_MS = 8_000L

        /** Quaddle gaits that loop until the next command (opencat-gemma contract kind "timed"). */
        val TIMED_CODES = setOf("kwkL", "kwkR", "kvtL", "kvtR")

        private val NUS_SERVICE: UUID = UUID.fromString("6e400001-b5a3-f393-e0a9-e50e24dcca9e")
        private val NUS_RX: UUID = UUID.fromString("6e400002-b5a3-f393-e0a9-e50e24dcca9e")
        private val NUS_TX: UUID = UUID.fromString("6e400003-b5a3-f393-e0a9-e50e24dcca9e")
        private val CCCD: UUID = UUID.fromString("00002902-0000-1000-8000-00805f9b34fb")
        private val NAME_PREFIXES = listOf("bittle", "nybble", "quaddle", "petoi")

        internal fun isPetoiName(name: String): Boolean {
            val lower = name.lowercase()
            return NAME_PREFIXES.any { lower.startsWith(it) }
        }
    }
}
