package com.gemmaproj.voicechat.engine

import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener

/**
 * One short Petoi skill over the BiBoard WebSocket on port 81.
 * A walk gait is followed by balance so the robot does not keep going.
 */
internal class PetoiMover {
    private val client = OkHttpClient.Builder()
        .connectTimeout(3, TimeUnit.SECONDS)
        .readTimeout(4, TimeUnit.SECONDS)
        .build()

    fun move(hostPort: String, skill: String): String {
        val endpoint = endpoint(hostPort)
        val skills = if (skill.startsWith("kvt")) listOf(skill) else listOf(skill, "kbalance")
        return send(endpoint, skills)
    }

    private fun send(endpoint: String, skills: List<String>): String {
        val payload = buildString {
            append("{\"type\":\"command\",\"taskId\":\"")
            append(System.currentTimeMillis())
            append("\",\"commands\":[")
            skills.forEachIndexed { index, skill ->
                if (index > 0) append(',')
                append('"')
                append(skill)
                append('"')
            }
            append("],\"timestamp\":")
            append(System.currentTimeMillis())
            append('}')
        }
        val latch = CountDownLatch(1)
        var reply = "no reply"
        val socket = client.newWebSocket(
            Request.Builder().url(endpoint).build(),
            object : WebSocketListener() {
                override fun onOpen(webSocket: WebSocket, response: Response) {
                    webSocket.send(payload)
                }

                override fun onMessage(webSocket: WebSocket, text: String) {
                    reply = text.take(180)
                    latch.countDown()
                    webSocket.close(1000, null)
                }

                override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                    reply = t.message ?: t.javaClass.simpleName
                    latch.countDown()
                }
            },
        )
        val heard = latch.await(4, TimeUnit.SECONDS)
        if (!heard) {
            socket.cancel()
            reply = "timed out"
        }
        return reply
    }

    private fun endpoint(hostPort: String): String {
        val trimmed = hostPort.trim()
        val withPort = if (trimmed.contains(":")) trimmed else "$trimmed:81"
        return "ws://$withPort"
    }
}

internal fun petoiSkill(dx: Float, dy: Float): String? {
    val distance = kotlin.math.hypot(dx.toDouble(), dy.toDouble()).toFloat()
    if (distance < 0.12f) return null
    return when {
        kotlin.math.abs(dx) >= kotlin.math.abs(dy) && dx > 0f -> "kwkR"
        kotlin.math.abs(dx) >= kotlin.math.abs(dy) -> "kwkL"
        dy < 0f -> "kvtF"
        else -> "kbkF"
    }
}
