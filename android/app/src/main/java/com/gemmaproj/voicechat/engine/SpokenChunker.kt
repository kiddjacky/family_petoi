package com.gemmaproj.voicechat.engine

/**
 * Splits a growing assistant reply into pieces Kokoro can speak.
 * A piece is one clause: it ends at a comma, semicolon, or sentence stop.
 * Text with no punctuation yet stays buffered so speech is a continuous
 * phrase, not a word at a time. The tail after generation finishes is spoken
 * even when it has no closing mark. Decimal points such as "3.14" are not cuts.
 */
class SpokenChunker {
    private var committed = 0

    fun takeReady(visible: String, finished: Boolean): List<String> {
        if (visible.length < committed) committed = 0
        val pieces = ArrayList<String>()
        while (committed < visible.length) {
            val rest = visible.substring(committed)
            val cut = findPunctuation(rest)
            if (cut <= 0) break
            val piece = rest.substring(0, cut).trim()
            committed += cut
            if (piece.isNotEmpty()) pieces += piece
        }
        if (finished && committed < visible.length) {
            val piece = visible.substring(committed).trim()
            committed = visible.length
            if (piece.isNotEmpty()) pieces += piece
        }
        return pieces
    }

    private fun findPunctuation(rest: String): Int {
        for (i in rest.indices) {
            val c = rest[i]
            if (isSentenceEnd(c) && !isDecimalDot(rest, i)) return i + 1
            if (isClauseBreak(c) && i >= CLAUSE_MIN) return i + 1
        }
        return -1
    }

    private fun isSentenceEnd(c: Char): Boolean =
        c == '.' || c == '!' || c == '?' || c == '。' || c == '！' || c == '？' || c == '\n' || c == '…'

    private fun isClauseBreak(c: Char): Boolean =
        c == ',' || c == '，' || c == '、' || c == ';' || c == '；' || c == ':' || c == '：'

    private fun isDecimalDot(text: String, index: Int): Boolean {
        if (text[index] != '.') return false
        val prevDigit = index > 0 && text[index - 1].isDigit()
        val nextDigit = index + 1 < text.length && text[index + 1].isDigit()
        return prevDigit && nextDigit
    }

    companion object {
        private const val CLAUSE_MIN = 1
    }
}
