package com.example.plantcapture

import android.util.Base64
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.File
import java.net.HttpURLConnection
import java.net.URI
import java.net.URL

internal object UploadClient {
    data class Result(val code: Int, val body: String)

    fun normalizeBaseUrl(raw: String): String {
        val value = raw.trim().trimEnd('/')
        require(value.isNotBlank()) { "Server URL is empty" }
        val uri = URI(value)
        require(uri.scheme == "http" || uri.scheme == "https") { "URL must start with http:// or https://" }
        require(!uri.host.isNullOrBlank()) { "URL must contain a host or IP address" }
        return value
    }

    fun testConnection(baseUrl: String): Result {
        val connection = (URL("${normalizeBaseUrl(baseUrl)}/health").openConnection() as HttpURLConnection)
        return connection.useConnection {
            requestMethod = "GET"
            connectTimeout = 5_000
            readTimeout = 5_000
            useCaches = false
            setRequestProperty("Accept", "application/json")
            setRequestProperty("Connection", "close")
            connect()
            Result(responseCode, readResponseBody())
        }
    }

    fun upload(
        baseUrl: String,
        token: String,
        file: File,
        metadataJson: String,
        onProgress: (Int) -> Unit,
    ): Result {
        require(file.isFile && file.length() > 0L) { "Video file is missing or empty" }
        require(token.length >= 10) { "Token must contain at least 10 characters" }

        // Validate the shared token before streaming a potentially large file.
        // This also makes a 401/404/limit error visible instead of leaving the
        // server with an unread MP4 request body.
        val ready = checkReady(baseUrl, token)
        require(ready.code in 200..299) {
            "Receiver preflight failed: HTTP ${ready.code}: ${ready.body}"
        }

        val safeName = file.name.replace(Regex("[^A-Za-z0-9._-]"), "_")
        val connection = (
            URL("${normalizeBaseUrl(baseUrl)}/upload/$safeName").openConnection() as HttpURLConnection
            )
        return connection.useConnection {
            requestMethod = "PUT"
            doInput = true
            doOutput = true
            useCaches = false
            instanceFollowRedirects = false
            connectTimeout = 15_000
            readTimeout = 120_000
            setFixedLengthStreamingMode(file.length())
            setRequestProperty("Authorization", "Bearer $token")
            setRequestProperty("Content-Type", "video/mp4")
            setRequestProperty("Accept", "application/json")
            setRequestProperty("Connection", "close")
            setRequestProperty("X-Plant-Capture-Protocol", "2")
            setRequestProperty(
                "X-Capture-Metadata-B64",
                Base64.encodeToString(metadataJson.toByteArray(Charsets.UTF_8), Base64.NO_WRAP),
            )

            BufferedInputStream(file.inputStream(), 1024 * 1024).use { input ->
                BufferedOutputStream(outputStream, 1024 * 1024).use { output ->
                    val buffer = ByteArray(1024 * 1024)
                    var sent = 0L
                    var lastProgress = -1
                    while (true) {
                        val count = input.read(buffer)
                        if (count < 0) break
                        output.write(buffer, 0, count)
                        sent += count
                        val progress = ((sent * 100L) / file.length()).toInt().coerceIn(0, 100)
                        if (progress != lastProgress) {
                            lastProgress = progress
                            onProgress(progress)
                        }
                    }
                    output.flush()
                }
            }
            Result(responseCode, readResponseBody())
        }
    }


    private fun checkReady(baseUrl: String, token: String): Result {
        val connection = (
            URL("${normalizeBaseUrl(baseUrl)}/ready").openConnection() as HttpURLConnection
            )
        return connection.useConnection {
            requestMethod = "GET"
            useCaches = false
            connectTimeout = 5_000
            readTimeout = 5_000
            setRequestProperty("Authorization", "Bearer $token")
            setRequestProperty("Accept", "application/json")
            setRequestProperty("Connection", "close")
            connect()
            Result(responseCode, readResponseBody())
        }
    }

    private inline fun <T> HttpURLConnection.useConnection(block: HttpURLConnection.() -> T): T {
        return try {
            block()
        } finally {
            disconnect()
        }
    }

    private fun HttpURLConnection.readResponseBody(): String {
        val stream = if (responseCode in 200..299) inputStream else errorStream
        return stream?.bufferedReader(Charsets.UTF_8)?.use { it.readText() }.orEmpty()
    }
}
