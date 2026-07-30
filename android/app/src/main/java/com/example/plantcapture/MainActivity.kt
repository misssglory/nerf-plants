package com.example.plantcapture

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.view.MotionEvent
import android.widget.Button
import android.widget.EditText
import android.widget.ProgressBar
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.io.File
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import java.util.concurrent.Executors
import kotlin.math.roundToLong

class MainActivity : AppCompatActivity() {
    private lateinit var previewView: PreviewView
    private lateinit var serverUrl: EditText
    private lateinit var token: EditText
    private lateinit var testConnection: Button
    private lateinit var lockButton: Button
    private lateinit var recordButton: Button
    private lateinit var stopButton: Button
    private lateinit var uploadProgress: ProgressBar
    private lateinit var statusText: TextView
    private lateinit var cameraInfo: TextView

    private val worker = Executors.newSingleThreadExecutor()
    private val uiExecutor by lazy { ContextCompat.getMainExecutor(this) }
    private lateinit var cameraController: CameraCaptureController
    private var currentFile: File? = null

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions(),
    ) { grants ->
        if (grants.values.all { it }) {
            startCamera()
        } else {
            setStatus("Camera/local-network permission was denied")
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        serverUrl = findViewById(R.id.serverUrl)
        token = findViewById(R.id.token)
        testConnection = findViewById(R.id.testConnection)
        lockButton = findViewById(R.id.lockButton)
        recordButton = findViewById(R.id.recordButton)
        stopButton = findViewById(R.id.stopButton)
        uploadProgress = findViewById(R.id.uploadProgress)
        statusText = findViewById(R.id.statusText)
        cameraInfo = findViewById(R.id.cameraInfo)

        val preferences = getSharedPreferences("plant_capture", MODE_PRIVATE)
        serverUrl.setText(preferences.getString("server_url", "http://192.168.1.100:8765"))
        token.setText(preferences.getString("token", "change-this-token"))

        cameraController = CameraCaptureController(this, uiExecutor)
        setupUi()
        requestRequiredPermissions()
    }

    private fun setupUi() {
        testConnection.setOnClickListener {
            saveConnectionSettings()
            testPcConnection()
        }

        previewView.setOnTouchListener { _, event ->
            if (event.action == MotionEvent.ACTION_UP) {
                setStatus("Focusing and metering at tapped point…")
                cameraController.focusAt(previewView, event.x, event.y, lockAfterFocus = false) { focused ->
                    setStatus(if (focused) "Focus set. Lock before recording." else "Metering applied; autofocus did not confirm.")
                }
            }
            true
        }

        lockButton.setOnClickListener {
            if (cameraController.state.aeAwbLocked) {
                cameraController.unlock { success ->
                    updateCameraState()
                    setStatus(if (success) "Exposure and white balance unlocked" else "Could not unlock camera controls")
                }
            } else {
                setStatus("Focusing center; locking exposure and white balance…")
                cameraController.focusCenterAndLock(previewView) { focused ->
                    updateCameraState()
                    val focusMessage = if (focused) "focus confirmed" else "focus not confirmed"
                    setStatus("Camera locked; $focusMessage. Keep lighting unchanged.")
                }
            }
        }

        recordButton.setOnClickListener {
            saveConnectionSettings()
            startRecording()
        }

        stopButton.setOnClickListener {
            stopButton.isEnabled = false
            setStatus("Finalizing MP4…")
            cameraController.stopRecording()
        }
    }

    private fun requestRequiredPermissions() {
        val required = mutableListOf(Manifest.permission.CAMERA)
        if (Build.VERSION.SDK_INT >= 37) {
            required += Manifest.permission.ACCESS_LOCAL_NETWORK
        }
        val missing = required.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }
        if (missing.isEmpty()) {
            startCamera()
        } else {
            permissionLauncher.launch(missing.toTypedArray())
        }
    }

    private fun startCamera() {
        cameraController.startCamera(
            owner = this,
            previewView = previewView,
            onReady = {
                updateCameraState()
                setStatus("Camera ready. Tap the plant, then lock exposure/color.")
            },
            onError = { setStatus("Camera error: ${it.message}") },
        )
    }

    private fun updateCameraState() {
        val state = cameraController.state
        cameraInfo.text = buildString {
            append("Quality: ${state.qualityPreference}; ${state.targetFpsRange}; ")
            append(if (state.aeAwbLocked) "AE/AWB locked" else "AE/AWB automatic")
            append("; stabilization requested off: ${state.stabilizationRequestedOff}")
        }
        lockButton.text = if (state.aeAwbLocked) {
            "Unlock exposure/color"
        } else {
            "Focus center + lock exposure/color"
        }
    }

    private fun startRecording() {
        if (cameraController.isRecording) return
        val capturesDir = File(filesDir, "captures")
        val timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"))
        val file = File(capturesDir, "plant_$timestamp.mp4")
        currentFile = file
        uploadProgress.progress = 0
        recordButton.isEnabled = false
        stopButton.isEnabled = true

        if (!cameraController.state.aeAwbLocked) {
            setStatus("Recording with automatic exposure/color. Locking first is strongly recommended.")
        }

        cameraController.startRecording(
            file = file,
            onStarted = {
                setStatus("Recording. Move slowly: lower ring → middle ring → upper ring.")
            },
            onProgress = { durationNanos, bytes ->
                val seconds = durationNanos / 1_000_000_000.0
                val mib = bytes / (1024.0 * 1024.0)
                statusText.text = "Recording ${"%.1f".format(seconds)} s • ${"%.1f".format(mib)} MiB"
            },
            onFinished = { completed ->
                recordButton.isEnabled = true
                stopButton.isEnabled = false
                uploadVideo(completed)
            },
            onError = { message ->
                recordButton.isEnabled = true
                stopButton.isEnabled = false
                setStatus("Recording failed: $message")
            },
        )
    }

    private fun testPcConnection() {
        testConnection.isEnabled = false
        setStatus("Testing PC receiver…")
        worker.execute {
            val result = runCatching { UploadClient.testConnection(serverUrl.text.toString()) }
            runOnUiThread {
                testConnection.isEnabled = true
                result.onSuccess {
                    setStatus("PC replied HTTP ${it.code}: ${it.body}")
                }.onFailure {
                    setStatus("Connection failed: ${it.message}. Check IP, firewall, and Wi‑Fi.")
                }
            }
        }
    }

    private fun uploadVideo(file: File) {
        val metadata = CaptureMetadata.from(file, cameraController.state).toJson()
        val baseUrl = serverUrl.text.toString()
        val sharedToken = token.text.toString()
        setStatus("Uploading ${formatBytes(file.length())} to PC…")

        worker.execute {
            val result = runCatching {
                UploadClient.upload(baseUrl, sharedToken, file, metadata) { progress ->
                    runOnUiThread { uploadProgress.progress = progress }
                }
            }
            runOnUiThread {
                result.onSuccess { response ->
                    if (response.code in 200..299) {
                        val deleted = file.delete()
                        currentFile = null
                        setStatus(
                            "Saved on PC. HTTP ${response.code}: ${response.body}" +
                                if (deleted) "\nLocal temporary copy removed." else "\nLocal copy: ${file.absolutePath}",
                        )
                    } else {
                        setStatus("Upload rejected HTTP ${response.code}: ${response.body}\nLocal copy: ${file.absolutePath}")
                    }
                }.onFailure {
                    setStatus("Upload failed: ${it.message}\nLocal copy kept at ${file.absolutePath}")
                }
            }
        }
    }

    private fun saveConnectionSettings() {
        getSharedPreferences("plant_capture", MODE_PRIVATE)
            .edit()
            .putString("server_url", serverUrl.text.toString().trim())
            .putString("token", token.text.toString())
            .apply()
    }

    private fun setStatus(message: String) {
        statusText.text = message
    }

    private fun formatBytes(bytes: Long): String {
        val mib = bytes / (1024.0 * 1024.0)
        return "${(mib * 10).roundToLong() / 10.0} MiB"
    }

    override fun onDestroy() {
        cameraController.close()
        worker.shutdownNow()
        super.onDestroy()
    }
}
