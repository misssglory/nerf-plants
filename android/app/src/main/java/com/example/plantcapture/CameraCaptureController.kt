package com.example.plantcapture

import android.content.Context
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraMetadata
import android.hardware.camera2.CaptureRequest
import android.util.Range
import androidx.camera.camera2.interop.Camera2CameraControl
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.CaptureRequestOptions
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.FocusMeteringAction
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.FallbackStrategy
import androidx.camera.video.FileOutputOptions
import androidx.camera.video.Quality
import androidx.camera.video.QualitySelector
import androidx.camera.video.Recorder
import androidx.camera.video.Recording
import androidx.camera.video.VideoCapture
import androidx.camera.video.VideoRecordEvent
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.io.File
import java.util.concurrent.Executor

@OptIn(ExperimentalCamera2Interop::class)
internal class CameraCaptureController(
    private val context: Context,
    private val eventExecutor: Executor,
) {
    data class CameraState(
        val qualityPreference: String = "UHD > FHD > HD",
        val targetFpsRange: String = "camera default",
        val aeAwbLocked: Boolean = false,
        val stabilizationRequestedOff: Boolean = false,
    )

    private var camera: Camera? = null
    private var videoCapture: VideoCapture<Recorder>? = null
    private var recording: Recording? = null
    private var selectedFpsRange: Range<Int>? = null
    private var canDisableVideoStabilization = false
    private var canDisableOpticalStabilization = false

    var state: CameraState = CameraState()
        private set

    val isRecording: Boolean
        get() = recording != null

    fun startCamera(
        owner: LifecycleOwner,
        previewView: PreviewView,
        onReady: (CameraState) -> Unit,
        onError: (Throwable) -> Unit,
    ) {
        val providerFuture = ProcessCameraProvider.getInstance(context)
        providerFuture.addListener(
            {
                try {
                    val provider = providerFuture.get()
                    val preview = Preview.Builder().build().also {
                        it.surfaceProvider = previewView.surfaceProvider
                    }

                    val qualitySelector = QualitySelector.fromOrderedList(
                        listOf(Quality.UHD, Quality.FHD, Quality.HD),
                        FallbackStrategy.lowerQualityOrHigherThan(Quality.SD),
                    )
                    val recorder = Recorder.Builder()
                        .setQualitySelector(qualitySelector)
                        .build()
                    val capture = VideoCapture.withOutput(recorder)

                    provider.unbindAll()
                    camera = provider.bindToLifecycle(
                        owner,
                        CameraSelector.DEFAULT_BACK_CAMERA,
                        preview,
                        capture,
                    )
                    videoCapture = capture
                    inspectAndApplyBaseCameraOptions()
                    onReady(state)
                } catch (t: Throwable) {
                    onError(t)
                }
            },
            ContextCompat.getMainExecutor(context),
        )
    }

    fun focusAt(previewView: PreviewView, x: Float, y: Float, lockAfterFocus: Boolean, done: (Boolean) -> Unit) {
        val activeCamera = camera ?: run {
            done(false)
            return
        }
        val point = previewView.meteringPointFactory.createPoint(x, y, 0.22f)
        val flags = FocusMeteringAction.FLAG_AF or
            FocusMeteringAction.FLAG_AE or
            FocusMeteringAction.FLAG_AWB
        val action = FocusMeteringAction.Builder(point, flags)
            .disableAutoCancel()
            .build()
        val future = activeCamera.cameraControl.startFocusAndMetering(action)
        future.addListener(
            {
                val focused = runCatching { future.get().isFocusSuccessful }.getOrDefault(false)
                if (lockAfterFocus) {
                    applyLocks(true) { done(focused) }
                } else {
                    done(focused)
                }
            },
            eventExecutor,
        )
    }

    fun focusCenterAndLock(previewView: PreviewView, done: (Boolean) -> Unit) {
        focusAt(
            previewView = previewView,
            x = previewView.width / 2f,
            y = previewView.height / 2f,
            lockAfterFocus = true,
            done = done,
        )
    }

    fun unlock(done: (Boolean) -> Unit = {}) {
        applyLocks(false, done)
    }

    fun startRecording(
        file: File,
        onStarted: () -> Unit,
        onProgress: (durationNanos: Long, bytes: Long) -> Unit,
        onFinished: (file: File) -> Unit,
        onError: (String) -> Unit,
    ) {
        if (recording != null) {
            onError("A recording is already active")
            return
        }
        val capture = videoCapture ?: run {
            onError("Camera is not ready")
            return
        }
        file.parentFile?.mkdirs()
        val output = FileOutputOptions.Builder(file).build()
        recording = capture.output
            .prepareRecording(context, output)
            .start(eventExecutor) { event ->
                when (event) {
                    is VideoRecordEvent.Start -> onStarted()
                    is VideoRecordEvent.Status -> onProgress(
                        event.recordingStats.recordedDurationNanos,
                        event.recordingStats.numBytesRecorded,
                    )
                    is VideoRecordEvent.Finalize -> {
                        recording = null
                        if (event.hasError()) {
                            onError("CameraX finalize error ${event.error}: ${event.cause?.message.orEmpty()}")
                        } else {
                            onFinished(file)
                        }
                    }
                    else -> Unit
                }
            }
    }

    fun stopRecording() {
        recording?.stop()
    }

    fun close() {
        recording?.stop()
        recording = null
    }

    private fun inspectAndApplyBaseCameraOptions() {
        val activeCamera = camera ?: return
        val info = Camera2CameraInfo.from(activeCamera.cameraInfo)

        val fpsRanges = info.getCameraCharacteristic(
            CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES,
        ).orEmpty()
        selectedFpsRange = chooseThirtyFpsRange(fpsRanges)

        val videoStabilizationModes = info.getCameraCharacteristic(
            CameraCharacteristics.CONTROL_AVAILABLE_VIDEO_STABILIZATION_MODES,
        ) ?: intArrayOf()
        canDisableVideoStabilization = videoStabilizationModes.contains(
            CameraMetadata.CONTROL_VIDEO_STABILIZATION_MODE_OFF,
        )

        val opticalStabilizationModes = info.getCameraCharacteristic(
            CameraCharacteristics.LENS_INFO_AVAILABLE_OPTICAL_STABILIZATION,
        ) ?: intArrayOf()
        canDisableOpticalStabilization = opticalStabilizationModes.contains(
            CameraMetadata.LENS_OPTICAL_STABILIZATION_MODE_OFF,
        )

        state = state.copy(
            targetFpsRange = selectedFpsRange?.let { "${it.lower}-${it.upper} fps" } ?: "camera default",
            stabilizationRequestedOff = canDisableVideoStabilization || canDisableOpticalStabilization,
        )
        setCamera2Options(aeAwbLocked = false, callback = {})
    }

    private fun chooseThirtyFpsRange(ranges: Array<out Range<Int>>): Range<Int>? {
        return ranges.firstOrNull { it.lower == 30 && it.upper == 30 }
            ?: ranges
                .filter { it.contains(30) }
                .minByOrNull { it.upper - it.lower }
    }

    private fun applyLocks(locked: Boolean, callback: (Boolean) -> Unit) {
        setCamera2Options(aeAwbLocked = locked) { success ->
            if (success) {
                state = state.copy(aeAwbLocked = locked)
            }
            callback(success)
        }
    }

    private fun setCamera2Options(aeAwbLocked: Boolean, callback: (Boolean) -> Unit) {
        val activeCamera = camera ?: run {
            callback(false)
            return
        }
        val builder = CaptureRequestOptions.Builder()
        selectedFpsRange?.let {
            builder.setCaptureRequestOption(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, it)
        }
        if (canDisableVideoStabilization) {
            builder.setCaptureRequestOption(
                CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE,
                CameraMetadata.CONTROL_VIDEO_STABILIZATION_MODE_OFF,
            )
        }
        if (canDisableOpticalStabilization) {
            builder.setCaptureRequestOption(
                CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE,
                CameraMetadata.LENS_OPTICAL_STABILIZATION_MODE_OFF,
            )
        }
        builder.setCaptureRequestOption(CaptureRequest.CONTROL_AE_LOCK, aeAwbLocked)
        builder.setCaptureRequestOption(CaptureRequest.CONTROL_AWB_LOCK, aeAwbLocked)

        val future = Camera2CameraControl.from(activeCamera.cameraControl)
            .setCaptureRequestOptions(builder.build())
        future.addListener(
            { callback(runCatching { future.get(); true }.getOrDefault(false)) },
            eventExecutor,
        )
    }
}
