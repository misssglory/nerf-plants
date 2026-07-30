package com.example.plantcapture

import android.os.Build
import org.json.JSONObject
import java.io.File
import java.time.Instant

internal data class CaptureMetadata(
    val capturedAtUtc: String,
    val device: String,
    val appVersion: String,
    val requestedQuality: String,
    val targetFpsRange: String,
    val aeAwbLocked: Boolean,
    val stabilizationRequestedOff: Boolean,
    val fileBytes: Long,
) {
    fun toJson(): String = JSONObject()
        .put("captured_at_utc", capturedAtUtc)
        .put("device", device)
        .put("app_version", appVersion)
        .put("requested_quality", requestedQuality)
        .put("target_fps_range", targetFpsRange)
        .put("ae_awb_locked", aeAwbLocked)
        .put("stabilization_requested_off", stabilizationRequestedOff)
        .put("file_bytes", fileBytes)
        .put(
            "capture_notes",
            "Back camera; prefer UHD then FHD; fixed 30 fps when supported; no audio; " +
                "capture scale and color card at start; move slowly around a still plant.",
        )
        .toString()

    companion object {
        fun from(file: File, cameraState: CameraCaptureController.CameraState): CaptureMetadata {
            return CaptureMetadata(
                capturedAtUtc = Instant.now().toString(),
                device = "${Build.MANUFACTURER} ${Build.MODEL}",
                appVersion = BuildConfig.VERSION_NAME,
                requestedQuality = cameraState.qualityPreference,
                targetFpsRange = cameraState.targetFpsRange,
                aeAwbLocked = cameraState.aeAwbLocked,
                stabilizationRequestedOff = cameraState.stabilizationRequestedOff,
                fileBytes = file.length(),
            )
        }
    }
}
