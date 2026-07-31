# Plant Capture: Android → Linux LAN video transfer

This repository contains:

- `server/`: a dependency-free Python receiver for Linux.
- `android/`: an Android CameraX app that records a full-quality MP4 locally, then uploads it to the Linux PC over the local network.

The MVP intentionally does **not** live-stream the camera. Local recording avoids dropped network frames and preserves the original encoded video for photogrammetry.

## 1. Start the Linux receiver

```bash
cd server
export PLANT_CAPTURE_TOKEN='replace-this-with-a-long-random-token'
python3 server.py --output "$HOME/PlantCaptures" --port 8765 --token "$PLANT_CAPTURE_TOKEN"
```

The server prints an address similar to:

```text
Phone server URL: http://192.168.1.42:8765
```

Use that exact URL and the same token in the Android app.

Test locally:

```bash
curl http://127.0.0.1:8765/health
```

If Linux uses UFW:

```bash
sudo ufw allow from 192.168.0.0/16 to any port 8765 proto tcp
```

Adjust the subnet if your LAN is different. Do not expose this HTTP server to the public internet.

Each successful upload creates:

```text
plant_YYYYMMDD_HHMMSS.mp4
plant_YYYYMMDD_HHMMSS.mp4.json
```

The JSON sidecar includes the SHA-256, byte count, receiving time, phone model, requested quality, FPS range, and camera-lock state.

## 2. Build the Android app

Recommended build environment:

- Android Studio with Android 17 / API 37 installed
- JDK 17
- Android Gradle Plugin 9.3.0
- Gradle 9.5
- CameraX 1.5.3

Open the `android` directory in Android Studio, let Gradle sync, connect the phone with USB debugging, and press **Run**.

The repository includes `gradle-wrapper.properties`, but not the binary wrapper JAR. Android Studio can sync the project directly. To add command-line wrapper files, run `gradle wrapper` once from a machine with Gradle installed.

### Permissions

The app requests:

- Camera permission
- Android 17 local-network permission when required
- Internet/network access

No microphone or shared-storage permission is used. The temporary MP4 is kept in app-private storage until the PC confirms the upload.

## 3. Capture procedure

1. Put the PC and phone on the same non-guest Wi-Fi network.
2. Use diffuse, stable illumination and stop fans or drafts.
3. Put a scale marker and color calibration card near the plant, in the same depth range.
4. Enter the PC URL and token; press **Test PC connection**.
5. Frame the complete plant. Tap the plant to meter/focus.
6. Press **Focus center + lock exposure/color**.
7. Press **Record** and hold the first view for 2–3 seconds with the scale/color card visible.
8. Walk slowly around the plant:
   - lower ring,
   - middle ring,
   - upper ring,
   - several top-oblique views.
9. Keep approximately 70–85% visual overlap and avoid sudden rotations.
10. Press **Stop + upload**. The phone finalizes the MP4 and sends it to the PC.

A 45–90 second capture is a reasonable first test. UHD video can be hundreds of megabytes, so use strong Wi-Fi and verify free storage on both devices.

## Camera choices in this MVP

- Back camera
- UHD preferred, then FHD, then HD
- 30–30 FPS requested when the camera advertises that range; otherwise the narrowest supported range containing 30 FPS
- Exposure and white balance can be locked after metering
- Electronic and optical stabilization are requested off when the device advertises an OFF mode
- Audio disabled
- No HDR or beautification requested

Camera vendors may ignore or reinterpret some Camera2 controls. The app records the requested state in metadata, but later versions should also capture per-frame Camera2 results for scientific auditing.

## Security note

The MVP uses HTTP on a trusted private LAN plus a shared bearer token. The token prevents casual unauthenticated uploads but does **not** encrypt the video. A production version should use TLS with device pairing, certificate pinning, or a mutually authenticated protocol.

## Next pipeline stage

The next PC-side stage should:

1. inspect the MP4 with `ffprobe`,
2. extract sharp frames at an adaptive interval,
3. reject blur and near-duplicates,
4. color-calibrate frames from the reference card,
5. run COLMAP / AliceVision / another SfM-MVS pipeline,
6. segment leaves in 2D and transfer masks to the mesh,
7. calculate one-sided leaf area and area-weighted greenness statistics.

---

## NixOS command-line build and installation (no Android Studio)

The project root now contains `flake.nix`. It supplies JDK 17, Android SDK API 37,
Build Tools 36.0.0, Gradle, and adb.

```bash
cd plant_capture
nix develop
cd android
./install-nixos.sh
```

The first run generates the missing Gradle 9.5.0 wrapper, downloads Maven
artifacts, builds `app-debug.apk`, installs it on the authorized USB-connected
phone, and starts the app.

To build without installing:

```bash
nix develop
cd android
./build-nixos.sh
```

APK location:

```text
android/app/build/outputs/apk/debug/app-debug.apk
```

To inspect runtime logs:

```bash
cd android
./logcat-nixos.sh
```

### Phone preparation

1. Open **Settings → About phone** and tap **Build number** seven times.
2. Open **Developer options** and enable **USB debugging**.
3. Connect the phone using a data-capable USB cable.
4. Accept the RSA authorization prompt.
5. Verify with `adb devices -l`.

On current NixOS/systemd releases, `android-tools` normally receives device
access through systemd uaccess. On older NixOS releases, enable the legacy ADB
module and add your user to `adbusers`:

```nix
{
  programs.adb.enable = true;
  users.users.YOUR_USER.extraGroups = [ "adbusers" ];
}
```

Log out and back in after changing group membership.

## Gradle wrapper bootstrap note

`build-nixos.sh` creates a missing wrapper inside an isolated empty Gradle
project. This is intentional: running the Gradle package from nixpkgs unstable
directly against the Android project can load AGP 9.3 with a newer incompatible
Gradle before the wrapper task starts. The actual Android build always runs
through the wrapper pinned to Gradle 9.5.0.


## Kotlin 2.3 / Android Context compatibility

`MainActivity` deliberately names its cached UI executor `uiExecutor`, not
`mainExecutor`. A Kotlin property named `mainExecutor` generates a JVM getter
called `getMainExecutor()`, which collides with the method inherited by
`AppCompatActivity` from Android `Context`.

---

## Nerfstudio reconstruction on NixOS

An additional shell and complete video-to-model workflow now live in
`reconstruction/`.

```bash
nix develop .#nerfstudio
cd reconstruction
./setup.sh
./process-video.sh "$HOME/PlantCaptures/capture.mp4" plant_001 120
./train.sh plant_001 nerfacto
CONFIG="$(./latest-config.sh plant_001)"
./export-mesh.sh "$CONFIG" plant_001 poisson
```

See [`reconstruction/README.md`](reconstruction/README.md) for Splatfacto,
mesh export, scaling and troubleshooting notes. The existing Android shell is
still the default; it can also be entered explicitly with:

```bash
nix develop .#android
```
