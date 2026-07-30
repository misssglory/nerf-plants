#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ -z "${ANDROID_SDK_ROOT:-}" ]]; then
  echo "ANDROID_SDK_ROOT is not set. Enter the project shell first:" >&2
  echo "  nix develop" >&2
  exit 1
fi

required_gradle="9.5.0"
wrapper_jar="gradle/wrapper/gradle-wrapper.jar"

bootstrap_wrapper() {
  local tmp
  tmp="$(mktemp -d)"
  trap 'rm -rf "${tmp:-}"' RETURN

  echo "Bootstrapping the Gradle ${required_gradle} wrapper in an isolated project..."

  # Do not run the host Gradle against this Android project. Nixpkgs unstable
  # may provide a newer Gradle that cannot configure AGP 9.3.0. Generate the
  # wrapper in an empty build, then let the wrapper itself run Gradle 9.5.0.
  cat > "$tmp/settings.gradle.kts" <<'SETTINGS'
rootProject.name = "plant-capture-wrapper-bootstrap"
SETTINGS

  gradle --no-daemon -p "$tmp" wrapper \
    --gradle-version "$required_gradle" \
    --distribution-type bin

  cp "$tmp/gradlew" ./gradlew
  cp "$tmp/gradlew.bat" ./gradlew.bat
  mkdir -p gradle/wrapper
  cp "$tmp/gradle/wrapper/gradle-wrapper.jar" "$wrapper_jar"
  cp "$tmp/gradle/wrapper/gradle-wrapper.properties" \
    gradle/wrapper/gradle-wrapper.properties
  chmod +x ./gradlew
}

if [[ ! -x ./gradlew || ! -f "$wrapper_jar" ]]; then
  bootstrap_wrapper
fi

actual_url="$(sed -n 's/^distributionUrl=//p' gradle/wrapper/gradle-wrapper.properties)"
if [[ "$actual_url" != *"gradle-${required_gradle}-bin.zip" ]]; then
  echo "Gradle wrapper points to an unexpected distribution:" >&2
  echo "  $actual_url" >&2
  echo "Expected Gradle ${required_gradle}. Remove ./gradlew and ${wrapper_jar}, then retry." >&2
  exit 1
fi

# This invocation uses exactly Gradle 9.5.0, not the possibly newer Gradle
# package from nixpkgs.
./gradlew --version
./gradlew --no-daemon :app:assembleDebug

apk="app/build/outputs/apk/debug/app-debug.apk"
if [[ ! -f "$apk" ]]; then
  echo "Build completed but APK was not found at $apk" >&2
  exit 1
fi

printf '\nAPK created:\n  %s\n' "$(realpath "$apk")"
