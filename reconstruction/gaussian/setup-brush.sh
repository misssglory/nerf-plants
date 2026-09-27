#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RECON_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
TOOLS_DIR="${PLANT_TOOLS_DIR:-$RECON_DIR/.tools}"
SOURCE_DIR="$TOOLS_DIR/src/brush"
INSTALL_DIR="$TOOLS_DIR/brush"
TARGET_DIR="${CARGO_TARGET_DIR:-$TOOLS_DIR/cargo-target/brush}"
BRUSH_VERSION="${BRUSH_VERSION:-v0.3.0}"
BUILD_VIEWER="${PLANT_BRUSH_BUILD_VIEWER:-1}"

for cmd in git cargo rustc; do
  command -v "$cmd" >/dev/null 2>&1 || {
    echo "Missing $cmd. Enter: nix develop .#gaussian" >&2
    exit 1
  }
done

mkdir -p "$TOOLS_DIR/src" "$INSTALL_DIR/bin" "$TARGET_DIR"

if [[ ! -d "$SOURCE_DIR/.git" ]]; then
  echo "Cloning Brush $BRUSH_VERSION..."
  git clone --filter=blob:none --depth 1 --branch "$BRUSH_VERSION" \
    https://github.com/ArthurBrussee/brush.git "$SOURCE_DIR"
else
  echo "Using existing Brush source: $SOURCE_DIR"
  git -C "$SOURCE_DIR" fetch --depth 1 origin tag "$BRUSH_VERSION" || true
  git -C "$SOURCE_DIR" checkout --detach "$BRUSH_VERSION"
fi

export CARGO_TARGET_DIR="$TARGET_DIR"
export CARGO_NET_GIT_FETCH_WITH_CLI=true

packages=(-p brush-cli)
if [[ "$BUILD_VIEWER" == "1" ]]; then
  packages+=(-p brush-app)
fi

printf 'Building Brush with rustc %s\n' "$(rustc --version)"
(
  cd "$SOURCE_DIR"
  cargo build --locked --release "${packages[@]}"
)

install -m755 "$TARGET_DIR/release/brush-cli" "$INSTALL_DIR/bin/brush-cli"
if [[ "$BUILD_VIEWER" == "1" ]]; then
  install -m755 "$TARGET_DIR/release/brush" "$INSTALL_DIR/bin/brush"
fi

cat > "$INSTALL_DIR/VERSION" <<VERSION
Brush source tag: $BRUSH_VERSION
Built at: $(date --iso-8601=seconds)
Rust: $(rustc --version)
VERSION

printf '\nBrush installed:\n  %s\n' "$INSTALL_DIR/bin/brush-cli"
if [[ -x "$INSTALL_DIR/bin/brush" ]]; then
  printf '  %s\n' "$INSTALL_DIR/bin/brush"
fi
printf '\nNext:\n  ./train-gaussian.sh plant_003 brush\n'
