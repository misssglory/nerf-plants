{
  description = "Plant Capture: Android capture plus Nerfstudio reconstruction tools on NixOS";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      supportedSystems = [ "x86_64-linux" "aarch64-linux" ];
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
    in {
      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs {
            inherit system;
            config = {
              allowUnfree = true;
              android_sdk.accept_license = true;
            };
          };

          androidComposition = pkgs.androidenv.composeAndroidPackages {
            platformVersions = [ "37" ];
            buildToolsVersions = [ "36.0.0" ];
            includeEmulator = false;
            includeSystemImages = false;
            includeNDK = false;
          };

          androidSdk = androidComposition.androidsdk;
          sdkRoot = "${androidSdk}/libexec/android-sdk";

          androidShell = pkgs.mkShell {
            packages = [
              pkgs.jdk17
              pkgs.gradle
              pkgs.android-tools
              androidSdk
            ];

            JAVA_HOME = "${pkgs.jdk17}";
            ANDROID_HOME = sdkRoot;
            ANDROID_SDK_ROOT = sdkRoot;

            # AGP normally downloads its own aapt2 binary. On NixOS it is
            # safer to force the patched aapt2 supplied by androidenv.
            GRADLE_OPTS = "-Dorg.gradle.project.android.aapt2FromMavenOverride=${sdkRoot}/build-tools/36.0.0/aapt2";

            shellHook = ''
              echo "Plant Capture Android CLI environment"
              echo "JAVA_HOME=$JAVA_HOME"
              echo "ANDROID_SDK_ROOT=$ANDROID_SDK_ROOT"
              echo "Build:   cd android && ./build-nixos.sh"
              echo "Install: cd android && ./install-nixos.sh"
            '';
          };

          # Binary Python/Conda packages such as Open3D and COLMAP are built
          # for conventional FHS Linux distributions. On NixOS their dynamic
          # libraries are not in a global /usr/lib search path, so expose the
          # common X11/OpenGL/runtime dependencies explicitly.
          reconstructionRuntimeLibraries = with pkgs; [
            stdenv.cc.cc.lib
            zlib
            openssl
            glib
            dbus
            fontconfig
            freetype
            libdrm
            libGL
            libglvnd
            libxkbcommon
            wayland
            xorg.libX11
            xorg.libXext
            xorg.libXrender
            xorg.libXi
            xorg.libXrandr
            xorg.libXfixes
            xorg.libXcursor
            xorg.libxcb
            xorg.libXau
            xorg.libXdmcp
          ];

          # Nerfstudio's own v1.1.5 Pixi environment pins Python 3.10,
          # PyTorch 2.2, CUDA 11.8 and COLMAP 3.9.x. The flake supplies Pixi
          # and host-side tools; reconstruction/pixi.toml supplies the Python
          # and CUDA user-space stack. Video preprocessing itself is CPU-only;
          # model training still normally requires a supported CUDA GPU.
          nerfstudioShell = pkgs.mkShell {
            packages = with pkgs; [
              pixi
              git
              git-lfs
              curl
              wget
              jq
              which
              file
              ffmpeg
              cmake
              ninja
              pkg-config
              gcc
              gnumake
            ];

            shellHook = ''
              export PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
              export PIXI_CACHE_DIR="''${PIXI_CACHE_DIR:-$HOME/.cache/rattler/cache}"
              export TORCH_EXTENSIONS_DIR="''${TORCH_EXTENSIONS_DIR:-$PROJECT_ROOT/.cache/torch_extensions}"
              mkdir -p "$PIXI_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"

              # Prebuilt Open3D/COLMAP wheels expect X11 and OpenGL libraries
              # in a conventional global loader path. NixOS intentionally has
              # no such /usr/lib path, so provide the exact Nix store paths.
              # The /run/opengl-driver entries expose the active GPU driver
              # when present; they are harmless on CPU-only preprocessing.
              export LD_LIBRARY_PATH="/run/opengl-driver/lib:/run/opengl-driver-32/lib:${pkgs.lib.makeLibraryPath reconstructionRuntimeLibraries}:''${LD_LIBRARY_PATH:-}"

              echo "Plant Capture Nerfstudio environment"
              echo "Setup:   cd reconstruction && ./setup.sh"
              echo "Process: cd reconstruction && ./process-video.sh VIDEO.mp4 NAME [FRAMES]"
              echo "Check:   cd reconstruction && ./check-accelerator.sh"
              echo "Train:   cd reconstruction && ./train.sh NAME nerfacto"
              echo "Export:  cd reconstruction && ./export-mesh.sh PATH/TO/config.yml NAME"
              echo
              if command -v nvidia-smi >/dev/null 2>&1; then
                nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true
              else
                echo "WARNING: nvidia-smi is unavailable. Nerfstudio training normally requires an NVIDIA CUDA GPU."
              fi
            '';
          };
        in
          {
            default = androidShell;
            android = androidShell;
          }
          // pkgs.lib.optionalAttrs (system == "x86_64-linux") {
            nerfstudio = nerfstudioShell;
          });
    };
}
