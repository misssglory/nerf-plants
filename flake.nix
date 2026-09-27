{
  description = "Plant Capture: Android capture, Nerfstudio and Gaussian reconstruction tools on NixOS";

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
            GRADLE_OPTS = "-Dorg.gradle.project.android.aapt2FromMavenOverride=${sdkRoot}/build-tools/36.0.0/aapt2";

            shellHook = ''
              echo "Plant Capture Android CLI environment"
              echo "JAVA_HOME=$JAVA_HOME"
              echo "ANDROID_SDK_ROOT=$ANDROID_SDK_ROOT"
              echo "Build:   cd android && ./build-nixos.sh"
              echo "Install: cd android && ./install-nixos.sh"
            '';
          };

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
              export LD_LIBRARY_PATH="/run/opengl-driver/lib:/run/opengl-driver-32/lib:${pkgs.lib.makeLibraryPath reconstructionRuntimeLibraries}:''${LD_LIBRARY_PATH:-}"

              echo "Plant Capture Nerfstudio environment"
              echo "Setup:    cd reconstruction && ./setup.sh"
              echo "Process:  cd reconstruction && ./process-video.sh VIDEO.mp4 NAME [FRAMES]"
              echo "NeRF:     cd reconstruction && ./train.sh NAME nerfacto"
              echo "Gaussian: cd reconstruction && ./train-gaussian.sh NAME splatfacto"
              echo "Export:   cd reconstruction && ./export-mesh.sh PATH/TO/config.yml NAME"
              echo
              if command -v nvidia-smi >/dev/null 2>&1; then
                nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true
              else
                echo "No NVIDIA CUDA driver detected. Use nix develop .#gaussian for AMD/Intel Brush training."
              fi
            '';
          };

          gaussianRuntimeLibraries = reconstructionRuntimeLibraries ++ (with pkgs; [
            vulkan-loader
            vulkan-validation-layers
            shaderc
            udev
            alsa-lib
          ]);

          # Brush uses wgpu/WebGPU rather than CUDA, so this shell works with
          # AMD, Intel and NVIDIA Vulkan drivers. It also includes Pixi so the
          # auto wrapper can still detect and dispatch to Splatfacto on CUDA.
          gaussianShell = pkgs.mkShell {
            packages = with pkgs; [
              rustc
              cargo
              rustfmt
              clippy
              git
              git-lfs
              curl
              jq
              which
              file
              ffmpeg
              pixi
              pkg-config
              cmake
              ninja
              llvmPackages.clang
              llvmPackages.lld
              vulkan-tools
              vulkan-loader
              vulkan-validation-layers
              shaderc
            ];

            LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
            RUST_BACKTRACE = "1";
            CARGO_NET_GIT_FETCH_WITH_CLI = "true";

            shellHook = ''
              export PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
              export PLANT_TOOLS_DIR="''${PLANT_TOOLS_DIR:-$PROJECT_ROOT/reconstruction/.tools}"
              export CARGO_HOME="''${CARGO_HOME:-$HOME/.cargo}"
              export CARGO_TARGET_DIR="''${CARGO_TARGET_DIR:-$PLANT_TOOLS_DIR/cargo-target/brush}"
              export WGPU_BACKEND="''${WGPU_BACKEND:-vulkan}"
              export LD_LIBRARY_PATH="/run/opengl-driver/lib:/run/opengl-driver-32/lib:${pkgs.lib.makeLibraryPath gaussianRuntimeLibraries}:''${LD_LIBRARY_PATH:-}"
              export VK_LAYER_PATH="/run/opengl-driver/share/vulkan/explicit_layer.d:${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d:''${VK_LAYER_PATH:-}"
              mkdir -p "$PLANT_TOOLS_DIR" "$CARGO_TARGET_DIR"

              echo "Plant Capture Gaussian environment (Brush/WebGPU)"
              echo "GPU check: cd reconstruction && ./gaussian/check-gaussian.sh"
              echo "Setup:     cd reconstruction && ./gaussian/setup-brush.sh"
              echo "Train:     cd reconstruction && ./train-gaussian.sh NAME brush"
              echo "View:      cd reconstruction && ./gaussian/view-brush.sh NAME"
              echo
              vulkaninfo --summary 2>/dev/null | sed -n '1,24p' || \
                echo "WARNING: Vulkan is unavailable. Check /run/opengl-driver and your Mesa/AMD driver."
            '';
          };
        in
          {
            default = androidShell;
            android = androidShell;
            gaussian = gaussianShell;
          }
          // pkgs.lib.optionalAttrs (system == "x86_64-linux") {
            nerfstudio = nerfstudioShell;
          });
    };
}
