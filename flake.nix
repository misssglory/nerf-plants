{
  description = "NixOS CLI environment for Plant Capture Android app";

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
        in {
          default = pkgs.mkShell {
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
        });
    };
}
