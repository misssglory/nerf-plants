{
  description = "rust-edge-gui development shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        runtimeLibraries = with pkgs; [
          vulkan-loader
          libGL
          libxkbcommon
          wayland
          xorg.libX11
          xorg.libXcursor
          xorg.libXi
          xorg.libXrandr
          xorg.libxcb
          xorg.libXext
          xorg.libXfixes
          xorg.libXrender
        ];
      in {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            cargo
            rustc
            rustfmt
            clippy
            pkg-config
            dbus
            networkmanager
            openssl
            vulkan-loader
            vulkan-tools
          ];
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath runtimeLibraries;
          shellHook = ''
            export WGPU_BACKEND="''${WGPU_BACKEND:-vulkan}"
            echo "rust-edge-gui"
            echo "Run: cargo run --release"
          '';
        };
      });
}
