{
  description = "Rust egui/wgpu green-shape and edge composer";

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
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            cargo
            rustc
            rustfmt
            clippy
            pkg-config
            dbus
            openssl
            vulkan-loader
            vulkan-tools
          ];

          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath runtimeLibraries;

          # wgpu normally selects Vulkan automatically on Linux. This makes the
          # preferred backend explicit while still allowing WGPU_BACKEND to be
          # overridden before entering the shell.
          shellHook = ''
            export WGPU_BACKEND="''${WGPU_BACKEND:-vulkan}"
            echo "Rust green-shape edge composer (egui + wgpu)"
            echo "WGPU_BACKEND=$WGPU_BACKEND"
            echo "Run: cargo run --release -- /path/to/image.jpg"
            echo "GPU check: vulkaninfo --summary"
          '';
        };
      });
}
