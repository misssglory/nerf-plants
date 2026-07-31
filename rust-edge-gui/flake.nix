{
  description = "Rust egui Canny edge detector development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        runtimeLibraries = with pkgs; [
          libGL
          libxkbcommon
          wayland
          xorg.libX11
          xorg.libXcursor
          xorg.libXi
          xorg.libXrandr
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
          ];

          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath runtimeLibraries;

          shellHook = ''
            echo "Rust egui edge detector"
            echo "Run: cargo run --release -- /path/to/image.jpg"
          '';
        };
      });
}
