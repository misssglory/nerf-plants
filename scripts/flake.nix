{
  description = "Python OpenCV image edge detection environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
      ];

      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
    in
    {
      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs {
            inherit system;
          };

          python = pkgs.python312.withPackages (pythonPackages: with pythonPackages; [
            opencv4
            numpy
          ]);
        in
        {
          default = pkgs.mkShell {
            packages = [
              python
            ];

            shellHook = ''
              echo "OpenCV edge detection environment"
              echo "Python: $(python --version)"
              echo "OpenCV: $(python -c 'import cv2; print(cv2.__version__)')"
              echo
              echo "Run:"
              echo "  python edge_detect.py image.jpg"
            '';
          };
        });
    };
}