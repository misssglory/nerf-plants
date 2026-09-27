# Gaussian splatting workflows

This project provides two separate Gaussian backends:

- **Brush** — cross-platform WebGPU training. This is the default for AMD Radeon
  780M, Intel and non-CUDA systems.
- **Nerfstudio Splatfacto** — Nerfstudio's CUDA/gsplat implementation. Use this
  on a supported NVIDIA GPU.

Nerfacto is not Gaussian splatting. It is a neural radiance field that evaluates
an MLP/hash-grid field along camera rays. Gaussian splatting stores explicit 3D
ellipsoidal Gaussians and rasterizes them into each view.

## AMD 780M: Brush

From the project root:

```bash
nix develop .#gaussian
cd reconstruction
./gaussian/check-gaussian.sh
./gaussian/setup-brush.sh
```

The setup script clones the pinned Brush `v0.3.0` source and builds both:

```text
reconstruction/.tools/brush/bin/brush-cli
reconstruction/.tools/brush/bin/brush
```

Train from an existing `ns-process-data` result:

```bash
./train-gaussian.sh plant_003 brush
```

A safer default profile for an integrated 780M is used:

```text
1280 px maximum image dimension
750,000 maximum splats
2 GiB image cache
30,000 iterations
validation every 500 iterations
PLY export every 5,000 iterations
```

Lower memory use:

```bash
PLANT_GS_MAX_RESOLUTION=960 \
PLANT_GS_MAX_SPLATS=400000 \
PLANT_GS_CACHE_SIZE=1GiB \
./train-gaussian.sh plant_003 brush 20000
```

Train with the native live viewer:

```bash
PLANT_GS_WITH_VIEWER=1 \
./train-gaussian.sh plant_003 brush
```

Open the latest exported PLY afterward:

```bash
./gaussian/view-brush.sh plant_003
```

Outputs are written to:

```text
outputs-gaussian/plant_003/brush/TIMESTAMP/
├── command.sh
├── train.log
├── plant_003_5000.ply
├── plant_003_10000.ply
└── latest.ply -> plant_003_....ply
```

Brush prints validation PSNR/SSIM and the current splat count. The wrapper does
not currently kill Brush automatically on a plateau; use the evaluation values
and exported snapshots to compare convergence.

## NVIDIA CUDA: Splatfacto

Enter the Nerfstudio shell and ensure its Pixi environment is installed:

```bash
nix develop .#nerfstudio
cd reconstruction
./setup.sh
./train-gaussian.sh plant_003 splatfacto
```

The existing validation early-stopping supervisor is reused for Splatfacto.
The web viewer normally opens on `http://localhost:7007`.

Export the trained Gaussian PLY:

```bash
./gaussian/export-splatfacto.sh plant_003
```

Open a saved Splatfacto checkpoint:

```bash
./gaussian/view-splatfacto.sh plant_003
```

## Automatic backend selection

```bash
./train-gaussian.sh plant_003
```

The wrapper selects Splatfacto only when the project's PyTorch runtime reports
CUDA as usable. Otherwise it selects Brush.

## Measurement warning

A Gaussian splat is ideal for interactive appearance and novel-view rendering,
but it is not a watertight triangle surface. Do not sum Gaussian scales to
claim physical leaf area. Use the splat for visualization and segmentation;
use a calibrated photogrammetric mesh for square-centimetre measurements.
