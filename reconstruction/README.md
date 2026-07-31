# Video to Nerfstudio model on NixOS

## Requirements

- x86_64 NixOS
- NVIDIA GPU and working proprietary/open NVIDIA driver
- `nvidia-smi` must work on the host
- enough disk space: the Pixi/CUDA environment and outputs can consume many GB

This directory uses the dependency versions from Nerfstudio 1.1.5's official
Pixi setup: Python 3.10, PyTorch 2.2, CUDA 11.8, and COLMAP 3.9.1.

## 1. Enter the shell

From the project root:

```bash
nix develop .#nerfstudio
cd reconstruction
```

## 2. Install the pinned environment

```bash
./setup.sh
```

## 3. Convert a phone video into a Nerfstudio dataset

```bash
./process-video.sh \
  "$HOME/PlantCaptures/plant_20260730_221500.mp4" \
  plant_001 \
  120
```

The final argument is the target number of extracted frames. Start around
80-160 for a slow 45-90 second orbit. Inspect `data/processed/plant_001/images`
and remove the dataset/reprocess if the frames are blurred or repetitive.

## 4A. Train Nerfacto for mesh export

```bash
./train.sh plant_001 nerfacto
```

The wrapper enables predicted normals so Poisson export is available.
Nerfstudio prints the viewer URL, normally on port 7007.

Find the newest config:

```bash
CONFIG="$(./latest-config.sh plant_001)"
echo "$CONFIG"
```

Export a mesh:

```bash
./export-mesh.sh "$CONFIG" plant_001 poisson
```

If Poisson fails or looks overly smooth:

```bash
./export-mesh.sh "$CONFIG" plant_001_tsdf tsdf
```

## 4B. Train Splatfacto for a fast visual model

```bash
./train.sh plant_001 splatfacto
CONFIG="$(./latest-config.sh plant_001)"
./export-splat.sh "$CONFIG" plant_001
```

Splatfacto usually trains faster and gives excellent novel views, but a
Gaussian splat is not a triangle mesh and should not be used directly for leaf
surface-area measurements.

## Scale and plant-measurement warning

Nerfstudio normalizes the scene. The exported geometry is not automatically in
centimetres. Keep a rigid marker of known length in the capture and use it to
rescale the final mesh before computing area.

Leaves are thin, partially specular, repetitive and easily moved by air. These
properties are hostile to both COLMAP and NeRF meshing. Use still air, diffuse
light, fixed focus/exposure/white balance, a textured background, and a slow
multi-height orbit. For metric leaf area, compare the result against a classic
COLMAP + OpenMVS or Metashape reconstruction.

## RTX 50-series / Blackwell note

The official Nerfstudio 1.1.5 environment is CUDA 11.8-era software. It can be
incompatible with GPUs requiring newer CUDA architecture support. If setup
fails while compiling tiny-cuda-nn or reports an unsupported compute
capability, use a modernized source environment, Splatfacto with a current
PyTorch/gsplat stack, or the COLMAP/OpenMVS route. Do not silently fall back to
CPU: training would be impractically slow.

## NumPy compatibility

This environment pins NumPy 1.26 because its pinned PyTorch 2.2/Open3D stack
contains binaries built against the NumPy 1.x ABI. Run
`./check-python-runtime.sh` to verify the torch/NumPy bridge before processing.
`process-video.sh` now removes only incomplete output directories that lack
`transforms.json`, and refuses to overwrite complete datasets.

## CPU compatibility mode and AMD integrated graphics

`train.sh` now checks `torch.cuda.is_available()` before launching Nerfstudio.
When no CUDA-capable NVIDIA GPU is present, Nerfacto is started with:

- `--machine.device-type cpu`
- `--mixed-precision False`
- the portable PyTorch field implementation
- reduced image scale and rays per batch
- 1000 iterations by default

Override the defaults with `PLANT_CPU_ITERATIONS`, `PLANT_CPU_RAYS`, and
`PLANT_CPU_IMAGE_SCALE`. This is intended for validation only; it does not use
the Radeon 780M iGPU. Splatfacto fails early without CUDA because the pinned
Nerfstudio/gsplat stack is CUDA-oriented.

Run `./check-accelerator.sh` to see what this Pixi PyTorch installation can use.


## Protobuf and TensorBoard compatibility

Nerfstudio 1.1.5 requires `protobuf<=3.20.3` and excludes 3.20.0. Modern
TensorBoard packages may pull protobuf 6.x, which makes Pixi report an
unsatisfiable Conda/PyPI solve before training starts. This project therefore
pins the compatible pair explicitly:

```text
protobuf 3.20.3
tensorboard 2.14.1
```

After applying this update to an existing checkout, discard the stale lock and
recreate the environment once:

```bash
cd reconstruction
rm -f pixi.lock
pixi install
pixi run python verify_environment.py
```

The verification output must report protobuf 3.20.3 before `train.sh` is run.

## Validation early stopping

`train.sh` enables early stopping by default. It reserves every eighth registered
video frame for validation, runs a full held-out render every 500 steps, and
stops after four consecutive validations without either:

- a PSNR increase of at least 0.03 dB; or
- an LPIPS decrease of at least 0.002.

The first 1000 steps are a warm-up and never trigger stopping. A hard maximum
still protects against an endless run: 10000 steps on CPU and 30000 on CUDA by
default.

```bash
./train.sh plant_001 nerfacto
```

Typical report:

```text
[quality] step=   2500  train_loss=0.04210 Δ=-0.00120 (-2.77%)  eval_loss=0.05190 Δ=-0.00030 (-0.57%)
          PSNR=24.3812 dB  SSIM=0.8124  LPIPS=0.17320  status=IMPROVED (PSNR +0.0812 dB)  patience=0/4
```

Training loss is printed for diagnosis, but stopping is based on held-out render
quality. A falling training loss alone can mean overfitting.

Results are written inside the run directory:

```text
outputs/plant_001/nerfacto/TIMESTAMP/
├── early_stopping.csv
├── early_stopping.json
├── best_checkpoint/
│   ├── best.ckpt
│   ├── best.json
│   └── config.yml
└── nerfstudio_models/
```

Useful overrides:

```bash
# Require six stale validations instead of four.
PLANT_EARLY_STOP_PATIENCE=6 ./train.sh plant_001 nerfacto

# Evaluate less often when CPU full-image renders are too expensive.
PLANT_EVAL_INTERVAL=1000 ./train.sh plant_001 nerfacto

# Keep the browser viewer in addition to TensorBoard metrics.
PLANT_TRAIN_VIS=viewer+tensorboard ./train.sh plant_001 nerfacto

# Disable early stopping for a short fixed smoke test.
PLANT_EARLY_STOP=0 PLANT_MAX_ITERATIONS=200 ./train.sh plant_001 nerfacto
```

After changing `pixi.toml`, run `pixi install` once. The supervisor uses the
TensorBoard event reader to observe Nerfstudio's `Train Loss`, `Eval Loss`, and
all-image PSNR/SSIM/LPIPS metrics while training is running.
