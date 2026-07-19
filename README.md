# NRC for Moer Renderer

Neural Radiance Cache (NRC) module for Moer. This directory contains:

- dataset collection tools from path tracing
- tiny-cuda-nn based training scripts
- NRC-assisted rendering executables
- dataset inspection and visualization utilities

This README explains the full workflow after Moer is configured: **collect -> train -> render**.

## Table of Contents

- [1. Prerequisites](#1-prerequisites)
- [2. Build with NRC](#2-build-with-nrc)
- [3. Scene Configuration](#3-scene-configuration)
- [4. Collect NRC Dataset](#4-collect-nrc-dataset)
- [5. Install tiny-cuda-nn (TCNN)](#5-install-tiny-cuda-nn-tcnn)
- [6. Train NRC Network](#6-train-nrc-network)
- [7. Run NRC Rendering](#7-run-nrc-rendering)
- [8. Utilities](#8-utilities)
- [9. Troubleshooting](#9-troubleshooting)

## 1. Prerequisites

- A working [Moer renderer](https://github.com/NJUCG/Moer) setup and build toolchain (CMake + C++17 compiler).
- Python 3.10+ recommended.
- NVIDIA GPU + CUDA for training/inference with tiny-cuda-nn.
- Python packages:
  - `torch` (CUDA build)
  - `numpy`
  - `tqdm`
  - `matplotlib` (for curves/plots)
  - optional: `commentjson`

> Note: current C++ runtime default command for python inference is:
> `conda run -n moer-build python`
> You can override it via `renderer.nrc.python_command`.

## 2. Build with NRC

At project root:

```bash
cmake -S . -B build -DENABLE_NRC=ON
cmake --build build --config Release
```

NRC-related executables are:

- `Moer-NRC-Collect`
- `Moer-NRC-Render`
- `Moer-NRC-FirstHit`

On Windows with multi-config generators, binaries are typically under `target/bin` with postfix like `_r` for Release.

Typical executable paths after a Release build:

- `target/bin/Moer-NRC-Collect_r.exe`
- `target/bin/Moer-NRC-Render_r.exe`
- `target/bin/Moer-NRC-FirstHit_r.exe`

## 3. Scene Configuration

NRC reads config from `scene.json -> renderer.nrc`.

Minimal example:

```json
{
  "renderer": {
    "spp": 128,
    "output_file": "classroom",
    "nrc": {
      "views_file": "nrc_views.json",
      "threads": 12,
      "flush_threshold": 32768,
      "max_bounce_record": 8,
      "global_keep": 0.25,
      "bounce_keep_probs": [1.0, 0.5, 0.25, 0.25, 0.1, 0.1, 0.1, 0.1],
      "test_views_file": "nrc_test_views.json",
      "checkpoint_file": "NRC/runtime/checkpoints/classroom/nrc_best.pt",
      "pt_max_bounce": 1,
      "energy_gain": 1.0,
      "firsthit_energy_gain": 1.0
    }
  }
}
```

Important keys:

- `views_file`: training data collection views.
- `test_views_file`: evaluation/render views.
- `checkpoint_file`: trained checkpoint for rendering.
- `max_bounce_record`: bounces beyond this value are dropped during collection.
- `bounce_keep_probs[i]`: keep probability for bounce `i+1`.
- `global_keep`: global multiplier on keep probability.

Supported `views_file` formats are documented in `NRC/doc/nrc_debug.md`.

## 4. Collect NRC Dataset

Run collector with scene directory:

```bash
target/bin/Moer-NRC-Collect_r.exe scenes/classroom
```

Typical outputs:

- `NRC/runtime/scenes/<scene_name>/_view_*.bin`
- `NRC/runtime/scenes/<scene_name>/_bounce_stats.txt`

Dataset binary format is documented in `NRC/doc/data_format.md`.

If `views_file` is not set, only `camera.transform` in `scene.json` is used.

## 5. Install tiny-cuda-nn (TCNN)

TCNN is required by `NRC/python/train_nrc.py` and `NRC/python/infer_nrc_batch.py`.

Recommended (inside your chosen Python environment):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install numpy tqdm matplotlib
```

Then install tiny-cuda-nn torch bindings from source:

```bash
git clone --recursive https://github.com/NVlabs/tiny-cuda-nn.git
cd tiny-cuda-nn/bindings/torch
pip install -v .
```

Verify:

```bash
python -c "import tinycudann as tcnn; print('tcnn ok')"
```

If your environment name is not `moer-build`, set in `scene.json`:

```json
"python_command": "conda run -n <your-env> python"
```

Or use a direct interpreter path.

## 6. Train NRC Network

From `NRC/python`:

```bash
python train_nrc.py \
  --dataset-dir ../runtime/scenes/classroom \
  --output-dir ../runtime/checkpoints/classroom \
  --config configs/nrc_mlp.json \
  --steps 20000 \
  --batch-size 262144 \
  --lr 1e-3
```

Main outputs in `--output-dir`:

- `nrc_latest.pt`
- `nrc_step_XXXXXX.pt`
- `loss_curve.png`
- `loss_history.json`
- `loss_history.csv`
- `train_meta.json`

## 7. Run NRC Rendering

### 7.1 Path-Cutoff NRC Render

```bash
target/bin/Moer-NRC-Render_r.exe scenes/classroom
```

Reads:

- `renderer.nrc.checkpoint_file`
- `renderer.nrc.test_views_file`
- optional `renderer.nrc.train_meta_file`

Output naming:

- `<output_file>_nrc_view_<i>`

### 7.2 First-Hit NRC Render

```bash
target/bin/Moer-NRC-FirstHit_r.exe scenes/classroom
```

Output naming:

- `<output_file>_nrc_firsthit_view_<i>`

## 8. Utilities

- Inspect dataset:

  ```bash
  python NRC/utils/nrc_inspect.py NRC/runtime/scenes/classroom/_view_0.bin
  ```

- Position statistics and PLY export:

  ```bash
  python NRC/utils/nrc_viz_positions.py NRC/runtime/scenes/classroom/_view_0.bin --hist pos_hist.png --ply pos.ply
  ```

- targetRGB distribution by bounce:

  ```bash
  python NRC/utils/nrc_viz_target_rgb.py NRC/runtime/scenes/classroom/_view_0.bin -o rgb_by_bounce.png --log-x
  ```

## 9. Troubleshooting

- **`tinycudann is required`**
  - Install TCNN torch bindings in the same Python environment used by `python_command`.

- **`CUDA is required for NRC inference`**
  - Use CUDA-enabled PyTorch and a valid NVIDIA driver.

- **`renderer.nrc.* is required`**
  - Check `checkpoint_file`, `test_views_file`, and path resolution (relative paths are resolved from `scene_dir`).

- **No dataset generated**
  - Verify `views_file` and scene camera setup.
  - Check `_bounce_stats.txt` for per-bounce written/candidate counts.

- **Training OOM**
  - Reduce `--batch-size` and/or model size in `configs/nrc_mlp.json`.

