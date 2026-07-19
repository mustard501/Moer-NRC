# NRC Python training

## Layout

```
NRC/python/
  dataset.py              # read all *.bin under a directory
  train_nrc.py            # tinycudann training loop
  configs/nrc_mlp.json    # encoding + FullyFusedMLP
```

## Feature preprocessing

| dims | content | transform |
|------|---------|-----------|
| 0:3 | position | AABB → `[0, 1]` |
| 3:6 | outgoingDir | unit vector, then `(v+1)/2` → `[0, 1]` for OneBlob |
| 6:9 | normal | unit vector (Identity) |

Labels: linear HDR `targetRGB`.

## Train

```bat
cd NRC\python
python train_nrc.py --dataset-dir ..\runtime\scenes\classroom --output-dir ..\runtime\checkpoints\classroom
```

Useful flags:

```bat
python train_nrc.py --dataset-dir <dir> --steps 20000 --batch-size 262144 --lr 0.001 --config configs\nrc_mlp.json --output-dir <dir>
```

Checkpoints are written to `--output-dir` as `nrc_step_XXXXXX.pt` and `nrc_latest.pt`, including AABB + network config for later inference.

Training also writes under the same directory:

- `loss_curve.png` — train/val loss curves (refreshed periodically)
- `loss_history.json` / `loss_history.csv` — raw logged points

Requires `matplotlib` for the PNG (`pip install matplotlib`).

Progress bar uses `tqdm` (`pip install tqdm`). Disable with `--no-progress`.
