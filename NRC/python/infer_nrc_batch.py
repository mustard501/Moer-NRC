#!/usr/bin/env python3
"""Run batched NRC inference from binary feature file."""

from __future__ import annotations

import argparse
import pickle
import struct
import sys
from pathlib import Path

import numpy as np
import torch

try:
    import tinycudann as tcnn
except ImportError as exc:
    raise SystemExit(
        "tinycudann is required. Install tiny-cuda-nn/bindings/torch in this environment."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch NRC inference helper")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--features-bin", type=Path, required=True)
    parser.add_argument("--output-bin", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1 << 20)
    return parser.parse_args()


def load_feature_bin(path: Path, n_input_dims: int) -> np.ndarray:
    data = path.read_bytes()
    if len(data) < 8:
        raise SystemExit(f"feature bin too small: {path}")
    (n_records,) = struct.unpack_from("<Q", data, 0)
    expected = 8 + n_records * n_input_dims * 4
    if len(data) != expected:
        raise SystemExit(
            f"feature bin size mismatch: got={len(data)} expected={expected} records={n_records}"
        )
    # frombuffer() over bytes returns a readonly view; copy to get writable storage
    # before converting to torch tensor.
    features = np.frombuffer(data, dtype=np.float32, offset=8).reshape(n_records, n_input_dims).copy()
    return features


def infer_input_dims_from_config(config: dict) -> int | None:
    encoding = config.get("encoding")
    if not isinstance(encoding, dict):
        return None
    nested = encoding.get("nested")
    if not isinstance(nested, list):
        return None
    dims = 0
    for item in nested:
        if not isinstance(item, dict):
            return None
        n_dims = item.get("n_dims_to_encode")
        if not isinstance(n_dims, int):
            return None
        dims += n_dims
    return dims if dims > 0 else None


def save_pred_bin(path: Path, preds: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_records = np.uint64(preds.shape[0])
    with path.open("wb") as f:
        f.write(struct.pack("<Q", int(n_records)))
        f.write(preds.astype(np.float32, copy=False).tobytes(order="C"))


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for NRC inference")

    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        # Backward compatibility with old torch versions lacking weights_only.
        ckpt = torch.load(args.checkpoint, map_location="cpu")
    except pickle.UnpicklingError:
        # Some checkpoints include objects not allowlisted by weights_only loader.
        # For trusted local checkpoints, fallback to the legacy behavior.
        print(
            "[infer] weights_only load failed; fallback to weights_only=False for trusted checkpoint.",
            file=sys.stderr,
        )
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    target_transform = str(ckpt.get("target_transform", "linear_clamp"))
    n_input_dims_ckpt = int(ckpt.get("n_input_dims", 0))
    n_input_dims_cfg = infer_input_dims_from_config(config)
    if n_input_dims_cfg is not None and n_input_dims_cfg != n_input_dims_ckpt:
        print(
            f"[infer] checkpoint n_input_dims={n_input_dims_ckpt} "
            f"mismatch config-derived={n_input_dims_cfg}, using {n_input_dims_cfg}.",
            file=sys.stderr,
        )
        n_input_dims = n_input_dims_cfg
    else:
        n_input_dims = n_input_dims_ckpt if n_input_dims_ckpt > 0 else 6
    n_output_dims = int(ckpt.get("n_output_dims", 3))

    model = tcnn.NetworkWithInputEncoding(
        n_input_dims=n_input_dims,
        n_output_dims=n_output_dims,
        encoding_config=config["encoding"],
        network_config=config["network"],
    ).to("cuda")
    model.load_state_dict(ckpt["model"])
    model.eval()
    if hasattr(model, "jit_fusion"):
        model.jit_fusion = tcnn.supports_jit_fusion()

    features_np = load_feature_bin(args.features_bin, n_input_dims)
    if features_np.shape[0] == 0:
        save_pred_bin(args.output_bin, np.zeros((0, 3), dtype=np.float32))
        return

    with torch.no_grad():
        x = torch.from_numpy(features_np).to("cuda")
        preds = torch.empty((x.shape[0], 3), dtype=torch.float32, device="cuda")
        bs = max(1, args.batch_size)
        raw_min = float("inf")
        raw_max = float("-inf")
        raw_sum = 0.0
        raw_neg = 0
        raw_count = 0
        decoded_nonzero = 0
        for start in range(0, x.shape[0], bs):
            end = min(start + bs, x.shape[0])
            y_raw = model(x[start:end]).float()
            raw_min = min(raw_min, float(y_raw.min().item()))
            raw_max = max(raw_max, float(y_raw.max().item()))
            raw_sum += float(y_raw.sum().item())
            raw_neg += int((y_raw < 0.0).sum().item())
            raw_count += int(y_raw.numel())

            if target_transform == "log1p":
                y = torch.clamp_min(torch.expm1(y_raw), 0.0)
            elif target_transform == "linear_softplus":
                y = torch.nn.functional.softplus(y_raw)
            else:
                y = torch.clamp_min(y_raw, 0.0)

            decoded_nonzero += int((y > 0.0).sum().item())
            preds[start:end] = y
        preds_np = preds.cpu().numpy()
        if raw_count > 0:
            raw_mean = raw_sum / raw_count
            raw_neg_ratio = raw_neg / raw_count
            decode_nonzero_ratio = decoded_nonzero / raw_count
            print(
                "[infer] raw_pred "
                f"min={raw_min:.6g} max={raw_max:.6g} mean={raw_mean:.6g} neg_ratio={raw_neg_ratio:.4f}; "
                f"target_transform={target_transform} decoded_nonzero_ratio={decode_nonzero_ratio:.4f}"
            )

    save_pred_bin(args.output_bin, preds_np)


if __name__ == "__main__":
    main()
