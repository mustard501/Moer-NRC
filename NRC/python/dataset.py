#!/usr/bin/env python3
"""NRC v1 dataset reader: load all .bin shards under a directory."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

MAGIC = b"NRC1"
VERSION = 1
HEADER_FMT = "<4sII4xQIIII3f3f8I"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
RECORD_SIZE = 52

RECORD_DTYPE = np.dtype(
    [
        ("position", "<f4", (3,)),
        ("outgoing_dir", "<f4", (3,)),
        ("normal", "<f4", (3,)),
        ("target_rgb", "<f4", (3,)),
        ("bounce", "<u2"),
        ("padding", "<u2"),
    ]
)


@dataclass(frozen=True)
class BinHeader:
    path: Path
    record_count: int
    feature_dim: int
    label_dim: int
    flags: int
    aabb_min: np.ndarray  # (3,)
    aabb_max: np.ndarray  # (3,)


def read_bin_header(path: Path | str) -> BinHeader:
    path = Path(path)
    file_size = path.stat().st_size
    with path.open("rb") as f:
        raw = f.read(HEADER_SIZE)
    if len(raw) < HEADER_SIZE:
        raise ValueError(f"{path}: file too small for header")

    unpacked = struct.unpack(HEADER_FMT, raw)
    magic = unpacked[0]
    version = unpacked[1]
    header_bytes = unpacked[2]
    record_count = unpacked[3]
    feature_dim = unpacked[4]
    label_dim = unpacked[5]
    flags = unpacked[6]
    aabb_min = np.array(unpacked[8:11], dtype=np.float64)
    aabb_max = np.array(unpacked[11:14], dtype=np.float64)

    if magic != MAGIC:
        raise ValueError(f"{path}: invalid magic {magic!r}")
    if version != VERSION:
        raise ValueError(f"{path}: unsupported version {version}")
    if header_bytes != HEADER_SIZE:
        raise ValueError(f"{path}: unexpected headerBytes={header_bytes}")

    expected = header_bytes + record_count * RECORD_SIZE
    if expected != file_size:
        inferred = (file_size - header_bytes) // RECORD_SIZE
        if (file_size - header_bytes) % RECORD_SIZE == 0 and inferred > 0:
            record_count = inferred
        else:
            raise ValueError(
                f"{path}: size mismatch (header says {expected}, actual {file_size})"
            )

    return BinHeader(
        path=path,
        record_count=record_count,
        feature_dim=feature_dim,
        label_dim=label_dim,
        flags=flags,
        aabb_min=aabb_min,
        aabb_max=aabb_max,
    )


def load_records(path: Path | str, record_count: int | None = None) -> np.ndarray:
    path = Path(path)
    if record_count is None:
        record_count = read_bin_header(path).record_count
    with path.open("rb") as f:
        f.seek(HEADER_SIZE)
        payload = f.read(record_count * RECORD_SIZE)
    expected = record_count * RECORD_SIZE
    if len(payload) != expected:
        raise ValueError(f"{path}: expected {expected} payload bytes, got {len(payload)}")
    return np.frombuffer(payload, dtype=RECORD_DTYPE, count=record_count).copy()


def discover_bin_files(dataset_dir: Path | str) -> list[Path]:
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"dataset directory not found: {dataset_dir}")
    files = sorted(p for p in dataset_dir.glob("*.bin") if p.is_file())
    if not files:
        raise FileNotFoundError(f"no .bin files under {dataset_dir}")
    return files


def union_aabb(headers: Sequence[BinHeader]) -> tuple[np.ndarray, np.ndarray]:
    aabb_min = np.min(np.stack([h.aabb_min for h in headers], axis=0), axis=0)
    aabb_max = np.max(np.stack([h.aabb_max for h in headers], axis=0), axis=0)
    return aabb_min, aabb_max


def normalize_positions(positions: np.ndarray, aabb_min: np.ndarray, aabb_max: np.ndarray) -> np.ndarray:
    """Map world positions to [0, 1] using scene AABB."""
    extent = np.maximum(aabb_max - aabb_min, 1e-8)
    return ((positions - aabb_min) / extent).astype(np.float32)


def normalize_vectors(vectors: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """L2-normalize rows to unit length."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return (vectors / np.maximum(norms, eps)).astype(np.float32)


def build_features(
    records: np.ndarray,
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Feature layout (6-D), matched to configs/nrc_mlp.json:
      [0:3] position in [0, 1]           -> HashGrid
      [3:6] outgoing dir mapped to [0,1] -> SphericalHarmonics (unit wo, then (v+1)/2)

    Returns features, targets, bounces.
    """
    positions = normalize_positions(records["position"], aabb_min, aabb_max)
    outgoing = normalize_vectors(records["outgoing_dir"])
    outgoing_01 = (0.5 * (outgoing + 1.0)).astype(np.float32)
    targets = records["target_rgb"].astype(np.float32)
    features = np.concatenate([positions, outgoing_01], axis=1).astype(np.float32)
    bounces = records["bounce"].astype(np.int32)
    return features, targets, bounces


class NRCRadianceDataset(Dataset):
    def __init__(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        bounces: torch.Tensor,
        *,
        aabb_min: np.ndarray,
        aabb_max: np.ndarray,
        bin_files: list[Path],
        split: str,
    ) -> None:
        self.features = features
        self.targets = targets
        self.bounces = bounces
        self.aabb_min = aabb_min
        self.aabb_max = aabb_max
        self.bin_files = bin_files
        self.split = split

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]

    @property
    def n_input_dims(self) -> int:
        return 6

    @property
    def n_output_dims(self) -> int:
        return 3

    @staticmethod
    def from_directory(
        dataset_dir: Path | str,
        *,
        val_ratio: float = 0.05,
        seed: int = 0,
    ) -> tuple["NRCRadianceDataset", "NRCRadianceDataset"]:
        if not 0.0 <= val_ratio < 1.0:
            raise ValueError("val_ratio must be in [0, 1)")

        dataset_dir = Path(dataset_dir)
        bin_files = discover_bin_files(dataset_dir)
        headers = [read_bin_header(p) for p in bin_files]
        aabb_min, aabb_max = union_aabb(headers)

        chunks: list[np.ndarray] = []
        for header in headers:
            print(
                f"[dataset] loading {header.path.name}: {header.record_count:,} records"
            )
            chunks.append(load_records(header.path, header.record_count))
        records = np.concatenate(chunks, axis=0)
        features_np, targets_np, bounces_np = build_features(records, aabb_min, aabb_max)

        n = features_np.shape[0]
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        n_val = int(n * val_ratio)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        features = torch.from_numpy(features_np)
        targets = torch.from_numpy(targets_np)
        bounces = torch.from_numpy(bounces_np)

        train_set = NRCRadianceDataset(
            features[train_idx],
            targets[train_idx],
            bounces[train_idx],
            aabb_min=aabb_min,
            aabb_max=aabb_max,
            bin_files=bin_files,
            split="train",
        )
        val_set = NRCRadianceDataset(
            features[val_idx],
            targets[val_idx],
            bounces[val_idx],
            aabb_min=aabb_min,
            aabb_max=aabb_max,
            bin_files=bin_files,
            split="val",
        )

        print(
            f"[dataset] train={len(train_set):,} val={len(val_set):,} "
            f"files={len(bin_files)} "
            f"aabb_min={aabb_min.tolist()} aabb_max={aabb_max.tolist()}"
        )
        return train_set, val_set
