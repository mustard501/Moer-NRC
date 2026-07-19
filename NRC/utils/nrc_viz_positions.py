#!/usr/bin/env python3
"""Axis-wise position histograms and PLY export from NRC v1 .bin datasets.

Loads all sample positions, plots X/Y/Z histograms, and writes a point cloud PLY.
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

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
    aabb_min: np.ndarray
    aabb_max: np.ndarray


def read_bin_header(path: Path | str) -> BinHeader:
    path = Path(path)
    file_size = path.stat().st_size
    with path.open("rb") as f:
        raw = f.read(HEADER_SIZE)
    if len(raw) < HEADER_SIZE:
        raise ValueError(f"{path}: file too small for header")

    unpacked = struct.unpack(HEADER_FMT, raw)
    magic, version, header_bytes, record_count = unpacked[0], unpacked[1], unpacked[2], unpacked[3]
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
            print(
                f"warning: {path.name}: size mismatch, using inferred recordCount={inferred:,}",
                file=sys.stderr,
            )
            record_count = inferred
        else:
            raise ValueError(
                f"{path}: size mismatch (header says {expected}, actual {file_size})"
            )

    return BinHeader(
        path=path,
        record_count=record_count,
        aabb_min=aabb_min,
        aabb_max=aabb_max,
    )


def load_positions(path: Path | str) -> tuple[np.ndarray, BinHeader]:
    """Load all position vectors as float32 array of shape (N, 3)."""
    path = Path(path)
    header = read_bin_header(path)
    n = header.record_count
    if n <= 0:
        return np.empty((0, 3), dtype=np.float32), header

    bytes_needed = n * RECORD_SIZE
    if bytes_needed <= 512 * 1024 * 1024:
        with path.open("rb") as f:
            f.seek(HEADER_SIZE)
            payload = f.read(n * RECORD_SIZE)
        records = np.frombuffer(payload, dtype=RECORD_DTYPE, count=n)
        positions = np.asarray(records["position"], dtype=np.float32).copy()
        return positions, header

    # Large file: stream only the 12-byte position field per record.
    print(f"note: {path.name}: streaming {n:,} positions from large file...", file=sys.stderr)
    positions = np.empty((n, 3), dtype=np.float32)
    with path.open("rb") as f:
        f.seek(HEADER_SIZE)
        for i in range(n):
            chunk = f.read(12)
            if len(chunk) < 12:
                positions = positions[:i]
                break
            positions[i] = np.frombuffer(chunk, dtype="<f4", count=3)
            f.seek(RECORD_SIZE - 12, os.SEEK_CUR)
            if (i + 1) % 5_000_000 == 0:
                print(f"  loaded {i + 1:,} / {n:,}", file=sys.stderr)
    return positions, header


def filter_finite(positions: np.ndarray) -> np.ndarray:
    finite = np.isfinite(positions).all(axis=1)
    dropped = int((~finite).sum())
    if dropped:
        print(f"warning: dropped {dropped:,} non-finite positions", file=sys.stderr)
    return positions[finite]


def write_ply(path: Path, positions: np.ndarray) -> None:
    """Write XYZ point cloud as binary little-endian PLY."""
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must be (N, 3), got {positions.shape}")

    n = int(positions.shape[0])
    path.parent.mkdir(parents=True, exist_ok=True)

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )

    xyz = np.ascontiguousarray(positions, dtype="<f4")
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        f.write(xyz.tobytes(order="C"))
    print(f"saved PLY: {path} ({n:,} points)")


def plot_axis_histograms(
    positions: np.ndarray,
    *,
    aabb_min: np.ndarray | None,
    aabb_max: np.ndarray | None,
    title: str,
    bins: int,
    output: Path | None,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt

    if positions.size == 0:
        raise ValueError("no positions to plot")

    xs, ys, zs = positions[:, 0], positions[:, 1], positions[:, 2]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(title, fontsize=13)

    axis_specs = [
        (axes[0], xs, "X", "tab:blue", 0),
        (axes[1], ys, "Y", "tab:orange", 1),
        (axes[2], zs, "Z", "tab:green", 2),
    ]
    for ax, data, name, color, axis_i in axis_specs:
        ax.hist(data, bins=bins, color=color, alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.set_title(f"{name}-axis (n={len(data):,})")
        ax.set_xlabel(name)
        ax.set_ylabel("count")
        ax.axvline(
            float(np.mean(data)),
            color="k",
            linestyle="--",
            linewidth=1.0,
            label=f"mean={np.mean(data):.4g}",
        )
        ax.axvline(
            float(np.median(data)),
            color="0.4",
            linestyle=":",
            linewidth=1.0,
            label=f"median={np.median(data):.4g}",
        )
        if aabb_min is not None and aabb_max is not None:
            lo = float(aabb_min[axis_i])
            hi = float(aabb_max[axis_i])
            d_lo, d_hi = float(data.min()), float(data.max())
            d_span = max(d_hi - d_lo, 1e-6)
            if abs(lo - d_lo) <= 2.0 * d_span and abs(hi - d_hi) <= 2.0 * d_span:
                ax.axvline(lo, color="crimson", linestyle="-.", linewidth=0.9, alpha=0.8, label="AABB")
                ax.axvline(hi, color="crimson", linestyle="-.", linewidth=0.9, alpha=0.8)
        ax.legend(fontsize=8, loc="best")
        stats = f"min={data.min():.4g}  max={data.max():.4g}  std={data.std():.4g}"
        ax.text(
            0.02,
            0.98,
            stats,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, edgecolor="0.8"),
        )

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"saved histogram: {output}")

    if show or output is None:
        plt.show()
    else:
        plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot X/Y/Z position histograms from NRC v1 .bin files and export all "
            "positions as a PLY point cloud."
        ),
    )
    parser.add_argument(
        "bin_files",
        nargs="+",
        type=Path,
        help="Path(s) to .bin dataset file(s); positions are concatenated",
    )
    parser.add_argument(
        "--ply",
        type=Path,
        default=None,
        help="Output PLY path (default: <first_bin_stem>_positions.ply next to first input)",
    )
    parser.add_argument(
        "--hist",
        "-o",
        type=Path,
        default=None,
        help="Save axis histograms to this image path (e.g. positions_hist.png)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=80,
        help="Histogram bin count per axis",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show interactive histogram window (also default if --hist is omitted)",
    )
    parser.add_argument(
        "--no-hist",
        action="store_true",
        help="Skip histogram plotting; only write PLY",
    )
    parser.add_argument(
        "--no-aabb",
        action="store_true",
        help="Do not mark scene AABB on histograms",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    all_positions: list[np.ndarray] = []
    aabb_mins: list[np.ndarray] = []
    aabb_maxs: list[np.ndarray] = []
    names: list[str] = []
    total_records = 0

    for path in args.bin_files:
        if not path.is_file():
            print(f"error: not a file: {path}", file=sys.stderr)
            return 1
        try:
            positions, header = load_positions(path)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        print(
            f"{path.name}: records={header.record_count:,}, "
            f"AABB=[{header.aabb_min[0]:.4g},{header.aabb_min[1]:.4g},{header.aabb_min[2]:.4g}]"
            f"→[{header.aabb_max[0]:.4g},{header.aabb_max[1]:.4g},{header.aabb_max[2]:.4g}]"
        )
        all_positions.append(positions)
        aabb_mins.append(header.aabb_min)
        aabb_maxs.append(header.aabb_max)
        names.append(path.name)
        total_records += header.record_count

    positions = (
        np.concatenate(all_positions, axis=0)
        if all_positions
        else np.empty((0, 3), dtype=np.float32)
    )
    positions = filter_finite(positions)
    if positions.size == 0:
        print("error: no finite positions found", file=sys.stderr)
        return 1

    ply_path = args.ply
    if ply_path is None:
        stem = args.bin_files[0].stem
        if len(args.bin_files) > 1:
            stem = f"{stem}_plus{len(args.bin_files) - 1}"
        ply_path = args.bin_files[0].with_name(f"{stem}_positions.ply")

    try:
        write_ply(ply_path, positions)
    except OSError as exc:
        print(f"error writing PLY: {exc}", file=sys.stderr)
        return 1

    if args.no_hist:
        return 0

    aabb_min = np.min(np.stack(aabb_mins, axis=0), axis=0) if aabb_mins else None
    aabb_max = np.max(np.stack(aabb_maxs, axis=0), axis=0) if aabb_maxs else None
    if args.no_aabb:
        aabb_min = aabb_max = None

    if len(names) == 1:
        title = f"Position axis distribution — {names[0]} (n={len(positions):,})"
    else:
        title = (
            f"Position axis distribution — {len(names)} files "
            f"(n={len(positions):,} / records={total_records:,})"
        )

    try:
        plot_axis_histograms(
            positions,
            aabb_min=aabb_min,
            aabb_max=aabb_max,
            title=title,
            bins=max(8, args.bins),
            output=args.hist,
            show=args.show or args.hist is None,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except ImportError:
        print(
            "error: matplotlib is required for histograms. "
            "Install with: pip install matplotlib  (or pass --no-hist)",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
