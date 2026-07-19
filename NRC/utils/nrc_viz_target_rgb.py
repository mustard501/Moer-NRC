#!/usr/bin/env python3
"""Plot targetRGB channel distributions from NRC v1 .bin datasets.

Loads all sample targetRGB values and plots R/G/B histograms with basic stats.
Values are linear HDR radiance (not tonemapped).
"""

from __future__ import annotations

import argparse
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
# position(12) + outgoingDir(12) + normal(12) = 36
TARGET_RGB_OFFSET = 36

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


def read_bin_header(path: Path | str) -> BinHeader:
    path = Path(path)
    file_size = path.stat().st_size
    with path.open("rb") as f:
        raw = f.read(HEADER_SIZE)
    if len(raw) < HEADER_SIZE:
        raise ValueError(f"{path}: file too small for header")

    unpacked = struct.unpack(HEADER_FMT, raw)
    magic, version, header_bytes, record_count = unpacked[0], unpacked[1], unpacked[2], unpacked[3]

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

    return BinHeader(path=path, record_count=record_count)


def load_target_rgb(path: Path | str) -> tuple[np.ndarray, BinHeader]:
    """Load all targetRGB vectors as float32 array of shape (N, 3)."""
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
        rgb = np.asarray(records["target_rgb"], dtype=np.float32).copy()
        return rgb, header

    print(f"note: {path.name}: streaming {n:,} targetRGB from large file...", file=sys.stderr)
    rgb = np.empty((n, 3), dtype=np.float32)
    with path.open("rb") as f:
        for i in range(n):
            f.seek(HEADER_SIZE + i * RECORD_SIZE + TARGET_RGB_OFFSET)
            chunk = f.read(12)
            if len(chunk) < 12:
                rgb = rgb[:i]
                break
            rgb[i] = np.frombuffer(chunk, dtype="<f4", count=3)
            if (i + 1) % 5_000_000 == 0:
                print(f"  loaded {i + 1:,} / {n:,}", file=sys.stderr)
    return rgb, header


def filter_finite(rgb: np.ndarray) -> np.ndarray:
    finite = np.isfinite(rgb).all(axis=1)
    dropped = int((~finite).sum())
    if dropped:
        print(f"warning: dropped {dropped:,} non-finite targetRGB rows", file=sys.stderr)
    return rgb[finite]


def print_channel_stats(rgb: np.ndarray) -> None:
    names = ("R", "G", "B")
    print(f"targetRGB stats over {len(rgb):,} samples:")
    for i, name in enumerate(names):
        ch = rgb[:, i]
        pcts = np.percentile(ch, [50, 90, 99, 99.9])
        print(
            f"  {name}: min={ch.min():.6g}  max={ch.max():.6g}  "
            f"mean={ch.mean():.6g}  std={ch.std():.6g}  "
            f"p50={pcts[0]:.6g}  p90={pcts[1]:.6g}  p99={pcts[2]:.6g}  p99.9={pcts[3]:.6g}"
        )
    black = int(np.all(rgb == 0.0, axis=1).sum())
    if black:
        print(f"  all-black samples: {black:,} ({100.0 * black / len(rgb):.2f}%)")


def plot_rgb_histograms(
    rgb: np.ndarray,
    *,
    title: str,
    bins: int,
    log_x: bool,
    clip_percentile: float | None,
    output: Path | None,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt

    if rgb.size == 0:
        raise ValueError("no targetRGB values to plot")

    plot_rgb = rgb
    if clip_percentile is not None:
        hi = float(np.percentile(rgb, clip_percentile))
        if hi > 0:
            plot_rgb = rgb[np.all(rgb <= hi, axis=1)]
            print(
                f"note: histogram clipped to <={clip_percentile:g}th percentile "
                f"(value={hi:.6g}); kept {len(plot_rgb):,} / {len(rgb):,} samples",
                file=sys.stderr,
            )

    channels = [
        (plot_rgb[:, 0], "R", "tab:red"),
        (plot_rgb[:, 1], "G", "tab:green"),
        (plot_rgb[:, 2], "B", "tab:blue"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(title, fontsize=13)

    for ax, (data, name, color) in zip(axes, channels):
        if log_x:
            positive = data[data > 0]
            if positive.size == 0:
                ax.text(0.5, 0.5, "no positive values", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{name} (log-x)")
                continue
            lo = max(float(positive.min()), 1e-8)
            hi = float(positive.max())
            bin_edges = np.logspace(np.log10(lo), np.log10(hi), bins + 1)
            ax.hist(positive, bins=bin_edges, color=color, alpha=0.85, edgecolor="white", linewidth=0.3)
            ax.set_xscale("log")
            ax.set_title(f"{name}-channel (n>0={len(positive):,})")
        else:
            ax.hist(data, bins=bins, color=color, alpha=0.85, edgecolor="white", linewidth=0.3)
            ax.set_title(f"{name}-channel (n={len(data):,})")

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
            color="0.35",
            linestyle=":",
            linewidth=1.0,
            label=f"median={np.median(data):.4g}",
        )
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
        description="Plot targetRGB R/G/B distributions from NRC v1 .bin dataset files.",
    )
    parser.add_argument(
        "bin_files",
        nargs="+",
        type=Path,
        help="Path(s) to .bin dataset file(s); targetRGB rows are concatenated",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Save histogram image (e.g. target_rgb_hist.png). If omitted, opens a window",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=80,
        help="Histogram bin count per channel",
    )
    parser.add_argument(
        "--log-x",
        action="store_true",
        help="Use log-scale X axis (positive values only); useful for HDR radiance",
    )
    parser.add_argument(
        "--clip-percentile",
        type=float,
        default=None,
        metavar="P",
        help="Only histogram samples with all channels <= P-th percentile (e.g. 99.5)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also show interactive window when --output is set",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.clip_percentile is not None and not (0.0 < args.clip_percentile <= 100.0):
        print("error: --clip-percentile must be in (0, 100]", file=sys.stderr)
        return 1

    all_rgb: list[np.ndarray] = []
    names: list[str] = []
    total_records = 0

    for path in args.bin_files:
        if not path.is_file():
            print(f"error: not a file: {path}", file=sys.stderr)
            return 1
        try:
            rgb, header = load_target_rgb(path)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        print(f"{path.name}: records={header.record_count:,}")
        all_rgb.append(rgb)
        names.append(path.name)
        total_records += header.record_count

    rgb = np.concatenate(all_rgb, axis=0) if all_rgb else np.empty((0, 3), dtype=np.float32)
    rgb = filter_finite(rgb)
    if rgb.size == 0:
        print("error: no finite targetRGB values found", file=sys.stderr)
        return 1

    print_channel_stats(rgb)

    if len(names) == 1:
        title = f"targetRGB distribution — {names[0]} (n={len(rgb):,})"
    else:
        title = (
            f"targetRGB distribution — {len(names)} files "
            f"(n={len(rgb):,} / records={total_records:,})"
        )

    try:
        plot_rgb_histograms(
            rgb,
            title=title,
            bins=max(8, args.bins),
            log_x=args.log_x,
            clip_percentile=args.clip_percentile,
            output=args.output,
            show=args.show or args.output is None,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except ImportError:
        print(
            "error: matplotlib is required. Install with: pip install matplotlib",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
