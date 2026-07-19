#!/usr/bin/env python3
"""Inspect NRC v1 radiance dataset (.bin) files."""

from __future__ import annotations

import argparse
import math
import os
import struct
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

MAGIC = b"NRC1"
VERSION = 1
HEADER_FMT = "<4sII4xQIIII3f3f8I"
RECORD_FMT = "<3f3f3f3fHH"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
RECORD_SIZE = struct.calcsize(RECORD_FMT)

FLAG_HAS_NORMAL = 1 << 0


@dataclass(frozen=True)
class DatasetHeader:
    magic: bytes
    version: int
    header_bytes: int
    record_count: int
    feature_dim: int
    label_dim: int
    flags: int
    reserved0: int
    scene_aabb_min: tuple[float, float, float]
    scene_aabb_max: tuple[float, float, float]
    reserved: tuple[int, ...]


@dataclass
class DatasetStats:
    bounce_counts: Counter[int]
    black_skipped: int = 0
    nan_skipped: int = 0
    rgb_sum: list[float] | None = None
    rgb_min: list[float] | None = None
    rgb_max: list[float] | None = None


def parse_header(data: bytes) -> DatasetHeader:
    if len(data) != HEADER_SIZE:
        raise ValueError(f"header size mismatch: expected {HEADER_SIZE}, got {len(data)}")

    unpacked = struct.unpack(HEADER_FMT, data)
    return DatasetHeader(
        magic=unpacked[0],
        version=unpacked[1],
        header_bytes=unpacked[2],
        record_count=unpacked[3],
        feature_dim=unpacked[4],
        label_dim=unpacked[5],
        flags=unpacked[6],
        reserved0=unpacked[7],
        scene_aabb_min=(unpacked[8], unpacked[9], unpacked[10]),
        scene_aabb_max=(unpacked[11], unpacked[12], unpacked[13]),
        reserved=unpacked[14:],
    )


def validate_header(header: DatasetHeader, file_size: int) -> list[str]:
    warnings: list[str] = []
    if header.magic != MAGIC:
        raise ValueError(f"invalid magic: {header.magic!r}, expected {MAGIC!r}")
    if header.version != VERSION:
        raise ValueError(f"unsupported version: {header.version}")
    if header.header_bytes != HEADER_SIZE:
        warnings.append(f"headerBytes={header.header_bytes}, expected {HEADER_SIZE}")
    if header.feature_dim != 9:
        warnings.append(f"unexpected featureDim={header.feature_dim}, expected 9")
    if header.label_dim != 3:
        warnings.append(f"unexpected labelDim={header.label_dim}, expected 3")

    expected_size = header.header_bytes + header.record_count * RECORD_SIZE
    if expected_size != file_size:
        warnings.append(
            f"file size mismatch: header says {expected_size} bytes, actual {file_size} bytes"
        )
        inferred = (file_size - header.header_bytes) // RECORD_SIZE
        if (file_size - header.header_bytes) % RECORD_SIZE == 0:
            warnings.append(f"inferred record count from file size: {inferred}")
    return warnings


def unpack_record(chunk: bytes) -> tuple:
    values = struct.unpack(RECORD_FMT, chunk)
    return (
        (values[0], values[1], values[2]),
        (values[3], values[4], values[5]),
        (values[6], values[7], values[8]),
        (values[9], values[10], values[11]),
        values[12],
        values[13],
    )


def iter_records(path: str, record_count: int) -> Iterable[tuple]:
    with open(path, "rb") as f:
        f.seek(HEADER_SIZE)
        for _ in range(record_count):
            chunk = f.read(RECORD_SIZE)
            if len(chunk) < RECORD_SIZE:
                break
            yield unpack_record(chunk)


def scan_records(
    path: str,
    record_count: int,
    *,
    collect_rgb: bool = False,
    progress_every: int = 0,
) -> DatasetStats:
    stats = DatasetStats(bounce_counts=Counter())
    if collect_rgb:
        stats.rgb_sum = [0.0, 0.0, 0.0]
        stats.rgb_min = [math.inf, math.inf, math.inf]
        stats.rgb_max = [-math.inf, -math.inf, -math.inf]

    processed = 0
    for position, outgoing_dir, normal, target_rgb, bounce, _padding in iter_records(path, record_count):
        if any(math.isnan(v) for v in (*position, *outgoing_dir, *normal, *target_rgb)):
            stats.nan_skipped += 1
            continue
        if target_rgb == (0.0, 0.0, 0.0):
            stats.black_skipped += 1

        stats.bounce_counts[bounce] += 1

        if collect_rgb and stats.rgb_sum is not None and stats.rgb_min is not None and stats.rgb_max is not None:
            for i, value in enumerate(target_rgb):
                stats.rgb_sum[i] += value
                stats.rgb_min[i] = min(stats.rgb_min[i], value)
                stats.rgb_max[i] = max(stats.rgb_max[i], value)

        processed += 1
        if progress_every > 0 and processed % progress_every == 0:
            print(f"  scanned {processed:,} / {record_count:,} records...", file=sys.stderr)

    return stats


def format_flags(flags: int) -> str:
    names: list[str] = []
    if flags & FLAG_HAS_NORMAL:
        names.append("HasNormal")
    if not names:
        return "None"
    return "|".join(names)


def print_header_report(path: str, header: DatasetHeader, file_size: int, warnings: Sequence[str]) -> None:
    print(f"File: {path}")
    print(f"  size: {file_size:,} bytes ({file_size / (1024 ** 3):.3f} GiB)")
    print(f"  magic: {header.magic.decode('ascii', errors='replace')}")
    print(f"  version: {header.version}")
    print(f"  headerBytes: {header.header_bytes}")
    print(f"  recordCount (header): {header.record_count:,}")
    print(f"  featureDim: {header.feature_dim}")
    print(f"  labelDim: {header.label_dim}")
    print(f"  flags: {header.flags} ({format_flags(header.flags)})")
    print(
        "  sceneAabbMin: "
        f"({header.scene_aabb_min[0]:.6g}, {header.scene_aabb_min[1]:.6g}, {header.scene_aabb_min[2]:.6g})"
    )
    print(
        "  sceneAabbMax: "
        f"({header.scene_aabb_max[0]:.6g}, {header.scene_aabb_max[1]:.6g}, {header.scene_aabb_max[2]:.6g})"
    )
    for warning in warnings:
        print(f"  warning: {warning}")


def print_stats_report(stats: DatasetStats, scanned_count: int) -> None:
    print(f"  scanned records: {scanned_count:,}")
    if stats.black_skipped:
        print(f"  all-black targetRGB: {stats.black_skipped:,}")
    if stats.nan_skipped:
        print(f"  records with NaN fields: {stats.nan_skipped:,}")

    print("  bounce distribution:")
    if not stats.bounce_counts:
        print("    (none)")
        return

    max_bounce = max(stats.bounce_counts)
    for bounce in range(1, max_bounce + 1):
        count = stats.bounce_counts.get(bounce, 0)
        if count == 0:
            continue
        ratio = 100.0 * count / scanned_count if scanned_count else 0.0
        print(f"    bounce {bounce:>2}: {count:>12,}  ({ratio:5.2f}%)")

    other = sum(c for b, c in stats.bounce_counts.items() if b <= 0 or b > max_bounce)
    if other:
        print(f"    other : {other:>12,}")

    if stats.rgb_sum and scanned_count > 0:
        mean = [v / scanned_count for v in stats.rgb_sum]
        print(
            "  targetRGB mean: "
            f"R={mean[0]:.6g}, G={mean[1]:.6g}, B={mean[2]:.6g}"
        )
        if stats.rgb_min and stats.rgb_max:
            print(
                "  targetRGB min : "
                f"R={stats.rgb_min[0]:.6g}, G={stats.rgb_min[1]:.6g}, B={stats.rgb_min[2]:.6g}"
            )
            print(
                "  targetRGB max : "
                f"R={stats.rgb_max[0]:.6g}, G={stats.rgb_max[1]:.6g}, B={stats.rgb_max[2]:.6g}"
            )


def print_sample_records(path: str, record_count: int, sample_count: int) -> None:
    print(f"  first {sample_count} records:")
    for index, record in enumerate(iter_records(path, record_count)):
        if index >= sample_count:
            break
        position, outgoing_dir, normal, target_rgb, bounce, _padding = record
        print(
            f"    [{index}] bounce={bounce} "
            f"pos=({position[0]:.4g},{position[1]:.4g},{position[2]:.4g}) "
            f"wo=({outgoing_dir[0]:.4g},{outgoing_dir[1]:.4g},{outgoing_dir[2]:.4g}) "
            f"rgb=({target_rgb[0]:.4g},{target_rgb[1]:.4g},{target_rgb[2]:.4g})"
        )


def inspect_file(
    path: str,
    *,
    header_only: bool,
    collect_rgb: bool,
    show_samples: int,
    progress_every: int,
) -> int:
    if not os.path.isfile(path):
        print(f"error: not a file: {path}", file=sys.stderr)
        return 1

    file_size = os.path.getsize(path)
    with open(path, "rb") as f:
        header_data = f.read(HEADER_SIZE)
    if len(header_data) < HEADER_SIZE:
        print(f"error: file too small to contain header: {path}", file=sys.stderr)
        return 1

    try:
        header = parse_header(header_data)
        warnings = validate_header(header, file_size)
    except ValueError as exc:
        print(f"error: {path}: {exc}", file=sys.stderr)
        return 1

    print_header_report(path, header, file_size, warnings)

    if header_only:
        print()
        return 0

    record_count = header.record_count
    if warnings:
        inferred = (file_size - header.header_bytes) // RECORD_SIZE
        if (file_size - header.header_bytes) % RECORD_SIZE == 0 and inferred != record_count:
            print(f"  note: scanning {inferred:,} records inferred from file size")
            record_count = inferred

    stats = scan_records(
        path,
        record_count,
        collect_rgb=collect_rgb,
        progress_every=progress_every,
    )
    scanned_count = sum(stats.bounce_counts.values())
    print_stats_report(stats, scanned_count)

    if show_samples > 0:
        print_sample_records(path, min(record_count, show_samples), show_samples)

    print()
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect NRC v1 radiance dataset (.bin) files.",
    )
    parser.add_argument(
        "bin_files",
        nargs="+",
        help="Path(s) to .bin dataset file(s)",
    )
    parser.add_argument(
        "--header-only",
        action="store_true",
        help="Only print header metadata, skip record scan",
    )
    parser.add_argument(
        "--rgb-stats",
        action="store_true",
        help="Compute min/max/mean statistics for targetRGB (extra pass over all records)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=0,
        metavar="N",
        help="Print first N sample records",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5_000_000,
        metavar="N",
        help="Print scan progress every N records to stderr (0 to disable)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    exit_code = 0
    for path in args.bin_files:
        code = inspect_file(
            path,
            header_only=args.header_only,
            collect_rgb=args.rgb_stats,
            show_samples=max(0, args.samples),
            progress_every=max(0, args.progress_every),
        )
        exit_code = max(exit_code, code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
