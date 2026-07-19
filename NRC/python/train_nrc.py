#!/usr/bin/env python3
"""Train an NRC radiance MLP with tiny-cuda-nn PyTorch bindings."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from dataset import NRCRadianceDataset

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[misc, assignment]

try:
    import tinycudann as tcnn
except ImportError as exc:
    raise SystemExit(
        "tinycudann is required. Install tiny-cuda-nn/bindings/torch in this environment."
    ) from exc


def load_json(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    try:
        import commentjson

        return commentjson.loads(text)
    except ImportError:
        return json.loads(text)


def logspace_mse_loss(pred_log: torch.Tensor, target_log: torch.Tensor) -> torch.Tensor:
    # Targets are pre-transformed to log1p space; optimize directly there.
    pred_f = pred_log.float()
    target_f = target_log.float()
    return torch.mean((pred_f - target_f) ** 2)


def to_log1p_targets(targets: torch.Tensor) -> torch.Tensor:
    return torch.log1p(torch.clamp_min(targets.float(), 0.0))


def feature_tv_loss(
    model: torch.nn.Module,
    features: torch.Tensor,
    *,
    noise_std: float = 1e-3,
    pos_only: bool = True,
) -> torch.Tensor:
    """
    Local smoothness regularizer in feature space:
      TV = mean(|f(x + δ) - f(x)|), with small random perturbation δ.
    """
    if noise_std <= 0.0:
        return torch.zeros((), device=features.device, dtype=torch.float32)

    delta = torch.randn_like(features) * noise_std
    if pos_only and features.shape[1] > 3:
        delta[:, 3:] = 0.0

    feat_perturbed = torch.clamp(features + delta, 0.0, 1.0)
    pred = model(features).float()
    pred_perturbed = model(feat_perturbed).float()
    return torch.mean(torch.abs(pred_perturbed - pred))


def save_loss_history(
    output_dir: Path,
    train_steps: list[int],
    train_losses: list[float],
    val_steps: list[int],
    val_losses: list[float],
) -> None:
    history = {
        "train": [{"step": s, "loss": l} for s, l in zip(train_steps, train_losses)],
        "val": [{"step": s, "loss": l} for s, l in zip(val_steps, val_losses)],
    }
    (output_dir / "loss_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )

    with (output_dir / "loss_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "step", "loss"])
        for step, loss in zip(train_steps, train_losses):
            writer.writerow(["train", step, loss])
        for step, loss in zip(val_steps, val_losses):
            writer.writerow(["val", step, loss])


def plot_loss_curves(
    output_dir: Path,
    train_steps: list[int],
    train_losses: list[float],
    val_steps: list[int],
    val_losses: list[float],
) -> Path | None:
    if not train_steps and not val_steps:
        return None

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "[train] matplotlib not installed; skip loss curve plot "
            "(pip install matplotlib). JSON/CSV history is still saved."
        )
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)

    if train_steps:
        ax.plot(train_steps, train_losses, label="train", color="#1f77b4", linewidth=1.5)
    if val_steps:
        ax.plot(
            val_steps,
            val_losses,
            label="val",
            color="#d62728",
            linewidth=1.5,
            marker="o",
            markersize=3,
        )

    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("NRC training loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    positive = [v for v in train_losses + val_losses if v > 0]
    if positive and (max(positive) / max(min(positive), 1e-12) > 50):
        ax.set_yscale("log")

    fig.tight_layout()
    png_path = output_dir / "loss_curve.png"
    fig.savefig(png_path)
    plt.close(fig)
    return png_path


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
    n_input_dims: int,
    n_output_dims: int,
    config: dict,
    train_loss: float,
    val_loss: float | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "aabb_min": aabb_min.astype(np.float32),
        "aabb_max": aabb_max.astype(np.float32),
        "n_input_dims": int(n_input_dims),
        "n_output_dims": int(n_output_dims),
        "config": config,
        "target_transform": "log1p",
        "args": vars(args),
        "train_loss": train_loss,
        "val_loss": val_loss,
    }
    torch.save(payload, path)
    print(f"\n[train] saved checkpoint -> {path}")


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    n = features.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        pred = model(features[start:end])
        loss = logspace_mse_loss(pred, targets[start:end])
        total += float(loss.item()) * (end - start)
        count += end - start
    model.train()
    return total / max(count, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NRC radiance cache with tinycudann.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Directory containing one or more NRC v1 .bin files",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs" / "nrc_mlp.json",
        help="JSON config with encoding/network sections",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("NRC/runtime/checkpoints"))
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=2**18)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--tv-weight",
        type=float,
        default=0.0,
        help="Weight of feature-space TV regularizer (0 disables)",
    )
    parser.add_argument(
        "--tv-noise-std",
        type=float,
        default=1e-3,
        help="Noise std for TV regularizer in normalized feature space",
    )
    parser.add_argument(
        "--tv-pos-only",
        action="store_true",
        help="Apply TV perturbation only on position dims [0:3]",
    )
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--val-every", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=8,
        help="Stop after N val checks without improvement (<=0 disables)",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help="Minimum val loss improvement to reset early-stop counter",
    )
    parser.add_argument(
        "--lr-scheduler-patience",
        type=int,
        default=2,
        help="ReduceLROnPlateau patience in number of val checks",
    )
    parser.add_argument(
        "--lr-scheduler-factor",
        type=float,
        default=0.5,
        help="LR multiply factor when val plateaus (0<f<1)",
    )
    parser.add_argument(
        "--lr-scheduler-min-lr",
        type=float,
        default=1e-5,
        help="Minimum LR for ReduceLROnPlateau",
    )
    parser.add_argument(
        "--plot-every",
        type=int,
        default=1000,
        help="Refresh loss_curve.png every N steps (also on save/end)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar",
    )
    parser.add_argument(
        "--keep-data-on-cpu",
        action="store_true",
        help="Do not cache the full dataset on GPU (slower host->device copies)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for tinycudann training")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")

    config = load_json(args.config)
    if "encoding" not in config or "network" not in config:
        raise SystemExit("config must contain 'encoding' and 'network'")

    train_set, val_set = NRCRadianceDataset.from_directory(
        args.dataset_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    if len(train_set) == 0:
        raise SystemExit("train split is empty")

    # Avoid PyTorch DataLoader: with batch_size ~2^18, per-sample __getitem__
    # dominates runtime. Index the full tensors directly instead.
    data_device = torch.device("cpu") if args.keep_data_on_cpu else device
    train_x = train_set.features.to(data_device)
    train_y = to_log1p_targets(train_set.targets).to(data_device)
    val_x = val_set.features.to(data_device) if len(val_set) > 0 else None
    val_y = to_log1p_targets(val_set.targets).to(data_device) if len(val_set) > 0 else None
    n_train = train_x.shape[0]
    batch_size = min(args.batch_size, n_train)

    bytes_est = (train_x.numel() + train_y.numel()) * 4
    if val_x is not None and val_y is not None:
        bytes_est += (val_x.numel() + val_y.numel()) * 4
    print(
        f"[train] device={device} data_device={data_device} "
        f"dataset_cache≈{bytes_est / (1024**2):.1f} MiB"
    )
    print("[train] target transform: log1p (optimize in log space, decode with expm1 at inference)")

    model = tcnn.NetworkWithInputEncoding(
        n_input_dims=train_set.n_input_dims,
        n_output_dims=train_set.n_output_dims,
        encoding_config=config["encoding"],
        network_config=config["network"],
    ).to(device)
    if hasattr(model, "jit_fusion"):
        model.jit_fusion = tcnn.supports_jit_fusion()
        print(f"[train] jit_fusion={model.jit_fusion}")
    print(model)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_scheduler_factor,
        patience=args.lr_scheduler_patience,
        min_lr=args.lr_scheduler_min_lr,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = args.output_dir / "train_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "dataset_dir": str(args.dataset_dir.resolve()),
                "config": str(args.config.resolve()),
                "aabb_min": train_set.aabb_min.tolist(),
                "aabb_max": train_set.aabb_max.tolist(),
                "train_samples": len(train_set),
                "val_samples": len(val_set),
                "bin_files": [p.name for p in train_set.bin_files],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    model.train()
    running = 0.0
    running_data = 0.0
    running_tv = 0.0
    last_train = None
    last_val = None
    train_steps: list[int] = []
    train_losses: list[float] = []
    val_steps: list[int] = []
    val_losses: list[float] = []
    best_val = float("inf")
    best_step = 0
    bad_val_count = 0
    stop_requested = False

    use_tqdm = (not args.no_progress) and (tqdm is not None)
    if not args.no_progress and tqdm is None:
        print("[train] tqdm not installed; install with: pip install tqdm")

    print(
        f"[train] steps={args.steps} batch_size={batch_size} "
        f"lr={args.lr} wd={args.weight_decay} train={len(train_set):,} val={len(val_set):,}"
    )

    step_iter = range(1, args.steps + 1)
    if use_tqdm:
        pbar = tqdm(step_iter, total=args.steps, desc="train", dynamic_ncols=True)
    else:
        pbar = step_iter
        t0 = time.perf_counter()

    for step in pbar:
        idx = torch.randint(0, n_train, (batch_size,), device=data_device)
        features = train_x[idx]
        targets = train_y[idx]
        if data_device.type != "cuda":
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

        pred = model(features)
        data_loss = logspace_mse_loss(pred, targets)
        tv_loss = torch.zeros((), device=device, dtype=torch.float32)
        if args.tv_weight > 0.0:
            tv_loss = feature_tv_loss(
                model,
                features,
                noise_std=args.tv_noise_std,
                pos_only=args.tv_pos_only,
            )
        loss = data_loss + args.tv_weight * tv_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_loss = float(loss.item())
        batch_data_loss = float(data_loss.item())
        batch_tv_loss = float(tv_loss.item())
        running += batch_loss
        running_data += batch_data_loss
        running_tv += batch_tv_loss

        if step % args.log_every == 0:
            avg = running / args.log_every
            avg_data = running_data / args.log_every
            avg_tv = running_tv / args.log_every
            running = 0.0
            running_data = 0.0
            running_tv = 0.0
            last_train = avg
            train_steps.append(step)
            train_losses.append(avg)
            if not use_tqdm:
                elapsed = time.perf_counter() - t0
                print(
                    f"[train] step={step}/{args.steps} loss={avg:.6g} "
                    f"data={avg_data:.6g} tv={avg_tv:.6g} time={elapsed:.1f}s"
                )
                t0 = time.perf_counter()

        if val_x is not None and val_y is not None and step % args.val_every == 0:
            last_val = evaluate(model, val_x, val_y, batch_size)
            val_steps.append(step)
            val_losses.append(last_val)
            scheduler.step(last_val)
            improved = last_val < (best_val - args.early_stop_min_delta)
            if improved:
                best_val = last_val
                best_step = step
                bad_val_count = 0
                best_ckpt = args.output_dir / "nrc_best.pt"
                save_checkpoint(
                    best_ckpt,
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    args=args,
                    aabb_min=train_set.aabb_min,
                    aabb_max=train_set.aabb_max,
                    n_input_dims=train_set.n_input_dims,
                    n_output_dims=train_set.n_output_dims,
                    config=config,
                    train_loss=float(loss.item()),
                    val_loss=last_val,
                )
            else:
                bad_val_count += 1
                if args.early_stop_patience > 0 and bad_val_count >= args.early_stop_patience:
                    stop_requested = True
            if not use_tqdm:
                lr_now = optimizer.param_groups[0]["lr"]
                print(
                    f"[train] step={step} val_loss={last_val:.6g} "
                    f"best={best_val:.6g}@{best_step} lr={lr_now:.3g} "
                    f"bad_val={bad_val_count}"
                )

        if use_tqdm:
            postfix = {
                "loss": f"{batch_loss:.4g}",
                "data": f"{batch_data_loss:.4g}",
            }
            if args.tv_weight > 0.0:
                postfix["tv"] = f"{batch_tv_loss:.3g}"
            if last_train is not None:
                postfix["train"] = f"{last_train:.4g}"
            if last_val is not None:
                postfix["val"] = f"{last_val:.4g}"
            postfix["lr"] = f"{optimizer.param_groups[0]['lr']:.3g}"
            pbar.set_postfix(postfix, refresh=False)

        should_plot = (
            step % args.plot_every == 0
            or step % args.save_every == 0
            or step == args.steps
            or stop_requested
        )
        if should_plot and train_steps:
            save_loss_history(
                args.output_dir, train_steps, train_losses, val_steps, val_losses
            )
            png = plot_loss_curves(
                args.output_dir, train_steps, train_losses, val_steps, val_losses
            )
            if png is not None and (step % args.save_every == 0 or step == args.steps):
                print(f"\n[train] saved loss curve -> {png}")

        if step % args.save_every == 0 or step == args.steps or stop_requested:
            ckpt = args.output_dir / f"nrc_step_{step:06d}.pt"
            save_checkpoint(
                ckpt,
                model=model,
                optimizer=optimizer,
                step=step,
                args=args,
                aabb_min=train_set.aabb_min,
                aabb_max=train_set.aabb_max,
                n_input_dims=train_set.n_input_dims,
                n_output_dims=train_set.n_output_dims,
                config=config,
                train_loss=float(loss.item()),
                val_loss=last_val,
            )
            latest = args.output_dir / "nrc_latest.pt"
            save_checkpoint(
                latest,
                model=model,
                optimizer=optimizer,
                step=step,
                args=args,
                aabb_min=train_set.aabb_min,
                aabb_max=train_set.aabb_max,
                n_input_dims=train_set.n_input_dims,
                n_output_dims=train_set.n_output_dims,
                config=config,
                train_loss=float(loss.item()),
                val_loss=last_val,
            )
        if stop_requested:
            print(
                f"\n[train] early stop at step={step} "
                f"(best_val={best_val:.6g} at step={best_step})"
            )
            break

    tcnn.free_temporary_memory()
    print("[train] done.")


if __name__ == "__main__":
    main()
