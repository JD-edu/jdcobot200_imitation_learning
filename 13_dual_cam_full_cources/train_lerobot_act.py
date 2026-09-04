#!/usr/bin/env python3
"""Train a standard LeRobot ACT checkpoint on the local JDCobot200 dataset."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("HF_HOME", str(Path(tempfile.gettempdir()) / "lerobot_training_hf"))

from lerobot_act_compat import enable_act_only_imports

enable_act_only_imports()

from lerobot.configs.types import FeatureType  # noqa: E402
from lerobot.datasets.lerobot_dataset import (  # noqa: E402
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from lerobot.datasets.utils import dataset_to_policy_features  # noqa: E402
from lerobot.policies.act.configuration_act import ACTConfig  # noqa: E402
from lerobot.policies.act.modeling_act import ACTPolicy  # noqa: E402
from lerobot.policies.act.processor_act import make_act_pre_post_processors  # noqa: E402


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "lerobot_dataset")
    parser.add_argument("--repo-id", default="local/jdcobot200_dual_camera_pick_place")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "act_jdcobot200")
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--n-action-steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--log-freq", type=int, default=50)
    parser.add_argument("--save-freq", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu", "mps"), default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def save_checkpoint(policy, preprocessor, postprocessor, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(directory)
    preprocessor.save_pretrained(directory)
    postprocessor.save_pretrained(directory)


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.chunk_size < 1:
        raise ValueError("steps, batch-size, and chunk-size must be positive")
    if not 1 <= args.n_action_steps <= args.chunk_size:
        raise ValueError("n-action-steps must be in [1, chunk-size]")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} is not empty; use --overwrite")
        shutil.rmtree(args.output_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    use_amp = args.amp and device.type == "cuda"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root)
    features = dataset_to_policy_features(metadata.features)
    output_features = {
        key: feature for key, feature in features.items()
        if feature.type is FeatureType.ACTION
    }
    input_features = {
        key: feature for key, feature in features.items()
        if key not in output_features
    }
    config = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        device=str(device),
        use_amp=use_amp,
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        pretrained_backbone_weights=None,
        optimizer_lr=args.lr,
        optimizer_weight_decay=args.weight_decay,
        optimizer_lr_backbone=args.lr,
        push_to_hub=False,
    )
    policy = ACTPolicy(config).to(device).train()
    preprocessor, postprocessor = make_act_pre_post_processors(
        config, dataset_stats=metadata.stats
    )
    delta_timestamps = {
        "action": [index / metadata.fps for index in config.action_delta_indices]
    }
    dataset = LeRobotDataset(
        args.repo_id,
        root=args.dataset_root,
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=len(dataset) >= args.batch_size,
        persistent_workers=args.num_workers > 0,
    )
    optimizer = torch.optim.AdamW(
        policy.get_optim_params(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    run_info = {
        "dataset_root": str(args.dataset_root.resolve()),
        "repo_id": args.repo_id,
        "dataset_episodes": metadata.total_episodes,
        "dataset_frames": metadata.total_frames,
        "fps": metadata.fps,
        "device": str(device),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "chunk_size": args.chunk_size,
        "n_action_steps": args.n_action_steps,
        "seed": args.seed,
    }
    (args.output_dir / "training_run.json").write_text(
        json.dumps(run_info, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(run_info, indent=2), flush=True)

    step = 0
    while step < args.steps:
        for batch in loader:
            batch = preprocessor(batch)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss, loss_dict = policy(batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
            step += 1
            if step == 1 or step % args.log_freq == 0:
                details = " ".join(f"{key}={value:.5f}" for key, value in loss_dict.items())
                print(f"step={step}/{args.steps} loss={loss.item():.5f} {details}", flush=True)
            if step % args.save_freq == 0:
                save_checkpoint(
                    policy, preprocessor, postprocessor,
                    args.output_dir / "checkpoints" / f"step_{step:06d}" / "pretrained_model",
                )
            if step >= args.steps:
                break

    final_dir = args.output_dir / "checkpoints" / "last" / "pretrained_model"
    save_checkpoint(policy, preprocessor, postprocessor, final_dir)
    print(f"Saved standard LeRobot ACT checkpoint to {final_dir.resolve()}")


if __name__ == "__main__":
    main()
