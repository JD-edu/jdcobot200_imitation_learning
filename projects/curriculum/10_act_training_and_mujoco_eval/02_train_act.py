#!/usr/bin/env python3
"""LeRobot Dataset v3로 JDCobot200 ACT policy를 학습합니다."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import sys
import time
import types

import lerobot
import numpy as np
import torch
from torch.utils.data import DataLoader

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features

# LeRobot 0.4.4의 policies/__init__.py가 사용하지 않는 GR00T까지 즉시 import하면서
# 현재 Python 환경에서 dataclass 오류를 낸다. ACT 하위 모듈만 정상적으로 import하도록
# package path를 등록한다. ACT 구현 파일 자체는 수정하거나 복제하지 않는다.
if "lerobot.policies" not in sys.modules:
    policies_package = types.ModuleType("lerobot.policies")
    policies_package.__path__ = [str(Path(lerobot.__file__).resolve().parent / "policies")]
    policies_package.__package__ = "lerobot.policies"
    sys.modules["lerobot.policies"] = policies_package

from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.processor_act import make_act_pre_post_processors


CAMERA_KEY = "observation.images.dataset_camera"


@dataclass
class TrainingSummary:
    lerobot_version: str
    dataset_repo_id: str
    dataset_root: str
    episodes: int
    frames: int
    fps: int
    device: str
    steps: int
    batch_size: int
    chunk_size: int
    n_action_steps: int
    learning_rate: float
    final_loss: float
    elapsed_s: float
    seed: int


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA를 요청했지만 torch.cuda.is_available()이 False입니다.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS를 요청했지만 사용할 수 없습니다.")
    return device


def make_delta_timestamps(indices: list[int], fps: int) -> list[float]:
    return [index / fps for index in indices]


def atomic_write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(path)


def save_pretrained(output: Path, policy: ACTPolicy, preprocessor, postprocessor,
                    optimizer: torch.optim.Optimizer, step: int,
                    summary: TrainingSummary) -> None:
    output.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(output)
    preprocessor.save_pretrained(output)
    postprocessor.save_pretrained(output)
    torch.save({"step": step, "optimizer": optimizer.state_dict()}, output / "training_state.pt")
    atomic_write_json(output / "training_summary.json", asdict(summary))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/jdcobot200_synthetic_pnp")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--n-action-steps", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--save-freq", type=int, default=20_000)
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--amp", action="store_true", help="CUDA 자동 혼합 정밀도 사용")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--dim-model", type=int, default=512)
    parser.add_argument("--dim-feedforward", type=int, default=3200)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-encoder-layers", type=int, default=4)
    parser.add_argument("--n-vae-encoder-layers", type=int, default=4)
    parser.add_argument("--no-pretrained-backbone", action="store_true")
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"출력 경로가 이미 존재합니다: {args.output}")
    if args.steps < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("steps와 batch-size는 1 이상, num-workers는 0 이상이어야 합니다.")
    if not 1 <= args.n_action_steps <= args.chunk_size:
        raise ValueError("n-action-steps는 1 이상 chunk-size 이하여야 합니다.")
    if args.save_freq < 1 or args.log_freq < 1:
        raise ValueError("save-freq와 log-freq는 1 이상이어야 합니다.")
    if args.dim_model % args.n_heads:
        raise ValueError("dim-model은 n-heads로 나누어져야 합니다.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    if args.amp and device.type != "cuda":
        raise ValueError("--amp는 CUDA에서만 사용할 수 있습니다.")

    dataset_root = args.dataset_root.resolve()
    metadata = LeRobotDatasetMetadata(args.repo_id, root=dataset_root)
    policy_features = dataset_to_policy_features(metadata.features)
    required = {CAMERA_KEY, "observation.state", "action"}
    if not required.issubset(policy_features):
        raise RuntimeError(f"ACT 필수 feature가 없습니다: {sorted(required - set(policy_features))}")
    output_features = {
        key: feature for key, feature in policy_features.items()
        if feature.type is FeatureType.ACTION
    }
    input_features = {
        key: feature for key, feature in policy_features.items()
        if key not in output_features
    }
    # 디버그 feature는 dataset_to_policy_features 단계에서 자동 제외된다.
    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        device=str(device),
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        dim_model=args.dim_model,
        dim_feedforward=args.dim_feedforward,
        n_heads=args.n_heads,
        n_encoder_layers=args.n_encoder_layers,
        n_vae_encoder_layers=args.n_vae_encoder_layers,
        pretrained_backbone_weights=(
            None if args.no_pretrained_backbone else "ResNet18_Weights.IMAGENET1K_V1"
        ),
        optimizer_lr=args.learning_rate,
        optimizer_weight_decay=args.weight_decay,
        optimizer_lr_backbone=args.learning_rate,
    )
    delta_timestamps = {
        "action": make_delta_timestamps(cfg.action_delta_indices, metadata.fps),
    }
    dataset = LeRobotDataset(
        args.repo_id,
        root=dataset_root,
        delta_timestamps=delta_timestamps,
    )
    if dataset.num_episodes < 2:
        print("경고: episode가 2개 미만이므로 실제 학습용으로 부족합니다.")

    policy = ACTPolicy(cfg).to(device)
    policy.train()
    preprocessor, postprocessor = make_act_pre_post_processors(
        cfg, dataset_stats=metadata.stats
    )
    optimizer = torch.optim.AdamW(
        policy.get_optim_params(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )
    iterator = iter(loader)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    started = time.monotonic()
    final_loss = float("nan")

    print(
        f"ACT 학습 시작: episodes={dataset.num_episodes}, frames={dataset.num_frames}, "
        f"device={device}, chunk={args.chunk_size}, batch={args.batch_size}"
    )
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = preprocessor(batch)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp):
            loss, loss_dict = policy.forward(batch)
        if not torch.isfinite(loss):
            raise RuntimeError(f"step {step}: loss가 유한하지 않습니다: {loss.item()}")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        final_loss = float(loss.detach().cpu())
        if step == 1 or step % args.log_freq == 0 or step == args.steps:
            elapsed = time.monotonic() - started
            print(f"step={step}/{args.steps} loss={final_loss:.6f} "
                  f"l1={loss_dict['l1_loss']:.6f} elapsed={elapsed:.1f}s")
        if step % args.save_freq == 0 and step < args.steps:
            checkpoint = args.output / "checkpoints" / f"{step:06d}" / "pretrained_model"
            summary = TrainingSummary(
                "0.4.4", args.repo_id, str(dataset_root), dataset.num_episodes,
                dataset.num_frames, dataset.fps, str(device), step, args.batch_size,
                args.chunk_size, args.n_action_steps, args.learning_rate, final_loss,
                time.monotonic() - started, args.seed,
            )
            save_pretrained(checkpoint, policy, preprocessor, postprocessor,
                            optimizer, step, summary)

    final_dir = args.output / "pretrained_model"
    summary = TrainingSummary(
        "0.4.4", args.repo_id, str(dataset_root), dataset.num_episodes,
        dataset.num_frames, dataset.fps, str(device), args.steps, args.batch_size,
        args.chunk_size, args.n_action_steps, args.learning_rate, final_loss,
        time.monotonic() - started, args.seed,
    )
    save_pretrained(final_dir, policy, preprocessor, postprocessor,
                    optimizer, args.steps, summary)
    print(f"학습 완료: {final_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"오류: {exc}")
        raise SystemExit(1)
