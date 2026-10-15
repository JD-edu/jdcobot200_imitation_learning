#!/usr/bin/env python3
"""Hugging Face LeRobot의 pretrained SmolVLA를 로컬 데이터로 미세조정합니다."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import sys
import tempfile
import time
import types

import draccus
from huggingface_hub import hf_hub_download
import lerobot
import numpy as np
import torch
from torch.utils.data import DataLoader

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features

# 설치된 checkout에서 사용하지 않는 GR00T가 policies/__init__.py에서 import되며 발생하는
# dataclass 오류를 피하고, 공식 SmolVLA 하위 모듈만 import한다.
if "lerobot.policies" not in sys.modules:
    package = types.ModuleType("lerobot.policies")
    package.__path__ = [str(Path(lerobot.__file__).resolve().parent / "policies")]
    package.__package__ = "lerobot.policies"
    sys.modules["lerobot.policies"] = package

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors


CAMERA_KEY = "observation.images.dataset_camera"


@dataclass
class Summary:
    lerobot_version: str
    pretrained_policy: str
    dataset_root: str
    dataset_repo_id: str
    episodes: int
    frames: int
    device: str
    steps: int
    batch_size: int
    chunk_size: int
    final_loss: float
    elapsed_s: float
    trainable_parameters: int
    total_parameters: int
    seed: int


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(path)


def load_compatible_config(model_id: str) -> SmolVLAConfig:
    """LeRobot checkout과 Hub config 사이의 `type` parsing 오류를 우회합니다.

    현재 설치의 PreTrainedConfig.from_pretrained는 `type`을 제거하기 전에 concrete
    SmolVLAConfig로 한 번 decode하여 실패한다. 같은 공식 config에서 discriminator만
    제거한 뒤 현재 SmolVLAConfig로 decode한다.
    """
    local = Path(model_id)
    config_path = local / "config.json" if local.is_dir() else Path(
        hf_hub_download(repo_id=model_id, filename="config.json")
    )
    with config_path.open(encoding="utf-8") as stream:
        raw = json.load(stream)
    policy_type = raw.pop("type", "smolvla")
    if policy_type != "smolvla":
        raise ValueError(f"SmolVLA checkpoint가 아닙니다: type={policy_type!r}")
    with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8") as stream:
        json.dump(raw, stream)
        stream.flush()
        with draccus.config_type("json"):
            return draccus.parse(SmolVLAConfig, stream.name, args=[])


def save(output: Path, policy, preprocessor, postprocessor, optimizer, scheduler,
         step: int, summary: Summary) -> None:
    output.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(output)
    preprocessor.save_pretrained(output)
    postprocessor.save_pretrained(output)
    torch.save({"step": step, "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict()}, output / "training_state.pt")
    atomic_json(output / "training_summary.json", asdict(summary))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/jdcobot200_smolvla_pnp")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pretrained-policy", default="lerobot/smolvla_base")
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--n-action-steps", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-10)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--save-freq", type=int, default=5000)
    parser.add_argument("--log-freq", type=int, default=20)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--amp", choices=["none", "bf16", "fp16"], default="bf16")
    parser.add_argument("--seed", type=int, default=1100)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"출력 폴더가 이미 존재합니다: {args.output}")
    if min(args.steps, args.batch_size, args.chunk_size, args.n_action_steps) < 1:
        raise ValueError("steps/batch/chunk/action-steps는 1 이상이어야 합니다.")
    if args.n_action_steps > args.chunk_size:
        raise ValueError("n-action-steps는 chunk-size 이하여야 합니다.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA를 사용할 수 없습니다. GPU 환경을 확인하세요.")
    if args.amp != "none" and device.type != "cuda":
        raise ValueError("AMP는 CUDA 학습에서만 지원합니다.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    root = args.dataset_root.resolve()
    metadata = LeRobotDatasetMetadata(args.repo_id, root=root)
    policy_features = dataset_to_policy_features(metadata.features)
    required = {CAMERA_KEY, "observation.state", "action"}
    if not required.issubset(policy_features):
        raise RuntimeError(f"필수 feature 누락: {sorted(required - set(policy_features))}")
    output_features = {k: v for k, v in policy_features.items()
                       if v.type is FeatureType.ACTION}
    input_features = {k: v for k, v in policy_features.items()
                      if k not in output_features}

    print(f"pretrained policy 다운로드/로드: {args.pretrained_policy}")
    cfg = load_compatible_config(args.pretrained_policy)
    cfg.input_features = input_features
    cfg.output_features = output_features
    cfg.device = str(device)
    cfg.chunk_size = args.chunk_size
    cfg.n_action_steps = args.n_action_steps
    cfg.optimizer_lr = args.learning_rate
    cfg.optimizer_weight_decay = args.weight_decay
    # SmolVLA 기본 미세조정: vision encoder와 VLM은 동결하고 action expert/state projection 학습.
    cfg.freeze_vision_encoder = True
    cfg.train_expert_only = True
    cfg.train_state_proj = True

    dataset = LeRobotDataset(
        args.repo_id, root=root,
        delta_timestamps={"action": [i / metadata.fps for i in range(args.chunk_size)]},
    )
    if dataset.num_episodes < 1:
        raise RuntimeError("학습 episode가 없습니다.")
    first = dataset[0]
    if not first.get("task"):
        raise RuntimeError("SmolVLA에 필요한 task 문장이 데이터에 없습니다.")

    policy = SmolVLAPolicy.from_pretrained(args.pretrained_policy, config=cfg).to(device)
    preprocessor, postprocessor = make_smolvla_pre_post_processors(
        cfg, dataset_stats=metadata.stats
    )
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    total = sum(p.numel() for p in policy.parameters())
    if trainable == 0:
        raise RuntimeError("학습 가능한 parameter가 없습니다.")
    optimizer = torch.optim.AdamW(
        policy.get_optim_params(), lr=args.learning_rate,
        weight_decay=args.weight_decay, betas=(0.9, 0.95), eps=1e-8,
    )

    def lr_factor(step: int) -> float:
        if args.warmup_steps > 0 and step < args.warmup_steps:
            return max(1e-8, (step + 1) / args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
        return 0.025 + 0.975 * 0.5 * (1.0 + np.cos(np.pi * np.clip(progress, 0, 1)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda",
                        drop_last=False, persistent_workers=args.num_workers > 0)
    iterator = iter(loader)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                 "none": torch.float32}[args.amp]
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp == "fp16")
    started = time.monotonic()
    final_loss = float("nan")
    print(f"SmolVLA 학습 시작: episodes={dataset.num_episodes}, frames={dataset.num_frames}, "
          f"device={device}, trainable={trainable:,}/{total:,}")

    policy.train()
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = preprocessor(batch)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=args.amp != "none"):
            loss, _loss_dict = policy.forward(batch)
        if not torch.isfinite(loss):
            raise RuntimeError(f"step {step}: loss가 유한하지 않습니다: {loss.item()}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        final_loss = float(loss.detach().cpu())
        if step == 1 or step % args.log_freq == 0 or step == args.steps:
            print(f"step={step}/{args.steps} loss={final_loss:.6f} "
                  f"lr={scheduler.get_last_lr()[0]:.2e} "
                  f"elapsed={time.monotonic() - started:.1f}s")
        if step % args.save_freq == 0 and step < args.steps:
            summary = Summary(lerobot.__version__, args.pretrained_policy, str(root),
                              args.repo_id, dataset.num_episodes, dataset.num_frames,
                              str(device), step, args.batch_size, args.chunk_size,
                              final_loss, time.monotonic() - started, trainable, total, args.seed)
            save(args.output / "checkpoints" / f"{step:06d}" / "pretrained_model",
                 policy, preprocessor, postprocessor, optimizer, scheduler, step, summary)

    summary = Summary(lerobot.__version__, args.pretrained_policy, str(root),
                      args.repo_id, dataset.num_episodes, dataset.num_frames,
                      str(device), args.steps, args.batch_size, args.chunk_size,
                      final_loss, time.monotonic() - started, trainable, total, args.seed)
    final = args.output / "pretrained_model"
    save(final, policy, preprocessor, postprocessor, optimizer, scheduler, args.steps, summary)
    print(f"학습 완료: {final}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"오류: {exc}")
        raise SystemExit(1)
