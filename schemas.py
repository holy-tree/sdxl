"""Configuration schemas for SDXL + ControlNet weather-restoration training / inference.

All defaults live here.  YAML configs are loaded into the dataclass, and any field
the user omits in YAML falls back to the dataclass default.  This is the single
source of truth for the experiment configuration — no scattered ``_DEFAULTS`` dict.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


_VALID_MIXED_PRECISION = ("no", "fp16", "bf16")
_VALID_CONDITIONING = ("lq", "gray", "canny")
_VALID_LR_SCHEDULER = (
    "linear", "cosine", "cosine_with_restarts", "polynomial",
    "constant", "constant_with_warmup",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML top-level must be a mapping, got {type(data).__name__}")
    return data


def _resolve_paths(output_dir: str, logging_dir: str) -> tuple[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = out / logging_dir
    log.mkdir(parents=True, exist_ok=True)
    return str(out), str(log)


def _apply_overrides(cfg: Any, override_list: Optional[List[str]]) -> None:
    if not override_list:
        return
    valid = {f.name for f in fields(cfg)}
    for item in override_list:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key not in valid:
            raise ValueError(
                f"Unknown --override key '{key}'. Valid keys: {sorted(valid)}"
            )
        try:
            cast_value: Any = yaml.safe_load(value)
        except Exception:  # noqa: BLE001
            cast_value = value
        setattr(cfg, key, cast_value)


# ---------------------------------------------------------------------------
# TrainConfig
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """All training-time hyperparameters.

    Defaults are chosen for SDXL + ControlNet weather restoration on a single GPU.
    The YAML file may override any subset of fields; everything else falls back here.
    """

    # ----- paths -----
    pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-xl-base-1.0"
    pretrained_vae_model_name_or_path: Optional[str] = None
    controlnet_model_name_or_path: Optional[str] = None
    revision: Optional[str] = None
    variant: Optional[str] = None
    output_dir: str = "./experiment/weather_controlnet"
    resume_from_checkpoint: Optional[str] = None
    cache_dir: Optional[str] = None

    # ----- data -----
    dataset_root: str = "./datasets/allweather"
    weather_types: List[str] = field(default_factory=lambda: ["rain", "snow", "haze"])
    splits: List[str] = field(default_factory=lambda: ["train"])
    resolution: int = 1024
    conditioning_type: str = "lq"
    weather_num_samples: Dict[str, int] = field(default_factory=dict)

    # ----- prompts / CFG -----
    use_prompt: bool = False
    prompt_ratio: float = 0.2
    proportion_empty_prompts: float = 0.0
    null_text_ratio: float = 0.5
    weather_prompts: Dict[str, str] = field(default_factory=dict)

    # ----- training -----
    learning_rate: float = 5.0e-5
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 5
    max_train_steps: int = -1
    seed: Optional[int] = 42
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = True
    use_8bit_adam: bool = False
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 0.01
    adam_epsilon: float = 1.0e-08
    max_grad_norm: float = 1.0
    set_grads_to_none: bool = True
    scale_lr: bool = False

    # ----- LR scheduler -----
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 500
    lr_num_cycles: float = 0.5
    lr_power: float = 1.0

    # ----- DataLoader -----
    dataloader_num_workers: int = 4
    dataloader_prefetch_factor: int = 4
    pin_memory: bool = True
    pin_memory_device: Optional[str] = "cuda"   # "cuda" / "xpu" / None
    persistent_workers: bool = True
    non_blocking_transfer: bool = True
    image_interpolation_mode: str = "lanczos"
    augment: bool = True
    channels_last: bool = True                  # NHWC layout for image tensors
    preload_dataset: bool = False               # pre-decode all images to RAM (small datasets)

    # ----- SDXL specific -----
    crops_coords_top_left_h: int = 0
    crops_coords_top_left_w: int = 0

    # ----- optimization flags -----
    enable_xformers_memory_efficient_attention: bool = False
    attention_backend: str = "auto"             # auto | xformers | sdpa | math
    enable_npu_flash_attention: bool = False
    allow_tf32: bool = False
    torch_compile: bool = False                 # torch.compile(unet/controlnet)
    torch_compile_mode: str = "reduce-overhead" # default | reduce-overhead | max-autotune

    # ----- checkpointing -----
    checkpointing_steps: int = 500
    checkpoints_total_limit: Optional[int] = 3

    # ----- logging / hub -----
    logging_dir: str = "logs"
    report_to: str = "tensorboard"
    tracker_project_name: str = "sdxl_weather_controlnet"
    push_to_hub: bool = False
    hub_token: Optional[str] = None
    hub_model_id: Optional[str] = None

    # ----- validation (in-training inference) -----
    run_validation: bool = True
    validation_steps: int = 500
    num_validation_images: int = 1
    validation_image: Optional[Any] = None
    validation_prompt: Any = ""
    validation_guidance_scale: float = 5.0
    validation_inference_steps: int = 20
    validation_negative_prompt: str = "dotted, noise, blur, lowres, smooth"

    # ------------------------------------------------------------------
    # Constructors / validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        if self.mixed_precision not in _VALID_MIXED_PRECISION:
            raise ValueError(
                f"mixed_precision must be one of {_VALID_MIXED_PRECISION}, got {self.mixed_precision!r}"
            )
        if self.conditioning_type not in _VALID_CONDITIONING:
            raise ValueError(
                f"conditioning_type must be one of {_VALID_CONDITIONING}, got {self.conditioning_type!r}"
            )
        if self.lr_scheduler not in _VALID_LR_SCHEDULER:
            raise ValueError(
                f"lr_scheduler must be one of {_VALID_LR_SCHEDULER}, got {self.lr_scheduler!r}"
            )
        if self.resolution <= 0 or self.resolution % 8 != 0:
            raise ValueError("resolution must be a positive integer divisible by 8")
        if self.train_batch_size <= 0:
            raise ValueError("train_batch_size must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        return cls(**_load_yaml(path))

    def resolve_paths(self) -> "TrainConfig":
        out, log = _resolve_paths(self.output_dir, self.logging_dir)
        self.output_dir = out
        self.logging_dir = log
        return self

    def apply_overrides(self, override_list: Optional[List[str]]) -> "TrainConfig":
        _apply_overrides(self, override_list)
        return self

    def to_namespace(self) -> argparse.Namespace:
        return argparse.Namespace(**asdict(self))


# ---------------------------------------------------------------------------
# TestConfig
# ---------------------------------------------------------------------------

@dataclass
class TestConfig:
    """Inference / visualization parameters for the trained ControlNet."""

    # ----- model -----
    pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-xl-base-1.0"
    pretrained_vae_model_name_or_path: Optional[str] = None
    controlnet_path: str = "./experiment/weather_controlnet"
    variant: Optional[str] = None
    revision: Optional[str] = None

    # ----- io -----
    lq_dir: str = "./experiment/test_imgs"
    gt_dir: Optional[str] = None
    output_dir: str = "./experiment/test_results"
    file_extension: str = ".png"
    limit: Optional[int] = None

    # ----- inference -----
    resolution: int = 1024
    mixed_precision: str = "bf16"
    num_inference_steps: int = 20
    guidance_scale: float = 5.0
    controlnet_conditioning_scale: float = 1.0
    seed: int = 42
    sample_times: int = 1

    # ----- prompts -----
    prompt: str = ""
    negative_prompt: str = "dotted, noise, blur, lowres, smooth"
    weather_prompts: Dict[str, str] = field(default_factory=dict)
    auto_weather_prompt: bool = False

    # ----- post-processing -----
    conditioning_type: str = "lq"
    align_method: str = "none"
    save_individual: bool = True
    save_comparison: bool = True

    # ----- speed -----
    enable_xformers_memory_efficient_attention: bool = False
    attention_backend: str = "auto"
    enable_model_cpu_offload: bool = False

    # ----- diagnostics -----
    print_model_info: bool = True
    print_color_stats: bool = True

    def __post_init__(self) -> None:
        if self.conditioning_type not in _VALID_CONDITIONING:
            raise ValueError(
                f"conditioning_type must be one of {_VALID_CONDITIONING}, got {self.conditioning_type!r}"
            )
        if self.mixed_precision not in _VALID_MIXED_PRECISION:
            raise ValueError(
                f"mixed_precision must be one of {_VALID_MIXED_PRECISION}, got {self.mixed_precision!r}"
            )
        if self.resolution <= 0 or self.resolution % 8 != 0:
            raise ValueError("resolution must be a positive integer divisible by 8")
        if self.align_method not in ("none", "wavelet", "adain"):
            raise ValueError(
                f"align_method must be one of ('none','wavelet','adain'), got {self.align_method!r}"
            )

    @classmethod
    def from_yaml(cls, path: str) -> "TestConfig":
        return cls(**_load_yaml(path))

    def resolve_paths(self) -> "TestConfig":
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        return self

    def apply_overrides(self, override_list: Optional[List[str]]) -> "TestConfig":
        _apply_overrides(self, override_list)
        return self

    def to_namespace(self) -> argparse.Namespace:
        return argparse.Namespace(**asdict(self))
