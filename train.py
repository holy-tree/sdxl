"""Training entry point for SDXL + ControlNet on paired weather-restoration data.

Run with::

    python train.py --config configs/train.yaml

The script mirrors the structure of ``train_controlnet_sdxl.py`` from the official
diffusers examples, but:
  * Hyperparameters live in a YAML file (loaded via ``--config``).
  * Uses a custom paired dataset (``dataloaders/PairedWeatherDataset``).
  * Freezes the SDXL UNet / VAE / text encoders; only the ControlNet branch is trained.
"""

from __future__ import annotations

import argparse
import gc
import logging
import math
import os
import random
import shutil
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionXLControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available, make_image_grid
from diffusers.utils.import_utils import is_torch_npu_available, is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from packaging import version

from dataloaders import (
    DEFAULT_WEATHER_PROMPTS,
    PairedWeatherDataset,
    PrecomputedEmbeddingDataset,
    build_precomputed_dataset,
)
from schemas import TrainConfig
from utils.attention import enable_efficient_attention, move_optimistic

logger = get_logger(__name__)
if is_wandb_available():
    import wandb
if is_torch_npu_available():
    torch.npu.config.allow_internal_format = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_text_encoder_class(pretrained_path: str, revision: str, subfolder: str = "text_encoder"):
    config = PretrainedConfig.from_pretrained(pretrained_path, subfolder=subfolder, revision=revision)
    arch = config.architectures[0]
    if arch == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    if arch == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    raise ValueError(f"Unsupported text encoder architecture '{arch}' in {pretrained_path}/{subfolder}")


def _seed_everything(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    pixel_values = torch.stack([e["pixel_values"] for e in examples]).to(memory_format=torch.contiguous_format).float()
    cond_values = torch.stack([e["conditioning_pixel_values"] for e in examples]).to(
        memory_format=torch.contiguous_format
    ).float()
    prompt_embeds = torch.stack([e["prompt_embeds"] for e in examples])
    text_embeds = torch.stack([e["text_embeds"] for e in examples])
    time_ids = torch.stack([e["time_ids"] for e in examples])
    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": cond_values,
        "prompt_ids": prompt_embeds,
        "unet_added_conditions": {"text_embeds": text_embeds, "time_ids": time_ids},
    }


def _resolve_validation_inputs(args) -> tuple[Optional[List[str]], Optional[List[str]]]:
    """Pick validation images/prompts.

    Validation images are auto-discovered from each weather's ``<dataset_root>/<weather>/test/LQ``
    folder. ``num_validation_images`` controls how many test images to take *per weather type*
    (so total = ``num_validation_images * len(weather_types)``). When ``validation_image`` is set
    in the YAML it acts as a manual override (paths or single path).
    """
    if not args.run_validation:
        return None, None

    val_prompts = args.validation_prompt
    if isinstance(val_prompts, str):
        val_prompts = [val_prompts] if val_prompts else [""]
    if not val_prompts:
        val_prompts = [""]

    val_imgs_cfg = args.validation_image
    if isinstance(val_imgs_cfg, str):
        val_imgs_cfg = [val_imgs_cfg]
    if val_imgs_cfg is None:
        root = Path(args.dataset_root)
        per_weather = max(1, int(args.num_validation_images))
        resolved: List[str] = []
        for weather in args.weather_types:
            test_lq = root / weather / "test" / "LQ"
            if not test_lq.is_dir():
                print(f"[验证] 跳过 {weather}: 缺少 {test_lq}")
                continue
            pngs = sorted(
                p for p in test_lq.iterdir()
                if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
            )
            if not pngs:
                print(f"[验证] 跳过 {weather}: {test_lq} 下无图片")
                continue
            selected = pngs[:per_weather]
            print(f"[验证] {weather}: 选 {len(selected)} 张 (per_weather={per_weather})")
            resolved.extend(str(p) for p in selected)
        if not resolved:
            print("[验证] 未在任意天气的 test/LQ 找到图片, 跳过本轮验证。")
            return None, None
        val_imgs = resolved
    else:
        val_imgs = list(val_imgs_cfg)

    if len(val_imgs) == 1 and len(val_prompts) > 1:
        val_imgs = val_imgs * len(val_prompts)
    elif len(val_prompts) == 1 and len(val_imgs) > 1:
        val_prompts = val_prompts * len(val_imgs)
    elif len(val_imgs) != len(val_prompts):
        raise ValueError(
            f"validation_image ({len(val_imgs)}) 与 validation_prompt ({len(val_prompts)}) 数量不一致"
        )
    return val_imgs, val_prompts


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _log_validation(
    *,
    vae: Optional[AutoencoderKL],
    unet: Optional[UNet2DConditionModel],
    controlnet: Optional[ControlNetModel],
    args,
    accelerator: Accelerator,
    weight_dtype: torch.dtype,
    step: int,
    is_final_validation: bool = False,
) -> None:
    val_imgs, val_prompts = _resolve_validation_inputs(args)
    if not val_imgs:
        return

    if not is_final_validation:
        controlnet = accelerator.unwrap_model(controlnet)
        pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            vae=vae,
            unet=unet,
            controlnet=controlnet,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
        )
    else:
        controlnet = ControlNetModel.from_pretrained(args.output_dir, torch_dtype=weight_dtype)
        if args.pretrained_vae_model_name_or_path is not None:
            vae = AutoencoderKL.from_pretrained(args.pretrained_vae_model_name_or_path, torch_dtype=weight_dtype)
        else:
            vae = AutoencoderKL.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="vae", torch_dtype=weight_dtype
            )
        pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            vae=vae,
            controlnet=controlnet,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
        )

    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention and is_xformers_available():
        try:
            pipeline.enable_xformers_memory_efficient_attention()
        except Exception as exc:  # noqa: BLE001
            print(f"[验证] xformers 启用失败: {exc}")

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=accelerator.device).manual_seed(int(args.seed))

    image_logs = []
    autocast_ctx = nullcontext() if is_final_validation else torch.autocast(accelerator.device.type)

    from torchvision import transforms as _tv
    resize = _tv.Resize(args.resolution, interpolation=_tv.InterpolationMode.LANCZOS)
    center_crop = _tv.CenterCrop(args.resolution)

    for val_prompt, val_image in zip(val_prompts, val_imgs):
        cond = Image.open(val_image).convert("RGB")
        cond = center_crop(resize(cond))
        images = []
        for _ in range(args.num_validation_images):
            with autocast_ctx:
                image = pipeline(
                    prompt=val_prompt,
                    image=cond,
                    num_inference_steps=args.validation_inference_steps,
                    guidance_scale=args.validation_guidance_scale,
                    negative_prompt=args.validation_negative_prompt,
                    generator=generator,
                ).images[0]
            images.append(image)
        image_logs.append(
            {"validation_image": cond, "images": images, "validation_prompt": val_prompt}
        )

    out_dir = Path(args.output_dir) / "validation" / f"step-{step:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for log in image_logs:
        prompt_str = (log["validation_prompt"] or "<empty>").replace(" ", "_")[:40]
        panel = [log["validation_image"]] + log["images"]
        grid = make_image_grid(panel, rows=1, cols=len(panel))
        grid.save(out_dir / f"{Path(val_imgs[0]).stem}_{prompt_str}.png")

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                formatted = [np.asarray(log["validation_image"])] + [np.asarray(img) for img in log["images"]]
                tracker.writer.add_images(
                    log["validation_prompt"] or "validation",
                    np.stack(formatted),
                    step,
                    dataformats="NHWC",
                )
        elif tracker.name == "wandb":
            wandb_images = [wandb.Image(log["validation_image"], caption="conditioning")]
            wandb_images += [wandb.Image(img, caption=log["validation_prompt"]) for img in log["images"]]
            tracker.log({"validation": wandb_images})

    del pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SDXL + ControlNet training for weather restoration.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--override",
        nargs=argparse.REMAINDER,
        default=None,
        help="Optional key=value overrides, e.g. --override train_batch_size=2 learning_rate=1e-4",
    )
    args = parser.parse_args()
    return args


def main() -> None:
    cli = parse_args()
    args = TrainConfig.from_yaml(cli.config).resolve_paths().apply_overrides(cli.override)

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError("Cannot use both --report_to=wandb and --hub_token (security).")

    if args.mixed_precision == "bf16" and torch.backends.mps.is_available():
        raise ValueError("MPS does not support bfloat16 mixed precision.")

    if args.resolution % 8 != 0:
        raise ValueError("`resolution` must be divisible by 8.")

    if args.pretrained_vae_model_name_or_path is not None:
        vae_subfolder = None
    else:
        vae_subfolder = "vae"

    if args.weather_num_samples is None:
        args.weather_num_samples = {}
    if args.weather_prompts is None:
        args.weather_prompts = {}
    args.weather_prompts = {**DEFAULT_WEATHER_PROMPTS, **args.weather_prompts}

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=args.logging_dir
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        diffusers.utils.logging.set_verbosity_info()
    else:
        diffusers.utils.logging.set_verbosity_error()

    _seed_everything(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        if args.push_to_hub:
            from huggingface_hub import create_repo

            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
            ).repo_id
            args.hub_model_id = repo_id

    if accelerator.is_main_process:
        snapshot = dict(vars(args))
        snapshot.pop("validation_prompt", None)
        snapshot.pop("validation_image", None)
        import yaml as _yaml

        with open(Path(args.output_dir) / "config_snapshot.yaml", "w", encoding="utf-8") as f:
            _yaml.safe_dump(snapshot, f, allow_unicode=True, sort_keys=False)

    # ---------------- Tokenizers & Text Encoders ----------------
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision, use_fast=False
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision, use_fast=False
    )
    text_encoder_cls_one = _import_text_encoder_class(args.pretrained_model_name_or_path, args.revision)
    text_encoder_cls_two = _import_text_encoder_class(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder=vae_subfolder,
        revision=args.revision,
        variant=args.variant,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )

    if args.controlnet_model_name_or_path:
        logger.info("Loading existing controlnet weights from %s", args.controlnet_model_name_or_path)
        controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
    else:
        logger.info("Initializing controlnet weights from unet (from_unet)")
        controlnet = ControlNetModel.from_unet(unet)

    # ---------------- Freeze everything except ControlNet ----------------
    def _unwrap(model):
        model = accelerator.unwrap_model(model)
        return model._orig_mod if is_compiled_module(model) else model

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                i = len(weights) - 1
                while len(weights) > 0:
                    weights.pop()
                    model = models[i]
                    sub_dir = "controlnet"
                    model.save_pretrained(os.path.join(output_dir, sub_dir))
                    i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                model = models.pop()
                load_model = ControlNetModel.from_pretrained(input_dir, subfolder="controlnet")
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    controlnet.train()

    if args.enable_npu_flash_attention and is_torch_npu_available():
        unet.enable_npu_flash_attention()

    backend = args.attention_backend
    if backend == "auto":
        backend = "xformers" if args.enable_xformers_memory_efficient_attention else "auto"
    used_unet = enable_efficient_attention(unet, backend=backend)
    used_cn = enable_efficient_attention(controlnet, backend=backend)
    if used_unet == "xformers" or used_cn == "xformers":
        logger.info("xformers memory-efficient attention enabled")
    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()
        unet.enable_gradient_checkpointing()

    if _unwrap(controlnet).dtype != torch.float32:
        raise ValueError(
            f"ControlNet must stay in float32, got {_unwrap(controlnet).dtype}."
        )

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise ImportError("Install bitsandbytes to use 8-bit AdamW.") from exc
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    optimizer = optimizer_class(
        controlnet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if args.pretrained_vae_model_name_or_path is not None:
        vae.to(accelerator.device, dtype=weight_dtype)
    else:
        vae.to(accelerator.device, dtype=torch.float32)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    # ---------------- Dataset & pre-computed text embeddings ----------------
    train_dataset = PairedWeatherDataset(
        dataset_root=args.dataset_root,
        weather_types=args.weather_types,
        splits=args.splits,
        resolution=args.resolution,
        conditioning_type=args.conditioning_type,
        null_text_ratio=args.null_text_ratio,
        use_prompt=args.use_prompt,
        prompt_ratio=args.prompt_ratio,
        weather_prompts=args.weather_prompts,
        weather_num_samples=args.weather_num_samples,
        interpolation=args.image_interpolation_mode,
        augment=args.augment,
        preload=args.preload_dataset,
    )

    original_size = (args.resolution, args.resolution)
    crops = (args.crops_coords_top_left_h, args.crops_coords_top_left_w)
    target_size = (args.resolution, args.resolution)

    pre_dataset: PrecomputedEmbeddingDataset = build_precomputed_dataset(
        dataset=train_dataset,
        text_encoders=[text_encoder_one, text_encoder_two],
        tokenizers=[tokenizer_one, tokenizer_two],
        device=accelerator.device,
        original_size=original_size,
        crops_coords_top_left=crops,
        target_size=target_size,
        proportion_empty_prompts=args.proportion_empty_prompts,
        batch_size=max(1, args.train_batch_size * 2),
    )

    del text_encoder_one, text_encoder_two, tokenizer_one, tokenizer_two
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_dataloader = torch.utils.data.DataLoader(
        pre_dataset,
        shuffle=True,
        collate_fn=_collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=args.pin_memory,
        pin_memory_device=args.pin_memory_device if args.pin_memory else None,
        persistent_workers=args.persistent_workers and args.dataloader_num_workers > 0,
        prefetch_factor=args.dataloader_prefetch_factor if args.dataloader_num_workers > 0 else None,
        drop_last=True,
    )

    if args.max_train_steps == -1:
        args.max_train_steps = None

    if args.max_train_steps is None:
        len_per_process = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_per_process / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=num_training_steps_for_scheduler,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler
    )

    if args.channels_last:
        try:
            controlnet.to(memory_format=torch.channels_last)
            logger.info("ControlNet cast to channels_last memory format")
        except Exception as exc:  # noqa: BLE001
            logger.warning("channels_last on ControlNet failed: %s", exc)

    if args.torch_compile and hasattr(torch, "compile"):
        try:
            unet = torch.compile(unet, mode=args.torch_compile_mode, fullgraph=False)
            controlnet = torch.compile(controlnet, mode=args.torch_compile_mode, fullgraph=False)
            logger.info("torch.compile enabled (mode=%s)", args.torch_compile_mode)
        except Exception as exc:  # noqa: BLE001
            logger.warning("torch.compile failed: %s", exc)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        tracker_cfg = dict(vars(args))
        tracker_cfg.pop("validation_prompt", None)
        tracker_cfg.pop("validation_image", None)
        tracker_cfg.pop("weather_prompts", None)
        accelerator.init_trackers(args.tracker_project_name, config=tracker_cfg)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(pre_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if dirs else None
        if path is None:
            accelerator.print("No checkpoint to resume from, starting fresh.")
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch
    initial_global_step = global_step

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    image_logs = None
    train_start = time.time()
    non_blocking = bool(args.non_blocking_transfer) and torch.cuda.is_available()
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet):
                if args.pretrained_vae_model_name_or_path is not None:
                    pixel_values = batch["pixel_values"].to(dtype=weight_dtype, non_blocking=non_blocking)
                else:
                    pixel_values = batch["pixel_values"]
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                if args.pretrained_vae_model_name_or_path is None:
                    latents = latents.to(weight_dtype, non_blocking=non_blocking)

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents.float(), noise.float(), timesteps).to(
                    dtype=weight_dtype, non_blocking=non_blocking
                )

                controlnet_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype, non_blocking=non_blocking)
                down_res, mid_res = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=batch["prompt_ids"],
                    added_cond_kwargs=batch["unet_added_conditions"],
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                )

                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=batch["prompt_ids"],
                    added_cond_kwargs=batch["unet_added_conditions"],
                    down_block_additional_residuals=[
                        s.to(dtype=weight_dtype, non_blocking=non_blocking) for s in down_res
                    ],
                    mid_block_additional_residual=mid_res.to(dtype=weight_dtype, non_blocking=non_blocking),
                    return_dict=False,
                )[0]

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(controlnet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.distributed_type == DistributedType.DEEPSPEED or accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        if args.checkpoints_total_limit is not None:
                            ckpts = [
                                d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")
                            ]
                            ckpts = sorted(ckpts, key=lambda x: int(x.split("-")[1]))
                            if len(ckpts) >= args.checkpoints_total_limit:
                                num_to_remove = len(ckpts) - args.checkpoints_total_limit + 1
                                removing = ckpts[:num_to_remove]
                                for r in removing:
                                    shutil.rmtree(os.path.join(args.output_dir, r))
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info("Saved state to %s", save_path)

                    if (
                        args.run_validation
                        and args.validation_steps > 0
                        and global_step % args.validation_steps == 0
                    ):
                        _log_validation(
                            vae=vae,
                            unet=unet,
                            controlnet=controlnet,
                            args=args,
                            accelerator=accelerator,
                            weight_dtype=weight_dtype,
                            step=global_step,
                        )

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        controlnet = _unwrap(controlnet)
        controlnet.save_pretrained(args.output_dir)
        if args.run_validation and args.validation_steps > 0:
            _log_validation(
                vae=None,
                unet=None,
                controlnet=None,
                args=args,
                accelerator=accelerator,
                weight_dtype=weight_dtype,
                step=global_step,
                is_final_validation=True,
            )

    accelerator.end_training()
    if accelerator.is_main_process:
        elapsed = time.time() - train_start
        print(f"[done] training finished in {elapsed/60.0:.1f} min, controlnet saved to {args.output_dir}")


if __name__ == "__main__":
    main()
