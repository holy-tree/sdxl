"""Inference / visualization script for SDXL + ControlNet weather restoration.

Run with::

    python test.py --config configs/test.yaml
    python test.py --lq_dir <dir> --gt_dir <dir> --controlnet_path <path> --output_dir <path>

For every LQ image the script produces:
  * The restored image saved under ``<output_dir>/restored/``
  * A side-by-side comparison panel under ``<output_dir>/compare/``:
        [LQ | Restored]               (when --gt_dir is not provided)
        [LQ | Restored | GT]          (when --gt_dir is provided)
"""

from __future__ import annotations

import argparse
import gc
import glob
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils import make_image_grid

from dataloaders import DEFAULT_WEATHER_PROMPTS, make_conditioning
from schemas import TestConfig
from utils.attention import enable_efficient_attention

logger = logging.getLogger(__name__)

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTENSIONS


def _collect_inputs(input_path: str, file_ext: str, limit: Optional[int]) -> List[Path]:
    p = Path(input_path)
    candidates: List[Path]
    if p.is_file() and _is_image(p):
        candidates = [p]
    elif p.is_dir():
        exts = {file_ext.lower()} if file_ext else IMG_EXTENSIONS
        candidates = sorted(q for q in p.iterdir() if q.is_file() and q.suffix.lower() in exts)
    else:
        matches = sorted(Path(m) for m in glob.glob(input_path) if _is_image(Path(m)))
        candidates = matches
    if not candidates:
        raise FileNotFoundError(f"未在 {input_path} 找到任何图片 (扩展名: {file_ext})")
    if limit is not None and limit > 0:
        candidates = candidates[: int(limit)]
    return candidates


def _maybe_match_gt(lq_path: Path, gt_dir: Optional[Path]) -> Optional[Path]:
    if gt_dir is None:
        return None
    cand = gt_dir / lq_path.name
    if cand.is_file():
        return cand
    return None


def _auto_prompt_for(stem: str, prompts: Dict[str, str]) -> str:
    low = stem.lower()
    for key in ("rain", "snow", "haze", "fog", "frost", "night"):
        if key in low:
            return prompts.get(key, "")
    return ""


def _resolve_dtype(name: str) -> torch.dtype:
    name = (name or "fp16").lower()
    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "no": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported mixed_precision '{name}'")
    return mapping[name]


def _print_model_info(pipeline: StableDiffusionXLControlNetPipeline) -> None:
    print("=" * 60)
    print("[Model Info]")
    for name, mod in (
        ("vae", pipeline.vae),
        ("text_encoder", pipeline.text_encoder),
        ("text_encoder_2", pipeline.text_encoder_2),
        ("unet", pipeline.unet),
        ("controlnet", pipeline.controlnet),
    ):
        print(
            f"  {name:18s} dtype={mod.dtype}  training={mod.training}  "
            f"gradient_checkpointing={getattr(mod, 'gradient_checkpointing', 'N/A')}"
        )
    print("=" * 60)


def _color_stats(img: Image.Image, label: str) -> Dict[str, float]:
    arr = np.asarray(img.convert("RGB")).astype(np.float32)
    mean = float(arr.mean())
    std = float(arr.std())
    ch_mean = arr.reshape(-1, 3).mean(axis=0).tolist()
    hsv = np.asarray(img.convert("HSV"))
    v_mean = float(hsv[..., 2].mean())
    if label:
        print(
            f"  [{label}] mean={mean:6.2f} std={std:6.2f} "
            f"R/G/B={ch_mean[0]:5.1f}/{ch_mean[1]:5.1f}/{ch_mean[2]:5.1f} V={v_mean:5.1f}"
        )
    return {"mean": mean, "std": std, "R": ch_mean[0], "G": ch_mean[1], "B": ch_mean[2], "V": v_mean}


def _align_color(pred: Image.Image, lq: Image.Image, method: str) -> Image.Image:
    method = (method or "none").lower()
    if method == "none":
        return pred
    if method == "adain":
        from torchvision.transforms.functional import to_tensor, to_pil_image

        pred_t = to_tensor(pred)
        lq_t = to_tensor(lq.convert("RGB"))
        mu_p = pred_t.mean(dim=(1, 2), keepdim=True)
        sd_p = pred_t.std(dim=(1, 2), keepdim=True) + 1e-5
        mu_l = lq_t.mean(dim=(1, 2), keepdim=True)
        sd_l = lq_t.std(dim=(1, 2), keepdim=True) + 1e-5
        out = (pred_t - mu_p) / sd_p * sd_l + mu_l
        return to_pil_image(out.clamp(0, 1))
    if method == "wavelet":
        try:
            from wavelet_color_fix import wavelet_color_fix

            return wavelet_color_fix(pred, lq)
        except ImportError:
            print("[wavelet] 未找到 wavelet_color_fix, 跳过色彩校正")
            return pred
    raise ValueError(f"Unknown align_method '{method}'")


def _build_pipeline(args, device: torch.device, weight_dtype: torch.dtype) -> StableDiffusionXLControlNetPipeline:
    print(f"[load] pretrained = {args.pretrained_model_name_or_path}")
    print(f"[load] controlnet = {args.controlnet_path}")

    # VAE fp32 (force_upcast 设计); ControlNet bf16 (匹配训练 autocast 计算精度)
    controlnet = ControlNetModel.from_pretrained(args.controlnet_path, torch_dtype=weight_dtype)
    pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        controlnet=controlnet,
        # 不传 torch_dtype, 让各组件保留 from_pretrained 默认 fp32
        variant=getattr(args, "variant", None),
        revision=getattr(args, "revision", None),
    )
    # UNet 默认 fp32, 显式转 bf16 (匹配训练); text_encoder 默认 fp16, 转 bf16 避免 dtype mismatch
    pipeline.unet.to(weight_dtype)
    pipeline.text_encoder.to(weight_dtype)
    pipeline.text_encoder_2.to(weight_dtype)

    for module in (pipeline.vae, pipeline.text_encoder, pipeline.text_encoder_2, pipeline.unet, pipeline.controlnet):
        module.requires_grad_(False)
        module.eval()
        if hasattr(module, "disable_gradient_checkpointing"):
            module.disable_gradient_checkpointing()

    if args.enable_xformers_memory_efficient_attention and is_xformers_available():
        try:
            pipeline.enable_xformers_memory_efficient_attention()
        except Exception as exc:  # noqa: BLE001
            print(f"[xformers] 启用失败: {exc}")

    backend = args.attention_backend
    if backend == "auto":
        backend = "xformers" if args.enable_xformers_memory_efficient_attention else "auto"
    used = enable_efficient_attention(pipeline.unet, backend=backend)
    used_cn = enable_efficient_attention(pipeline.controlnet, backend=backend)
    print(f"[setup] attention backend: unet={used}, controlnet={used_cn}")

    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    # 训练时 ControlNet 接收 [0,1]; 关闭 pipeline 默认的 [-1,1] 归一化
    pipeline.control_image_processor.do_normalize = False
    # bf16 latents + fp32 VAE 时 PyTorch 不会自动 cast, 必须显式 wrap 避免 dtype mismatch
    # 训练时 VAE 是 fp32, 推理保持 fp32 decode 才能避免精度损失导致输出偏暗
    if pipeline.vae.dtype != weight_dtype:
        _wrap_vae_decode_for_fp32(pipeline.vae)
    if getattr(args, "enable_model_cpu_offload", False) and device.type == "cuda":
        pipeline.enable_model_cpu_offload()
    else:
        pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def _wrap_vae_decode_for_fp32(vae):
    """Wrap vae.decode so that bf16 latents are cast to vae.dtype before decoding."""
    original_decode = vae.decode
    def _wrapped(latents, *args, **kwargs):
        if latents.dtype != vae.dtype:
            latents = latents.to(vae.dtype)
        return original_decode(latents, *args, **kwargs)
    vae.decode = _wrapped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SDXL + ControlNet weather-restoration inference.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    parser.add_argument(
        "--override", nargs=argparse.REMAINDER, default=None, help="key=value overrides"
    )
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None)
    parser.add_argument("--controlnet_path", type=str, default=None)
    parser.add_argument("--lq_dir", type=str, default=None)
    parser.add_argument("--gt_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def _merge_cli_over_yaml(args: argparse.Namespace) -> TestConfig:
    """Build a :class:`TestConfig` from the YAML first, then let CLI flags win."""
    cfg = TestConfig.from_yaml(args.config) if args.config else TestConfig()
    for cli_key in ("pretrained_model_name_or_path", "controlnet_path", "lq_dir", "gt_dir", "output_dir"):
        value = getattr(args, cli_key, None)
        if value is not None:
            setattr(cfg, cli_key, value)
    cfg.resolve_paths()
    cfg.apply_overrides(args.override)
    return cfg


def main() -> None:
    args = parse_args()
    args = _merge_cli_over_yaml(args)

    required = ["pretrained_model_name_or_path", "controlnet_path", "lq_dir", "output_dir"]
    for k in required:
        if not getattr(args, k, None):
            raise ValueError(f"必须提供 {k} (YAML 或命令行)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = _resolve_dtype(args.mixed_precision)
    print(f"[setup] device={device}, dtype={weight_dtype}, resolution={args.resolution}")

    file_ext = getattr(args, "file_extension", ".png")
    limit = getattr(args, "limit", None)
    lq_paths = _collect_inputs(args.lq_dir, file_ext, limit)
    gt_dir = Path(args.gt_dir) if args.gt_dir else None
    print(f"[setup] 共找到 {len(lq_paths)} 张输入图 (gt_dir={gt_dir})")

    out_root = Path(args.output_dir)
    out_restored = out_root / "restored"
    out_compare = out_root / "compare"
    out_restored.mkdir(parents=True, exist_ok=True)
    if args.save_comparison:
        out_compare.mkdir(parents=True, exist_ok=True)

    pipeline = _build_pipeline(args, device, weight_dtype)
    if args.print_model_info:
        _print_model_info(pipeline)

    weather_prompts = {**DEFAULT_WEATHER_PROMPTS, **(args.weather_prompts or {})}
    auto_prompt = bool(getattr(args, "auto_weather_prompt", False))
    base_prompt = args.prompt or ""
    negative_prompt = args.negative_prompt or ""

    _interp_mode = getattr(transforms.InterpolationMode, args.image_interpolation_mode.upper(), transforms.InterpolationMode.BILINEAR)
    resize = transforms.Resize(args.resolution, interpolation=_interp_mode)
    center_crop = transforms.CenterCrop(args.resolution)
    to_tensor = transforms.ToTensor()

    times: List[float] = []
    for idx, lq_path in enumerate(lq_paths):
        print("=" * 60)
        print(f"[{idx + 1}/{len(lq_paths)}] {lq_path.name}")
        lq_pil = Image.open(lq_path).convert("RGB")
        original_size = lq_pil.size
        cond_pil = make_conditioning(lq_pil, args.conditioning_type)
        cond_pil = center_crop(resize(cond_pil))
        lq_for_align = center_crop(resize(lq_pil))
        if args.print_color_stats:
            _color_stats(cond_pil, "LQ(cond)")

        gt_pil: Optional[Image.Image] = None
        gt_match = _maybe_match_gt(lq_path, gt_dir)
        if gt_match is not None:
            gt_pil = Image.open(gt_match).convert("RGB")
            gt_pil = center_crop(resize(gt_pil))
            if args.print_color_stats:
                _color_stats(gt_pil, "GT")

        prompt = base_prompt or (_auto_prompt_for(lq_path.stem, weather_prompts) if auto_prompt else "")

        for sample_idx in range(int(args.sample_times or 1)):
            generator = None
            if args.seed is not None:
                generator = torch.Generator(device=device).manual_seed(int(args.seed) + sample_idx)
            t0 = time.time()
            # 不要用 autocast(weight_dtype): 即使 VAE/ControlNet 已加载成 fp32,
            # autocast 仍会把所有 op 强制降到 bf16, 导致 VAE decode 数值精度损失 → 输出偏暗
            with torch.no_grad():
                pred = pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=cond_pil,
                    num_inference_steps=int(args.num_inference_steps),
                    guidance_scale=float(args.guidance_scale),
                    controlnet_conditioning_scale=float(getattr(args, "controlnet_conditioning_scale", 1.0)),
                    height=args.resolution,
                    width=args.resolution,
                    generator=generator,
                ).images[0]
            dt = time.time() - t0
            times.append(dt)

            pred = _align_color(pred, lq_for_align, args.align_method)
            if args.print_color_stats:
                _color_stats(pred, f"PRED(sample{sample_idx})")

            # 直接保存 pipeline 原生输出分辨率 args.resolution (即 512×512),
            # 不要 resize 回 original_size, 否则 BICUBIC 上采样会引入模糊
            if args.save_individual:
                out_path = out_restored / f"{lq_path.stem}_sample{sample_idx:02d}.png"
                pred.save(out_path)
                print(f"  saved -> {out_path}  ({dt:.2f}s, size={pred.size})")

            if args.save_comparison:
                # 用 3 行 × 1 列布局, 总图接近正方形, 比 1×3 横条观感更好
                if gt_pil is not None:
                    panels: List[Image.Image] = [cond_pil, pred, gt_pil]
                    rows, cols = 3, 1
                    labels = ["LQ", "PRED", "GT"]
                else:
                    panels: List[Image.Image] = [cond_pil, pred]
                    rows, cols = 2, 1
                    labels = ["LQ", "PRED"]
                grid = make_image_grid(panels, rows=rows, cols=cols)
                # 拼一个竖向小标签条 (白色背景) 写 LQ/PRED/GT
                from PIL import ImageDraw, ImageFont
                w, h = panels[0].size
                label_h = 32
                grid_with_labels = Image.new("RGB", (w, rows * h + label_h * rows), "white")
                draw = ImageDraw.Draw(grid_with_labels)
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
                except OSError:
                    font = ImageFont.load_default()
                for i, (panel, label) in enumerate(zip(panels, labels)):
                    y_label = i * (h + label_h)
                    draw.rectangle([0, y_label, w, y_label + label_h], fill="white")
                    draw.text((10, y_label + 6), label, fill="black", font=font)
                    grid_with_labels.paste(panel, box=(0, y_label + label_h))
                cmp_path = out_compare / f"{lq_path.stem}_sample{sample_idx:02d}.png"
                grid_with_labels.save(cmp_path)
                print(f"  compare -> {cmp_path}  (size={grid_with_labels.size})")

    if times:
        print("=" * 60)
        print(
            f"[done] avg={np.mean(times):.3f}s  min={np.min(times):.3f}s  "
            f"max={np.max(times):.3f}s  total={np.sum(times):.2f}s"
        )
        print(f"  restored: {out_restored}")
        if args.save_comparison:
            print(f"  compare:  {out_compare}")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    main()
