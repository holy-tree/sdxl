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

import math

import numpy as np
import torch
import torch.nn.functional as F
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


def _pil_to_tensor_01(img: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a float tensor in [0, 1] on CPU, shape [3, H, W]."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0  # [H, W, 3]
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _calc_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """PSNR (dB) between two [0, 1] tensors of identical shape."""
    mse = float(((pred - target) ** 2).mean().item())
    if mse <= 1e-12:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)


def _calc_ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> float:
    """Simplified SSIM (box-filter approximation). Returns a float in [-1, 1]."""
    if pred.ndim == 3:
        pred = pred.mean(dim=0, keepdim=True)
        target = target.mean(dim=0, keepdim=True)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    pad = window_size // 2
    mu_p = F.avg_pool2d(pred, window_size, stride=1, padding=pad)
    mu_t = F.avg_pool2d(target, window_size, stride=1, padding=pad)
    mu_p_sq = mu_p * mu_p
    mu_t_sq = mu_t * mu_t
    mu_pt = mu_p * mu_t
    sigma_p_sq = F.avg_pool2d(pred * pred, window_size, stride=1, padding=pad) - mu_p_sq
    sigma_t_sq = F.avg_pool2d(target * target, window_size, stride=1, padding=pad) - mu_t_sq
    sigma_pt = F.avg_pool2d(pred * target, window_size, stride=1, padding=pad) - mu_pt
    num = (2.0 * mu_pt + c1) * (2.0 * sigma_pt + c2)
    den = (mu_p_sq + mu_t_sq + c1) * (sigma_p_sq + sigma_t_sq + c2)
    ssim_map = num / den.clamp_min(1e-12)
    return float(ssim_map.mean().item())


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

    # VAE fp32 (force_upcast 设计); ControlNet fp32 (匹配训练); UNet/text_encoder bf16
    # ControlNet 训练时强制 fp32 (train.py:656), 推理也保持 fp32 才能精确还原 residual 幅度
    controlnet = ControlNetModel.from_pretrained(args.controlnet_path, torch_dtype=torch.float32)
    pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        controlnet=controlnet,
        variant=getattr(args, "variant", None),
        revision=getattr(args, "revision", None),
    )
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
    # 训练时 ControlNet 也是 fp32, 推理保持 fp32 才能精确还原 residual 幅度
    # bf16 noisy_latents / encoder_hidden_states 在 fp32 ControlNet 前自动 cast
    if pipeline.controlnet.dtype != weight_dtype:
        _wrap_controlnet_for_fp32(pipeline.controlnet)
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


def _wrap_controlnet_for_fp32(controlnet):
    """Wrap controlnet forward so that bf16 inputs (noisy_latents, encoder_hidden_states)
    are cast to fp32 before entering the fp32 ControlNet.
    训练时 ControlNet 是 fp32 (train.py:656), 推理保持 fp32 才能精确还原 residual 幅度,
    但 scheduler / text_encoder 输出是 bf16, 这里统一在入口 cast.
    """
    original_forward = controlnet.forward
    def _wrapped(sample, timestep, encoder_hidden_states, controlnet_cond, **kwargs):
        target_dtype = next(controlnet.parameters()).dtype
        if sample.dtype != target_dtype:
            sample = sample.to(target_dtype)
        if encoder_hidden_states.dtype != target_dtype:
            encoder_hidden_states = encoder_hidden_states.to(target_dtype)
        if controlnet_cond.dtype != target_dtype:
            controlnet_cond = controlnet_cond.to(target_dtype)
        # added_cond_kwargs 也需要 cast (text_embeds / time_ids)
        if "added_cond_kwargs" in kwargs and kwargs["added_cond_kwargs"] is not None:
            ackw = kwargs["added_cond_kwargs"]
            kwargs["added_cond_kwargs"] = {
                k: (v.to(target_dtype) if torch.is_tensor(v) and v.dtype != target_dtype else v)
                for k, v in ackw.items()
            }
        return original_forward(
            sample, timestep, encoder_hidden_states, controlnet_cond, **kwargs
        )
    controlnet.forward = _wrapped


def _discover_test_datasets(root: Path) -> Dict[str, List[Tuple[Path, Path]]]:
    """Auto-discover (lq, gt) pairs under root/{weather}/{sub_dataset}/{gt,lq}/."""
    weathers: Dict[str, List[Tuple[Path, Path]]] = {}
    for weather_dir in sorted(root.iterdir()):
        if not weather_dir.is_dir() or weather_dir.name.startswith("."):
            continue
        pairs: List[Tuple[Path, Path]] = []
        for sub_dir in sorted(weather_dir.iterdir()):
            if not sub_dir.is_dir():
                continue
            gt_dir = sub_dir / "gt"
            lq_dir = sub_dir / "lq"
            if not gt_dir.is_dir() or not lq_dir.is_dir():
                continue
            gt_names = {p.name for p in gt_dir.iterdir() if _is_image(p)}
            lq_names = {p.name for p in lq_dir.iterdir() if _is_image(p)}
            common = sorted(gt_names & lq_names)
            for name in common:
                pairs.append((lq_dir / name, gt_dir / name))
        if pairs:
            weathers[weather_dir.name] = pairs
    return weathers


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
    parser.add_argument("--test_dataset_root", type=str, default=None)
    parser.add_argument("--num_samples_per_weather", type=int, default=None)
    parser.add_argument("--num_save_per_weather", type=int, default=None)
    return parser.parse_args()


def _merge_cli_over_yaml(args: argparse.Namespace) -> TestConfig:
    """Build a :class:`TestConfig` from the YAML first, then let CLI flags win."""
    cfg = TestConfig.from_yaml(args.config) if args.config else TestConfig()
    for cli_key in ("pretrained_model_name_or_path", "controlnet_path", "lq_dir", "gt_dir", "output_dir",
                    "test_dataset_root", "num_samples_per_weather", "num_save_per_weather"):
        value = getattr(args, cli_key, None)
        if value is not None:
            setattr(cfg, cli_key, value)
    cfg.resolve_paths()
    cfg.apply_overrides(args.override)
    return cfg


def main() -> None:
    args = parse_args()
    args = _merge_cli_over_yaml(args)

    if not getattr(args, "pretrained_model_name_or_path", None):
        raise ValueError("必须提供 pretrained_model_name_or_path (YAML 或命令行)")
    if not getattr(args, "controlnet_path", None):
        raise ValueError("必须提供 controlnet_path (YAML 或命令行)")
    if not getattr(args, "output_dir", None):
        raise ValueError("必须提供 output_dir (YAML 或命令行)")
    has_auto = bool(getattr(args, "test_dataset_root", None))
    has_legacy = bool(getattr(args, "lq_dir", None))
    if not has_auto and not has_legacy:
        raise ValueError("必须提供 test_dataset_root 或 lq_dir (YAML 或命令行)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = _resolve_dtype(args.mixed_precision)
    print(f"[setup] device={device}, dtype={weight_dtype}, resolution={args.resolution}")

    # 每次测试用独立时间戳文件夹, 避免覆盖之前的输出
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.output_dir) / timestamp
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[output] root = {out_root}")

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

    num_save = int(getattr(args, "num_save_per_weather", 10))

    all_psnr: List[float] = []
    all_ssim: List[float] = []
    all_times: List[float] = []

    def _evaluate_one(
        lq_path: Path, gt_path: Optional[Path],
        prompt: str,
        out_restored: Optional[Path], out_compare: Optional[Path],
        weather: str, pair_idx: int,
    ) -> Tuple[Optional[float], Optional[float], float]:
        """Evaluate a single LQ/GT pair. Returns (psnr, ssim, dt)."""
        lq_pil = Image.open(lq_path).convert("RGB")
        cond_pil = make_conditioning(lq_pil, args.conditioning_type)
        cond_pil = center_crop(resize(cond_pil))
        lq_for_align = center_crop(resize(lq_pil))
        if args.print_color_stats:
            _color_stats(cond_pil, f"{weather} LQ(cond)")

        gt_pil: Optional[Image.Image] = None
        if gt_path is not None:
            gt_pil = Image.open(gt_path).convert("RGB")
            gt_pil = center_crop(resize(gt_pil))
            if args.print_color_stats:
                _color_stats(gt_pil, f"{weather} GT")

        best_psnr: Optional[float] = None
        best_ssim: Optional[float] = None
        best_pred: Optional[Image.Image] = None
        best_dt: float = 0.0

        n_samples = int(args.sample_times or 1)
        for si in range(n_samples):
            generator = None
            if args.seed is not None:
                generator = torch.Generator(device=device).manual_seed(int(args.seed) + pair_idx * n_samples + si)
            t0 = time.time()
            # 不要用 autocast: VAE decode 精度损失 → 输出偏暗
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

            pred = _align_color(pred, lq_for_align, args.align_method)
            if args.print_color_stats:
                _color_stats(pred, f"{weather} PRED(sample{si})")

            psnr_i = None
            ssim_i = None
            if gt_pil is not None:
                pred_t = _pil_to_tensor_01(pred).unsqueeze(0)
                gt_t = _pil_to_tensor_01(gt_pil).unsqueeze(0)
                psnr_i = _calc_psnr(pred_t, gt_t)
                ssim_i = _calc_ssim(pred_t, gt_t)
                print(f"  [{weather}] psnr={psnr_i:.2f}  ssim={ssim_i:.4f}  ({dt:.2f}s)")

            # 用第一帧保存
            if si == 0:
                if out_restored is not None and args.save_individual:
                    out_path = out_restored / f"{pair_idx:04d}_{lq_path.stem}.png"
                    pred.save(out_path)

                if out_compare is not None:
                    if gt_pil is not None:
                        panels: List[Image.Image] = [cond_pil, pred, gt_pil]
                        rows, cols = 3, 1
                        labels = ["LQ", "PRED", "GT"]
                    else:
                        panels: List[Image.Image] = [cond_pil, pred]
                        rows, cols = 2, 1
                        labels = ["LQ", "PRED"]
                    grid = make_image_grid(panels, rows=rows, cols=cols)
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
                    cmp_path = out_compare / f"{pair_idx:04d}_{lq_path.stem}.png"
                    grid_with_labels.save(cmp_path)

                best_pred = pred
                best_dt = dt

            if psnr_i is not None and (best_psnr is None or psnr_i > best_psnr):
                best_psnr = psnr_i
                best_ssim = ssim_i

        return best_psnr, best_ssim, best_dt

    # ── Auto-discovery mode ──────────────────────────────────────────
    if has_auto:
        test_root = Path(args.test_dataset_root)  # type: ignore[arg-type]
        weather_datasets = _discover_test_datasets(test_root)
        if not weather_datasets:
            raise FileNotFoundError(f"在 {test_root} 下未找到任何天气数据集 (需要 rain/snow/haze/*/{ {gt,lq}/ } 结构)")
        print(f"[setup] 发现 {len(weather_datasets)} 个天气: {', '.join(weather_datasets.keys())}")
        for w, pairs in weather_datasets.items():
            print(f"  {w}: {len(pairs)} 对")

        num_samples = int(getattr(args, "num_samples_per_weather", 50))
        per_weather_metrics: Dict[str, Tuple[float, float]] = {}

        for weather in sorted(weather_datasets.keys()):
            pairs = weather_datasets[weather]
            rng = random.Random(args.seed)
            rng.shuffle(pairs)
            eval_pairs = pairs[:num_samples]
            save_pairs = pairs[:num_save]

            out_w = out_root / weather
            out_w_restored = out_w / "restored"
            out_w_compare = out_w / "compare"
            out_w_restored.mkdir(parents=True, exist_ok=True)
            if args.save_comparison:
                out_w_compare.mkdir(parents=True, exist_ok=True)

            print(f"\n{'=' * 60}")
            print(f"[{weather}] 处理 {len(eval_pairs)}/{len(pairs)} 对 (保存前 {min(len(save_pairs), num_save)} 对)")

            prompt = weather_prompts.get(weather, base_prompt)
            w_psnr: List[float] = []
            w_ssim: List[float] = []
            w_times: List[float] = []

            for pi, (lq_p, gt_p) in enumerate(eval_pairs):
                save_this = pi < len(save_pairs)
                restored_out = out_w_restored if save_this else None
                cmp_out = out_w_compare if save_this and args.save_comparison else None
                r_psnr, r_ssim, r_dt = _evaluate_one(
                    lq_p, gt_p, prompt,
                    restored_out, cmp_out,
                    weather, pi,
                )
                w_times.append(r_dt)
                if r_psnr is not None:
                    w_psnr.append(r_psnr)
                    w_ssim.append(r_ssim)

            if w_psnr:
                mean_psnr = np.mean(w_psnr)
                mean_ssim = np.mean(w_ssim)
                print(f"\n[{weather}] avg_psnr={mean_psnr:.2f}dB  avg_ssim={mean_ssim:.4f}  (n={len(w_psnr)})")
                per_weather_metrics[weather] = (mean_psnr, mean_ssim)
                all_psnr.extend(w_psnr)
                all_ssim.extend(w_ssim)
                all_times.extend(w_times)

        # ── 总体汇总 ──
        if all_psnr:
            print(f"\n{'=' * 60}")
            print("[Result Summary]")
            for weather, (p, s) in per_weather_metrics.items():
                print(f"  {weather:10s}: psnr={p:.2f}dB  ssim={s:.4f}")
            print(f"  {'Overall':10s}: psnr={np.mean(all_psnr):.2f}dB  ssim={np.mean(all_ssim):.4f}  (n={len(all_psnr)})")
            print(f"  {'Time':10s}: avg={np.mean(all_times):.3f}s  total={np.sum(all_times):.2f}s")
            print(f"  output: {out_root}")

    # ── Legacy mode (单目录 lq_dir / gt_dir) ─────────────────────────
    else:
        file_ext = getattr(args, "file_extension", ".png")
        limit = getattr(args, "limit", None)
        lq_paths = _collect_inputs(args.lq_dir, file_ext, limit)  # type: ignore[arg-type]
        gt_dir = Path(args.gt_dir) if args.gt_dir else None
        print(f"[setup] 共找到 {len(lq_paths)} 张输入图 (gt_dir={gt_dir})")

        out_restored = out_root / "restored"
        out_compare = out_root / "compare"
        out_restored.mkdir(parents=True, exist_ok=True)
        if args.save_comparison:
            out_compare.mkdir(parents=True, exist_ok=True)

        for idx, lq_path in enumerate(lq_paths):
            print("=" * 60)
            print(f"[{idx + 1}/{len(lq_paths)}] {lq_path.name}")
            gt_match = _maybe_match_gt(lq_path, gt_dir)
            prompt = base_prompt or (_auto_prompt_for(lq_path.stem, weather_prompts) if auto_prompt else "")
            cmp_out = out_compare if args.save_comparison else None
            r_psnr, r_ssim, r_dt = _evaluate_one(
                lq_path, gt_match, prompt,
                out_restored, cmp_out,
                "legacy", idx,
            )
            if r_psnr is not None:
                all_psnr.append(r_psnr)
                all_ssim.append(r_ssim)
            all_times.append(r_dt)

        if all_times:
            print("=" * 60)
            print(
                f"[done] avg={np.mean(all_times):.3f}s  min={np.min(all_times):.3f}s  "
                f"max={np.max(all_times):.3f}s  total={np.sum(all_times):.2f}s"
            )
            if all_psnr:
                print(
                    f"       avg_psnr={np.mean(all_psnr):.2f}dB  avg_ssim={np.mean(all_ssim):.4f}  "
                    f"(n={len(all_psnr)})"
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
