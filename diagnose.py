"""诊断脚本：精确定位"全黑图像"的根因。

四个测试:
1) base SDXL + dummy CN, cn_scale=0   → 验证 base SDXL 是否本身就有问题
2) trained CN + bf16 VAE, cn_scale=1  → 复现训练时 _log_validation 行为
3) trained CN + bf16 VAE, cn_scale=0  → 验证 trained ControlNet 即使不加也全黑
4) trained CN + fp32 VAE, cn_scale=1  → 验证 bf16 VAE 是否是元凶
"""

from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import sys
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from torchvision import transforms

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionXLControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from transformers import (
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)


CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b")
CONTROLNET_PATH = "experiment/weather_controlnet"
COND_IMG_PATH = "D:/Projects/pycharm/WeaFU-main/dataprocess/rain/train/LQ/rain-1089.png"
OUT_DIR = Path("experiment/diagnose")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESOLUTION = 512
WEIGHT_DTYPE = torch.bfloat16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
NUM_STEPS = 20
GUIDANCE = 5.0
NEG_PROMPT = "dotted, noise, blur, lowres, smooth"


def stats(img: Image.Image, label: str) -> dict:
    arr = np.asarray(img.convert("RGB")).astype(np.float32)
    info = {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
    print(f"  [{label}] mean={info['mean']:.2f}  std={info['std']:.2f}  min={info['min']:.0f}  max={info['max']:.0f}")
    return info


def build_pipeline(use_trained_cn: bool, fp32_vae: bool):
    """直接从缓存子目录手动组装管线 (避开 model_index.json 缺失问题)."""
    print(f"\n[load] use_trained_cn={use_trained_cn}, fp32_vae={fp32_vae}")
    vae_dtype = torch.float32 if fp32_vae else WEIGHT_DTYPE
    vae = AutoencoderKL.from_pretrained(os.path.join(CACHE_DIR, "vae"), torch_dtype=vae_dtype)
    unet = UNet2DConditionModel.from_pretrained(os.path.join(CACHE_DIR, "unet"), torch_dtype=WEIGHT_DTYPE)
    text_encoder = CLIPTextModel.from_pretrained(os.path.join(CACHE_DIR, "text_encoder"), torch_dtype=WEIGHT_DTYPE)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(os.path.join(CACHE_DIR, "text_encoder_2"), torch_dtype=WEIGHT_DTYPE)
    tokenizer = CLIPTokenizer.from_pretrained(os.path.join(CACHE_DIR, "tokenizer"))
    tokenizer_2 = CLIPTokenizer.from_pretrained(os.path.join(CACHE_DIR, "tokenizer_2"))
    scheduler = DDPMScheduler.from_pretrained(os.path.join(CACHE_DIR, "scheduler"))

    if use_trained_cn:
        cn = ControlNetModel.from_pretrained(CONTROLNET_PATH, torch_dtype=WEIGHT_DTYPE)
    else:
        cn = ControlNetModel.from_unet(unet).to(WEIGHT_DTYPE)

    pipeline = StableDiffusionXLControlNetPipeline(
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        unet=unet,
        controlnet=cn,
        scheduler=scheduler,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(DEVICE)
    pipeline.set_progress_bar_config(disable=True)
    print(
        f"  vae={pipeline.vae.dtype}  unet={pipeline.unet.dtype}  "
        f"cn={pipeline.controlnet.dtype}  te1={pipeline.text_encoder.dtype}  "
        f"te2={pipeline.text_encoder_2.dtype}"
    )
    return pipeline


def run(pipeline, cond, tag, controlnet_scale):
    generator = torch.Generator(device=DEVICE).manual_seed(SEED)
    image = pipeline(
        prompt="",
        negative_prompt=NEG_PROMPT,
        image=cond,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE,
        height=RESOLUTION,
        width=RESOLUTION,
        controlnet_conditioning_scale=controlnet_scale,
        generator=generator,
    ).images[0]
    info = stats(image, f"{tag} (scale={controlnet_scale})")
    image.save(OUT_DIR / f"{tag}.png")
    return image, info


def main():
    print(f"[setup] device={DEVICE}, dtype={WEIGHT_DTYPE}, resolution={RESOLUTION}")

    cond_pil = Image.open(COND_IMG_PATH).convert("RGB")
    interp = transforms.InterpolationMode.BILINEAR
    cond_pil = transforms.CenterCrop(RESOLUTION)(transforms.Resize(RESOLUTION, interpolation=interp)(cond_pil))
    cond_pil.save(OUT_DIR / "cond_lq.png")
    stats(cond_pil, "LQ input")

    # ============ TEST A ============
    print("\n===== TEST A: dummy CN, cn_scale=0 (与无 ControlNet 等价) =====")
    pipe = build_pipeline(use_trained_cn=False, fp32_vae=False)
    img_a, info_a = run(pipe, cond_pil, "A_dummy_scale0", 0.0)
    del pipe
    torch.cuda.empty_cache()

    # ============ TEST B ============
    print("\n===== TEST B: trained CN + bf16 VAE, cn_scale=1 (复现 _log_validation) =====")
    pipe = build_pipeline(use_trained_cn=True, fp32_vae=False)
    img_b, info_b = run(pipe, cond_pil, "B_trained_bf16vae_scale1", 1.0)
    del pipe
    torch.cuda.empty_cache()

    # ============ TEST C ============
    print("\n===== TEST C: trained CN + bf16 VAE, cn_scale=0 =====")
    pipe = build_pipeline(use_trained_cn=True, fp32_vae=False)
    img_c, info_c = run(pipe, cond_pil, "C_trained_bf16vae_scale0", 0.0)
    del pipe
    torch.cuda.empty_cache()

    # ============ TEST D ============
    print("\n===== TEST D: trained CN + fp32 VAE, cn_scale=1 =====")
    pipe = build_pipeline(use_trained_cn=True, fp32_vae=True)
    img_d, info_d = run(pipe, cond_pil, "D_trained_fp32vae_scale1", 1.0)
    del pipe
    torch.cuda.empty_cache()

    print("\n===== 汇总 =====")
    print(f"  A (dummy CN, scale=0):             mean={info_a['mean']:6.2f}  std={info_a['std']:6.2f}  max={info_a['max']:.0f}")
    print(f"  B (trained, bf16 VAE, scale=1):    mean={info_b['mean']:6.2f}  std={info_b['std']:6.2f}  max={info_b['max']:.0f}")
    print(f"  C (trained, bf16 VAE, scale=0):    mean={info_c['mean']:6.2f}  std={info_c['std']:6.2f}  max={info_c['max']:.0f}")
    print(f"  D (trained, fp32 VAE, scale=1):    mean={info_d['mean']:6.2f}  std={info_d['std']:6.2f}  max={info_d['max']:.0f}")
    print(f"\n所有结果: {OUT_DIR}")


if __name__ == "__main__":
    main()