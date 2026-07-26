"""诊断脚本：精确定位"全黑图像"的根因。

四个测试:
1) base SDXL + dummy CN, cn_scale=0      → 验证 base SDXL 是否本身就有问题
2) trained CN + bf16 VAE, cn_scale=1     → 复现训练时 _log_validation 行为
3) trained CN + bf16 VAE, cn_scale=0     → 验证 trained ControlNet 即便不加也全黑
4) trained CN + fp32 VAE, cn_scale=1     → 验证 bf16 VAE 是否是元凶

默认路径为 autodl 服务器配置 (可通过命令行参数修改):
  --hf_cache       HF 模型缓存目录
  --controlnet_dir ControlNet 权重目录 (包含 config.json + diffusion_pytorch_model.*)
  --cond_img       conditioning 图像路径
  --out_dir        结果输出目录
"""

from __future__ import annotations

import os
import sys
import argparse

# 服务器路径设置 (放最前)
DEFAULT_HF_HOME = "/root/autodl-tmp/hf_cache"
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
DEFAULT_PRETRAINED = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_CONTROLNET_DIR = "/root/autodl-tmp/experiment/weather_controlnet/checkpoint-2000"
DEFAULT_COND_IMG = "/root/autodl-tmp/datasets/rain/test/LQ/000003.jpg"
DEFAULT_OUT_DIR = "/root/autodl-tmp/experiment/diagnose"

os.environ.setdefault("HF_HOME", DEFAULT_HF_HOME)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", DEFAULT_HF_HOME)
os.environ.setdefault("HF_ENDPOINT", DEFAULT_HF_ENDPOINT)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

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


RESOLUTION = 512
WEIGHT_DTYPE = torch.bfloat16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
NUM_STEPS = 20
GUIDANCE = 5.0
NEG_PROMPT = "dotted, noise, blur, lowres, smooth"


def stats(img, label):
    arr = np.asarray(img.convert("RGB")).astype(np.float32)
    info = {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
    print(
        f"  [{label}] mean={info['mean']:6.2f}  std={info['std']:6.2f}  "
        f"min={info['min']:.0f}  max={info['max']:.0f}"
    )
    return info


def find_snapshot_dir(hf_cache_root: str, repo_id: str) -> str:
    """在 HF 缓存里找 snapshots/<hash> 目录."""
    repo_dir = os.path.join(hf_cache_root, "models--" + repo_id.replace("/", "--"))
    snapshots = os.path.join(repo_dir, "snapshots")
    if not os.path.isdir(snapshots):
        raise FileNotFoundError(f"找不到 {snapshots}")
    # 取第一个 snapshot 目录
    for name in sorted(os.listdir(snapshots)):
        full = os.path.join(snapshots, name)
        if os.path.isdir(full):
            return full
    raise FileNotFoundError(f"{snapshots} 下没有 snapshot 子目录")


def build_pipeline(
    pretrained_dir: str,
    controlnet_dir: str,
    use_trained_cn: bool,
    fp32_vae: bool,
):
    """手动组装管线 (避免依赖 model_index.json)."""
    print(f"\n[load] use_trained_cn={use_trained_cn}, fp32_vae={fp32_vae}")
    print(f"  pretrained_dir = {pretrained_dir}")
    print(f"  controlnet_dir = {controlnet_dir if use_trained_cn else '<dummy>'}")

    vae_dtype = torch.float32 if fp32_vae else WEIGHT_DTYPE
    vae = AutoencoderKL.from_pretrained(os.path.join(pretrained_dir, "vae"), torch_dtype=vae_dtype)
    unet = UNet2DConditionModel.from_pretrained(os.path.join(pretrained_dir, "unet"), torch_dtype=WEIGHT_DTYPE)
    text_encoder = CLIPTextModel.from_pretrained(os.path.join(pretrained_dir, "text_encoder"), torch_dtype=WEIGHT_DTYPE)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        os.path.join(pretrained_dir, "text_encoder_2"), torch_dtype=WEIGHT_DTYPE
    )
    tokenizer = CLIPTokenizer.from_pretrained(os.path.join(pretrained_dir, "tokenizer"))
    tokenizer_2 = CLIPTokenizer.from_pretrained(os.path.join(pretrained_dir, "tokenizer_2"))
    scheduler = DDPMScheduler.from_pretrained(os.path.join(pretrained_dir, "scheduler"))

    if use_trained_cn:
        cn = ControlNetModel.from_pretrained(controlnet_dir, torch_dtype=WEIGHT_DTYPE)
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
    image.save(os.path.join(args.out_dir, f"{tag}.png"))
    return image, info


def main():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_cache", default=os.environ.get("HF_HOME", DEFAULT_HF_HOME))
    parser.add_argument("--pretrained_dir", default=None,
                        help="SDXL 模型目录. 默认自动从 hf_cache 推断")
    parser.add_argument("--controlnet_dir", default=DEFAULT_CONTROLNET_DIR)
    parser.add_argument("--cond_img", default=DEFAULT_COND_IMG)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--ckpt", default=None,
                        help="可选: 指定 checkpoint 目录 (会自动加 controlnet/)")
    args = parser.parse_args()

    if args.ckpt is not None:
        args.controlnet_dir = os.path.join(args.ckpt, "controlnet")

    os.makedirs(args.out_dir, exist_ok=True)

    if args.pretrained_dir is None:
        args.pretrained_dir = find_snapshot_dir(args.hf_cache, DEFAULT_PRETRAINED)
    print(f"[setup] device={DEVICE}, dtype={WEIGHT_DTYPE}, resolution={RESOLUTION}")
    print(f"[setup] HF cache    = {args.hf_cache}")
    print(f"[setup] pretrained  = {args.pretrained_dir}")
    print(f"[setup] controlnet  = {args.controlnet_dir}")
    print(f"[setup] cond_img    = {args.cond_img}")
    print(f"[setup] out_dir     = {args.out_dir}")

    cond_pil = Image.open(args.cond_img).convert("RGB")
    interp = transforms.InterpolationMode.BILINEAR
    cond_pil = transforms.CenterCrop(RESOLUTION)(
        transforms.Resize(RESOLUTION, interpolation=interp)(cond_pil)
    )
    cond_pil.save(os.path.join(args.out_dir, "cond_lq.png"))
    stats(cond_pil, "LQ input")

    # ============ TEST A ============
    print("\n===== TEST A: dummy CN, cn_scale=0 (与无 ControlNet 等价) =====")
    pipe = build_pipeline(args.pretrained_dir, args.controlnet_dir,
                          use_trained_cn=False, fp32_vae=False)
    img_a, info_a = run(pipe, cond_pil, "A_dummy_scale0", 0.0)
    del pipe
    torch.cuda.empty_cache()

    # ============ TEST B ============
    print("\n===== TEST B: trained CN + bf16 VAE, cn_scale=1 (复现 _log_validation) =====")
    pipe = build_pipeline(args.pretrained_dir, args.controlnet_dir,
                          use_trained_cn=True, fp32_vae=False)
    img_b, info_b = run(pipe, cond_pil, "B_trained_bf16vae_scale1", 1.0)
    del pipe
    torch.cuda.empty_cache()

    # ============ TEST C ============
    print("\n===== TEST C: trained CN + bf16 VAE, cn_scale=0 =====")
    pipe = build_pipeline(args.pretrained_dir, args.controlnet_dir,
                          use_trained_cn=True, fp32_vae=False)
    img_c, info_c = run(pipe, cond_pil, "C_trained_bf16vae_scale0", 0.0)
    del pipe
    torch.cuda.empty_cache()

    # ============ TEST D ============
    print("\n===== TEST D: trained CN + fp32 VAE, cn_scale=1 =====")
    pipe = build_pipeline(args.pretrained_dir, args.controlnet_dir,
                          use_trained_cn=True, fp32_vae=True)
    img_d, info_d = run(pipe, cond_pil, "D_trained_fp32vae_scale1", 1.0)
    del pipe
    torch.cuda.empty_cache()

    print("\n===== 汇总 =====")
    print(f"  A (dummy CN, scale=0):             mean={info_a['mean']:6.2f}  std={info_a['std']:6.2f}  max={info_a['max']:.0f}")
    print(f"  B (trained, bf16 VAE, scale=1):    mean={info_b['mean']:6.2f}  std={info_b['std']:6.2f}  max={info_b['max']:.0f}")
    print(f"  C (trained, bf16 VAE, scale=0):    mean={info_c['mean']:6.2f}  std={info_c['std']:6.2f}  max={info_c['max']:.0f}")
    print(f"  D (trained, fp32 VAE, scale=1):    mean={info_d['mean']:6.2f}  std={info_d['std']:6.2f}  max={info_d['max']:.0f}")
    print(f"\n所有结果: {args.out_dir}")

    print("\n诊断结论判断:")
    if info_a["std"] < 5:
        print("  ⚠ TEST A 全黑 →  base SDXL 本身就有问题 (可能是权重加载/精度/网络问题)")
    elif info_b["std"] < 5 and info_c["std"] > 30:
        print("  ⚠ A 正常, C 正常, B 全黑 →  ControlNet 残差是元凶 (训练崩坏或 conditioning 不匹配)")
    elif info_b["std"] < 5 and info_c["std"] < 5:
        print("  ⚠ A 正常, 但 C 也全黑 →  trained ControlNet 即便 scale=0 也破坏输出 (异常)")
    elif info_b["std"] < 5 and info_d["std"] > 30:
        print("  ⚠ A 正常, B 全黑, D 正常 →  bf16 VAE 是元凶 (换成 fp32)")
    else:
        print("  看起来一切正常, 全黑不是来自这几个常见原因")


if __name__ == "__main__":
    main()