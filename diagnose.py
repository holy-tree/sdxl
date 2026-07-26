"""诊断脚本：精确定位"全黑图像"的根因。

直接读取本地文件，不走 HF Hub API。
四个测试:
1) base SDXL + dummy CN, cn_scale=0   → 验证 base SDXL 是否本身有问题
2) trained CN + bf16 VAE, cn_scale=1   → 复现训练时 _log_validation 行为
3) trained CN + bf16 VAE, cn_scale=0   → 验证 trained CN 即使不加也全黑
4) trained CN + fp32 VAE, cn_scale=1   → 验证 bf16 VAE 是否是元凶

用法:
    python diagnose.py \
        --hf_cache /root/autodl-tmp/hf_cache \
        --controlnet_dir /root/autodl-tmp/experiment/weather_controlnet/checkpoint-2000/controlnet \
        --cond_img /root/autodl-tmp/datasets/rain/test/LQ/000003.jpg \
        --out_dir /root/autodl-tmp/experiment/diagnose
"""

from __future__ import annotations

import os
import sys
import json
import argparse

DEFAULT_HF_CACHE = "/root/autodl-tmp/hf_cache"
DEFAULT_PRETRAINED_SNAPSHOT = None   # 自动从 hf_cache 推断
DEFAULT_CONTROLNET_DIR = "/root/autodl-tmp/experiment/weather_controlnet/checkpoint-2000/controlnet"
DEFAULT_COND_IMG = "/root/autodl-tmp/datasets/rain/test/LQ/000003.jpg"
DEFAULT_OUT_DIR = "/root/autodl-tmp/experiment/diagnose"

import numpy as np
import torch
from PIL import Image
from pathlib import Path
from torchvision import transforms
import copy

# 先设置HF环境变量（在import diffusers之前）
HF_HOME = os.environ.get("HF_HOME", DEFAULT_HF_CACHE)
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "")
# 强制离线：不走任何网络
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# 清除HF_ENDPOINT避免路径被当成repo_id去验证
if "HF_ENDPOINT" in os.environ:
    del os.environ["HF_ENDPOINT"]

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


def find_snapshot_dir(cache_root: str, repo_id: str) -> str:
    """从HF缓存找 snapshot 目录."""
    repo_dir = os.path.join(cache_root, "models--" + repo_id.replace("/", "--"))
    snapshots = os.path.join(repo_dir, "snapshots")
    if not os.path.isdir(snapshots):
        raise FileNotFoundError(f"找不到 snapshots 目录: {snapshots}")
    for name in sorted(os.listdir(snapshots)):
        full = os.path.join(snapshots, name)
        if os.path.isdir(full):
            return full
    raise FileNotFoundError(f"{snapshots} 下没有 snapshot 子目录")


def load_config_json(path: str) -> dict:
    with open(os.path.join(path, "config.json")) as f:
        return json.load(f)


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


def build_pipeline(pretrained_dir: str, controlnet_dir: str, use_trained_cn: bool, fp32_vae: bool):
    """手动组装管线，绕过 from_pretrained 的 Hub 验证."""
    print(f"\n[load] use_trained_cn={use_trained_cn}, fp32_vae={fp32_vae}")
    print(f"  pretrained_dir = {pretrained_dir}")
    print(f"  controlnet_dir = {controlnet_dir if use_trained_cn else '<dummy>'}")

    vae_dtype = torch.float32 if fp32_vae else WEIGHT_DTYPE

    # 1. VAE
    vae_cfg = load_config_json(os.path.join(pretrained_dir, "vae"))
    vae = AutoencoderKL.from_pretrained(pretrained_dir, subfolder="vae", torch_dtype=vae_dtype)

    # 2. UNet
    unet = UNet2DConditionModel.from_pretrained(pretrained_dir, subfolder="unet", torch_dtype=WEIGHT_DTYPE)

    # 3. Text encoders
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_dir, subfolder="text_encoder", torch_dtype=WEIGHT_DTYPE
    )
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        pretrained_dir, subfolder="text_encoder_2", torch_dtype=WEIGHT_DTYPE
    )

    # 4. Tokenizers
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_dir, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(pretrained_dir, subfolder="tokenizer_2")

    # 5. Scheduler
    scheduler = DDPMScheduler.from_pretrained(pretrained_dir, subfolder="scheduler")

    # 6. ControlNet - 手动加载，不走 Hub 验证
    if use_trained_cn:
        import json
        cfg_path = os.path.join(controlnet_dir, "config.json")
        with open(cfg_path) as f:
            cn_cfg = json.load(f)

        weight_safetensors = os.path.join(controlnet_dir, "diffusion_pytorch_model.safetensors")
        weight_bin = os.path.join(controlnet_dir, "diffusion_pytorch_model.bin")

        if os.path.exists(weight_safetensors):
            from safetensors.torch import load_file as safe_load
            state_dict = safe_load(weight_safetensors)
        elif os.path.exists(weight_bin):
            state_dict = torch.load(weight_bin, map_location="cpu")
        else:
            raise FileNotFoundError(
                f"No weight file found in {controlnet_dir}: "
                f"expected diffusion_pytorch_model.bin or diffusion_pytorch_model.safetensors"
            )

        # 直接用字典创建模型，再加载权重
        cn = ControlNetModel.from_config(cn_cfg)
        cn.load_state_dict(state_dict, strict=False)
        cn.to(dtype=WEIGHT_DTYPE)
    else:
        # 从 UNet 初始化 (零卷积)
        cn = ControlNetModel.from_unet(unet)
        cn.to(WEIGHT_DTYPE)

    # 7. 组装 pipeline
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


def run(pipeline, cond, tag, controlnet_scale, out_dir):
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
    out_path = os.path.join(out_dir, f"{tag}.png")
    image.save(out_path)
    print(f"  saved -> {out_path}")
    return image, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_cache", default=DEFAULT_HF_CACHE)
    parser.add_argument("--pretrained_dir", default=None,
                        help="SDXL模型目录，默认自动从hf_cache推断")
    parser.add_argument("--controlnet_dir", default=DEFAULT_CONTROLNET_DIR)
    parser.add_argument("--cond_img", default=DEFAULT_COND_IMG)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 自动找SDXL snapshot目录
    if args.pretrained_dir is None:
        args.pretrained_dir = find_snapshot_dir(args.hf_cache, "stabilityai/stable-diffusion-xl-base-1.0")

    print(f"[setup] device={DEVICE}, dtype={WEIGHT_DTYPE}, resolution={RESOLUTION}")
    print(f"[setup] HF cache    = {args.hf_cache}")
    print(f"[setup] pretrained  = {args.pretrained_dir}")
    print(f"[setup] controlnet  = {args.controlnet_dir}")
    print(f"[setup] cond_img    = {args.cond_img}")
    print(f"[setup] out_dir     = {args.out_dir}")

    # 确认controlnet目录存在
    if not os.path.isdir(args.controlnet_dir):
        print(f"\nERROR: controlnet_dir 不存在: {args.controlnet_dir}")
        print("请确认以下目录结构:")
        parent = os.path.dirname(args.controlnet_dir)
        print(f"  ls {parent}")
        sys.exit(1)

    # 确认pretrained目录存在
    if not os.path.isdir(args.pretrained_dir):
        print(f"\nERROR: pretrained_dir 不存在: {args.pretrained_dir}")
        sys.exit(1)

    # 确认测试图像存在
    if not os.path.isfile(args.cond_img):
        print(f"\nERROR: cond_img 不存在: {args.cond_img}")
        sys.exit(1)

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
    img_a, info_a = run(pipe, cond_pil, "A_dummy_scale0", 0.0, args.out_dir)
    del pipe
    torch.cuda.empty_cache()

    # ============ TEST B ============
    print("\n===== TEST B: trained CN + bf16 VAE, cn_scale=1 (复现 _log_validation) =====")
    pipe = build_pipeline(args.pretrained_dir, args.controlnet_dir,
                          use_trained_cn=True, fp32_vae=False)
    img_b, info_b = run(pipe, cond_pil, "B_trained_bf16vae_scale1", 1.0, args.out_dir)
    del pipe
    torch.cuda.empty_cache()

    # ============ TEST C ============
    print("\n===== TEST C: trained CN + bf16 VAE, cn_scale=0 =====")
    pipe = build_pipeline(args.pretrained_dir, args.controlnet_dir,
                          use_trained_cn=True, fp32_vae=False)
    img_c, info_c = run(pipe, cond_pil, "C_trained_bf16vae_scale0", 0.0, args.out_dir)
    del pipe
    torch.cuda.empty_cache()

    # ============ TEST D ============
    print("\n===== TEST D: trained CN + fp32 VAE, cn_scale=1 =====")
    pipe = build_pipeline(args.pretrained_dir, args.controlnet_dir,
                          use_trained_cn=True, fp32_vae=True)
    img_d, info_d = run(pipe, cond_pil, "D_trained_fp32vae_scale1", 1.0, args.out_dir)
    del pipe
    torch.cuda.empty_cache()

    print("\n===== 汇总 =====")
    print(f"  A (dummy CN, scale=0):             mean={info_a['mean']:6.2f}  std={info_a['std']:6.2f}  max={info_a['max']:.0f}")
    print(f"  B (trained, bf16 VAE, scale=1):    mean={info_b['mean']:6.2f}  std={info_b['std']:6.2f}  max={info_b['max']:.0f}")
    print(f"  C (trained, bf16 VAE, scale=0):    mean={info_c['mean']:6.2f}  std={info_c['std']:6.2f}  max={info_c['max']:.0f}")
    print(f"  D (trained, fp32 VAE, scale=1):    mean={info_d['mean']:6.2f}  std={info_d['std']:6.2f}  max={info_d['max']:.0f}")
    print(f"\n所有结果: {args.out_dir}")

    print("\n诊断结论判断:")
    all_normal = all(info["std"] > 30 for info in [info_a, info_b, info_c, info_d])
    all_black = all(info["std"] < 5 for info in [info_a, info_b, info_c, info_d])

    if all_black:
        print("  ⚠ 全部测试都全黑 → 管线配置有严重问题（权重加载失败/dtype错误等）")
    elif all_normal:
        print("  ✓ 全部测试都正常 → 不是 ControlNet 问题，可能是保存/读取流程有误")
    elif info_a["std"] < 5:
        print("  ⚠ TEST A 全黑 →  base SDXL 本身有问题 (权重/精度/dtype)")
    elif info_b["std"] < 5 and info_c["std"] > 30:
        print("  ⚠ A 正常, C 正常, B 全黑 →  ControlNet 残差是元凶 (训练崩坏或 conditioning 不匹配)")
    elif info_b["std"] < 5 and info_c["std"] < 5:
        print("  ⚠ A 正常, C 也全黑 →  trained CN 即使 scale=0 也破坏输出 (异常)")
    elif info_b["std"] < 5 and info_d["std"] > 30:
        print("  ⚠ A 正常, B 全黑, D 正常 →  bf16 VAE 是元凶")
    else:
        print("  混合结果，请贴给开发者分析")


if __name__ == "__main__":
    main()
