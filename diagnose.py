"""诊断脚本：定位"整体颜色错误"的根因。

直接读取本地文件，不走 HF Hub API。
测试矩阵 (默认 RESOLUTION_INFER=1024, 匹配实际 test.yaml):

  A) dummy CN, bf16 VAE, scale=0           → 验证 base SDXL 本身
  B) trained CN, bf16 VAE, scale=1         → 复现 test.py / _log_validation 现状
  C) trained CN, bf16 VAE, scale=0         → ControlNet 残差是否自身异常
  D) trained CN, fp32 VAE, scale=1         → 验证 bf16 VAE 是否是色偏元凶 (关键)
  E) trained CN, bf16 VAE, scale=0.1       → 缩小残差, 看是否回到 base
  F) trained CN, bf16 VAE, scale=0.5       → 缩小残差
  G) trained CN, bf16 VAE, RES=512         → 排除分辨率不一致 (vs B)
  H) trained CN, fp32 VAE, RES=512         → 训练分辨率 + 正确 VAE 精度 (最理想)
  I) trained CN, bf16 VAE, RES=512, lanczos → 训练时使用的插值方式

可选: --gt_img 给出 GT 图, 脚本会额外算 pred vs GT 的 R/G/B 通道均值差
      (色偏方向: 正值=偏红/绿/蓝, 负值=偏其补色).

用法:
    python diagnose.py \
        --hf_cache D:/Projects/pycharm/hf_cache \
        --controlnet_dir D:/Projects/pycharm/sdxl/experiment/weather_controlnet/checkpoint-2000/controlnet \
        --cond_img D:/.../rain/test/LQ/000003.jpg \
        --gt_img   D:/.../rain/test/GT/000003.jpg \
        --out_dir  D:/Projects/pycharm/sdxl/experiment/diagnose
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from typing import Optional, Tuple, Dict

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path


# ---------------------------------------------------------------------------
# HF 离线环境 (必须在 import diffusers 之前)
# ---------------------------------------------------------------------------
DEFAULT_HF_CACHE = "./hf_cache"
DEFAULT_PRETRAINED_SNAPSHOT = None
DEFAULT_CONTROLNET_DIR = "./experiment/weather_controlnet/controlnet"
DEFAULT_COND_IMG = "./datasets/rain/test/LQ/000003.jpg"
DEFAULT_GT_IMG = None
DEFAULT_OUT_DIR = "./experiment/diagnose"

HF_HOME = os.environ.get("HF_HOME", DEFAULT_HF_CACHE)
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", HF_HOME)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
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


# ---------------------------------------------------------------------------
# 默认超参数 (匹配 test.yaml)
# ---------------------------------------------------------------------------
RESOLUTION_INFER = 512        # test.yaml 实际推理分辨率
RESOLUTION_TRAIN = 512         # train.yaml 训练分辨率
WEIGHT_DTYPE = torch.bfloat16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
NUM_STEPS = 20
GUIDANCE = 5.0
NEG_PROMPT = "dotted, noise, blur, lowres, smooth"


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def find_snapshot_dir(cache_root: str, repo_id: str) -> str:
    """从 HF 缓存找 snapshot 目录."""
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


def stats(img: Image.Image, label: str) -> Dict[str, float]:
    arr = np.asarray(img.convert("RGB")).astype(np.float32)
    info = {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "R": float(arr[..., 0].mean()),
        "G": float(arr[..., 1].mean()),
        "B": float(arr[..., 2].mean()),
    }
    print(
        f"  [{label}] mean={info['mean']:6.2f}  std={info['std']:6.2f}  "
        f"min={info['min']:.0f}  max={info['max']:.0f}  "
        f"R/G/B={info['R']:5.1f}/{info['G']:5.1f}/{info['B']:5.1f}"
    )
    return info


def diff_channels(pred: Image.Image, ref: Image.Image, label: str) -> Dict[str, float]:
    """逐通道算 pred - ref 的均值 (颜色偏移方向) + std (噪声).

    正值: pred 偏该通道色; 负值: 偏其补色.
    """
    if pred.size != ref.size:
        ref = ref.resize(pred.size, Image.BICUBIC)
    A = np.asarray(pred.convert("RGB")).astype(np.float32)
    B = np.asarray(ref.convert("RGB")).astype(np.float32)
    d = A - B
    out = {
        "mean_dR": float(d[..., 0].mean()),
        "mean_dG": float(d[..., 1].mean()),
        "mean_dB": float(d[..., 2].mean()),
        "std":     float(d.std()),
    }
    print(
        f"  [{label}] ΔR={out['mean_dR']:+6.2f}  ΔG={out['mean_dG']:+6.2f}  "
        f"ΔB={out['mean_dB']:+6.2f}  std={out['std']:6.2f}"
    )
    return out


# ---------------------------------------------------------------------------
# Pipeline 构造
# ---------------------------------------------------------------------------
def build_pipeline(
    pretrained_dir: str,
    controlnet_dir: str,
    use_trained_cn: bool,
    fp32_vae: bool,
    weight_dtype: torch.dtype = WEIGHT_DTYPE,
):
    """手动组装管线, 绕过 from_pretrained 的 Hub 验证.

    fp32_vae=True 时 VAE 强制 fp32, 其余仍按 weight_dtype (bf16) 加载.
    这是 diffusers 官方推荐的 SDXL 推理配置, 与 bf16 UNet 完全兼容.
    """
    print(f"\n[load] use_trained_cn={use_trained_cn}, fp32_vae={fp32_vae}, dtype={weight_dtype}")
    print(f"  pretrained_dir = {pretrained_dir}")
    print(f"  controlnet_dir = {controlnet_dir if use_trained_cn else '<dummy>'}")

    vae_dtype = torch.float32 if fp32_vae else weight_dtype

    # 1. VAE
    vae = AutoencoderKL.from_pretrained(
        pretrained_dir, subfolder="vae", torch_dtype=vae_dtype
    )

    # 2. UNet
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_dir, subfolder="unet", torch_dtype=weight_dtype
    )

    # 3. Text encoders
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_dir, subfolder="text_encoder", torch_dtype=weight_dtype
    )
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        pretrained_dir, subfolder="text_encoder_2", torch_dtype=weight_dtype
    )

    # 4. Tokenizers
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_dir, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(pretrained_dir, subfolder="tokenizer_2")

    # 5. Scheduler
    scheduler = DDPMScheduler.from_pretrained(pretrained_dir, subfolder="scheduler")

    # 6. ControlNet
    if use_trained_cn:
        cfg_path = os.path.join(controlnet_dir, "config.json")
        with open(cfg_path) as f:
            cn_cfg = json.load(f)

        weight_safetensors = os.path.join(
            controlnet_dir, "diffusion_pytorch_model.safetensors"
        )
        weight_bin = os.path.join(controlnet_dir, "diffusion_pytorch_model.bin")

        if os.path.exists(weight_safetensors):
            from safetensors.torch import load_file as safe_load
            state_dict = safe_load(weight_safetensors)
        elif os.path.exists(weight_bin):
            state_dict = torch.load(weight_bin, map_location="cpu")
        else:
            raise FileNotFoundError(
                f"No weight file in {controlnet_dir}: "
                f"expected .safetensors or .bin"
            )

        cn = ControlNetModel.from_config(cn_cfg)
        cn.load_state_dict(state_dict, strict=False)
        cn.to(dtype=weight_dtype)
    else:
        cn = ControlNetModel.from_unet(unet)
        cn.to(weight_dtype)

    # 7. 组装
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

    # 关键: bf16 latents + fp32 VAE 时, PyTorch 不会自动 cast, 必须显式 wrap
    if fp32_vae:
        _wrap_vae_decode_for_fp32(pipeline.vae)

    print(
        f"  vae={pipeline.vae.dtype}  unet={pipeline.unet.dtype}  "
        f"cn={pipeline.controlnet.dtype}  te1={pipeline.text_encoder.dtype}  "
        f"te2={pipeline.text_encoder_2.dtype}"
    )
    return pipeline


def _wrap_vae_decode_for_fp32(vae):
    """Wrap vae.decode so that bf16 latents are cast to vae.dtype before decoding.

    训练时 VAE 是 fp32 (train.py:687), 推理时若把 VAE 维持在 fp32 而 UNet 仍 bf16,
    scheduler 输出的 latents 是 bf16, 直接 decode 会触发:
        RuntimeError: Input type (c10::BFloat16) and bias type (float) should be the same
    """
    original_decode = vae.decode
    def _wrapped(latents, *args, **kwargs):
        if latents.dtype != vae.dtype:
            latents = latents.to(vae.dtype)
        return original_decode(latents, *args, **kwargs)
    vae.decode = _wrapped


# ---------------------------------------------------------------------------
# 单次推理
# ---------------------------------------------------------------------------
def run(
    pipeline,
    cond,
    tag: str,
    controlnet_scale: float,
    out_dir: str,
    resolution: int,
    gt_pil: Optional[Image.Image] = None,
):
    os.makedirs(out_dir, exist_ok=True)
    generator = torch.Generator(device=DEVICE).manual_seed(SEED)
    image = pipeline(
        prompt="",
        negative_prompt=NEG_PROMPT,
        image=cond,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE,
        height=resolution,
        width=resolution,
        controlnet_conditioning_scale=controlnet_scale,
        generator=generator,
    ).images[0]
    info = stats(image, f"{tag} (scale={controlnet_scale}, res={resolution})")

    # 与 LQ 差 (颜色偏移方向)
    diff_channels(image, cond, f"{tag} vs LQ")

    # 与 GT 差 (可选)
    if gt_pil is not None:
        # GT 可能尺寸不同, 先 resize 到 pred 尺寸
        diff_channels(image, gt_pil, f"{tag} vs GT")

    out_path = os.path.join(out_dir, f"{tag}.png")
    image.save(out_path)
    print(f"  saved -> {out_path}")
    return image, info


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_cache", default=DEFAULT_HF_CACHE)
    parser.add_argument("--pretrained_dir", default=None,
                        help="SDXL 模型目录, 默认自动从 hf_cache 推断")
    parser.add_argument("--controlnet_dir", default=DEFAULT_CONTROLNET_DIR)
    parser.add_argument("--cond_img", default=DEFAULT_COND_IMG)
    parser.add_argument("--gt_img", default=DEFAULT_GT_IMG,
                        help="可选: GT 图, 启用后额外算 pred vs GT 通道差")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--skip", nargs="*", default=[],
                        help="跳过指定测试, e.g. --skip A C F")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.pretrained_dir is None:
        args.pretrained_dir = find_snapshot_dir(
            args.hf_cache, "stabilityai/stable-diffusion-xl-base-1.0"
        )

    print(f"[setup] device={DEVICE}, dtype={WEIGHT_DTYPE}")
    print(f"[setup] HF cache    = {args.hf_cache}")
    print(f"[setup] pretrained  = {args.pretrained_dir}")
    print(f"[setup] controlnet  = {args.controlnet_dir}")
    print(f"[setup] cond_img    = {args.cond_img}")
    print(f"[setup] gt_img      = {args.gt_img}")
    print(f"[setup] out_dir     = {args.out_dir}")

    for p, name in [
        (args.controlnet_dir, "controlnet_dir"),
        (args.pretrained_dir, "pretrained_dir"),
        (args.cond_img, "cond_img"),
    ]:
        if not os.path.exists(p):
            print(f"\nERROR: {name} 不存在: {p}")
            sys.exit(1)

    # 准备 LQ
    cond_pil = Image.open(args.cond_img).convert("RGB")
    interp = transforms.InterpolationMode.BILINEAR
    cond_pil_1024 = transforms.CenterCrop(RESOLUTION_INFER)(
        transforms.Resize(RESOLUTION_INFER, interpolation=interp)(cond_pil)
    )
    cond_pil_512 = transforms.CenterCrop(RESOLUTION_TRAIN)(
        transforms.Resize(RESOLUTION_TRAIN, interpolation=interp)(cond_pil)
    )
    cond_pil_1024.save(os.path.join(args.out_dir, "cond_lq_1024.png"))
    cond_pil_512.save(os.path.join(args.out_dir, "cond_lq_512.png"))
    stats(cond_pil_1024, f"LQ input @ {RESOLUTION_INFER}")
    stats(cond_pil_512, f"LQ input @ {RESOLUTION_TRAIN}")

    # 准备 GT (可选)
    gt_pil: Optional[Image.Image] = None
    if args.gt_img and os.path.isfile(args.gt_img):
        gt_pil = Image.open(args.gt_img).convert("RGB")
        gt_pil = transforms.CenterCrop(RESOLUTION_INFER)(
            transforms.Resize(RESOLUTION_INFER, interpolation=interp)(gt_pil)
        )
        gt_pil.save(os.path.join(args.out_dir, "gt_ref.png"))
        stats(gt_pil, f"GT @ {RESOLUTION_INFER}")

    # 收集结果, 末尾做汇总
    results: Dict[str, Dict] = {}

    def _do(tag: str, fn):
        if tag in args.skip:
            print(f"\n[skip] {tag}")
            return
        fn()

    # ---------- A: dummy CN, bf16 VAE, scale=0, res=1024 ----------
    def test_a():
        pipe = build_pipeline(args.pretrained_dir, args.controlnet_dir,
                              use_trained_cn=False, fp32_vae=False)
        img, info = run(pipe, cond_pil_1024, "A_dummy_scale0",
                        0.0, args.out_dir, RESOLUTION_INFER, gt_pil)
        results["A"] = info
        del pipe
        torch.cuda.empty_cache()
    _do("A", test_a)

    # ---------- B: trained CN, bf16 VAE, scale=1, res=1024 (复现 test.py) ----------
    def test_b():
        pipe = build_pipeline(args.pretrained_dir, args.controlnet_dir,
                              use_trained_cn=True, fp32_vae=False)
        img, info = run(pipe, cond_pil_1024, "B_trained_bf16vae_scale1",
                        1.0, args.out_dir, RESOLUTION_INFER, gt_pil)
        results["B"] = info
        del pipe
        torch.cuda.empty_cache()
    _do("B", test_b)

    # ---------- C: trained CN, bf16 VAE, scale=0 ----------
    def test_c():
        pipe = build_pipeline(args.pretrained_dir, args.controlnet_dir,
                              use_trained_cn=True, fp32_vae=False)
        img, info = run(pipe, cond_pil_1024, "C_trained_bf16vae_scale0",
                        0.0, args.out_dir, RESOLUTION_INFER, gt_pil)
        results["C"] = info
        del pipe
        torch.cuda.empty_cache()
    _do("C", test_c)

    # ---------- D: trained CN, fp32 VAE, scale=1 ⭐ 关键对照 ----------
    def test_d():
        pipe = build_pipeline(args.pretrained_dir, args.controlnet_dir,
                              use_trained_cn=True, fp32_vae=True)
        img, info = run(pipe, cond_pil_1024, "D_trained_fp32vae_scale1",
                        1.0, args.out_dir, RESOLUTION_INFER, gt_pil)
        results["D"] = info
        del pipe
        torch.cuda.empty_cache()
    _do("D", test_d)

    # ---------- E: bf16 VAE, scale=0.1 ----------
    def test_e():
        pipe = build_pipeline(args.pretrained_dir, args.controlnet_dir,
                              use_trained_cn=True, fp32_vae=False)
        run(pipe, cond_pil_1024, "E_trained_scale0.1",
            0.1, args.out_dir, RESOLUTION_INFER, gt_pil)
        del pipe
        torch.cuda.empty_cache()
    _do("E", test_e)

    # ---------- F: bf16 VAE, scale=0.5 ----------
    def test_f():
        pipe = build_pipeline(args.pretrained_dir, args.controlnet_dir,
                              use_trained_cn=True, fp32_vae=False)
        run(pipe, cond_pil_1024, "F_trained_scale0.5",
            0.5, args.out_dir, RESOLUTION_INFER, gt_pil)
        del pipe
        torch.cuda.empty_cache()
    _do("F", test_f)

    # ---------- G: bf16 VAE, res=512 (匹配训练分辨率) ----------
    def test_g():
        pipe = build_pipeline(args.pretrained_dir, args.controlnet_dir,
                              use_trained_cn=True, fp32_vae=False)
        run(pipe, cond_pil_512, "G_trained_bf16vae_res512",
            1.0, args.out_dir, RESOLUTION_TRAIN)
        del pipe
        torch.cuda.empty_cache()
    _do("G", test_g)

    # ---------- H: fp32 VAE, res=512 ⭐ 训练分布 + 正确 VAE 精度 ----------
    def test_h():
        pipe = build_pipeline(args.pretrained_dir, args.controlnet_dir,
                              use_trained_cn=True, fp32_vae=True)
        run(pipe, cond_pil_512, "H_trained_fp32vae_res512",
            1.0, args.out_dir, RESOLUTION_TRAIN)
        del pipe
        torch.cuda.empty_cache()
    _do("H", test_h)

    # ---------- I: bf16 VAE, res=512, lanczos (训练时使用的插值) ----------
    def test_i():
        cond_lanczos = transforms.CenterCrop(RESOLUTION_TRAIN)(
            transforms.Resize(RESOLUTION_TRAIN,
                              interpolation=transforms.InterpolationMode.LANCZOS)(cond_pil)
        )
        cond_lanczos.save(os.path.join(args.out_dir, "cond_lq_512_lanczos.png"))
        pipe = build_pipeline(args.pretrained_dir, args.controlnet_dir,
                              use_trained_cn=True, fp32_vae=False)
        run(pipe, cond_lanczos, "I_trained_bf16vae_res512_lanczos",
            1.0, args.out_dir, RESOLUTION_TRAIN)
        del pipe
        torch.cuda.empty_cache()
    _do("I", test_i)

    # ============ 汇总 ============
    print("\n" + "=" * 60)
    print("汇总 (std 高=有结构, std 低=几乎纯灰/全黑)")
    if results:
        for tag, info in results.items():
            print(
                f"  {tag}: mean={info['mean']:6.2f}  std={info['std']:6.2f}  "
                f"R/G/B={info['R']:5.1f}/{info['G']:5.1f}/{info['B']:5.1f}"
            )

    print("\n诊断结论判断:")
    print("  - 若 D (fp32 VAE) 比 B (bf16 VAE) 颜色更接近 GT / LQ → 锁定 VAE 精度为色偏元凶")
    print("    修复: pipeline.vae.to(torch.float32)  (test.py: _build_pipeline 末尾)")
    print("  - 若 G (res=512) 比 B (res=1024) 颜色更接近 GT → 训练/推理分辨率不一致也是诱因")
    print("    修复: test.yaml resolution 改为 512, 或重训到 1024")
    print("  - 若 I (lanczos) 比 G (bilinear) 颜色更接近 GT → 训练/推理插值不一致也是诱因")
    print("    修复: schemas.TestConfig 加 image_interpolation_mode 字段, 默认 lanczos")
    print("  - 若 E/F (scale 缩小) 仍接近纯灰 → 训练残差过大, 需调小 learning_rate / 加 L1")
    print(f"\n所有结果图: {args.out_dir}")


if __name__ == "__main__":
    main()
