"""Weather-restoration paired dataset (LQ/GT) for SDXL ControlNet training.

Expected directory layout (auto-discovered under ``dataset_root``)::

    dataset_root/
        rain/{train,test}/{GT,LQ}/<image>.png
        snow/{train,test}/{GT,LQ}/<image>.png
        haze/{train,test}/{GT,LQ}/<image>.png
        ...

The dataset is intentionally framework-agnostic (a plain ``torch.utils.data.Dataset``)
so it can be paired with arbitrary training loops.  Conditioning options include
raw LQ, grayscale, and Canny edge maps.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils import data as torch_data
from torchvision import transforms

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


DEFAULT_WEATHER_PROMPTS: Dict[str, str] = {
    "rain": "rainy scene, rain streaks on the image, wet surfaces, overcast sky",
    "snow": "snowy scene, snowflakes covering the image, cold atmosphere, white noise",
    "haze": "hazy scene, foggy atmosphere, low visibility, grayish tone",
    "fog": "foggy scene, low visibility, soft white veil over the landscape",
    "frost": "frosted scene, icy texture, frozen edges, cold blue tone",
    "night": "nighttime scene, low light, dark sky, artificial illumination",
}

VALID_CONDITIONING_TYPES = ("lq", "gray", "canny")


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTENSIONS


def _to_pil(arr: np.ndarray) -> Image.Image:
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return Image.fromarray(arr.astype(np.uint8))


def canny_edge(image_pil: Image.Image, low: int = 100, high: int = 200) -> Image.Image:
    if cv2 is None:
        gray = image_pil.convert("L")
        return gray.convert("RGB")
    arr = np.asarray(image_pil.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low, high)
    return _to_pil(edges)


def make_conditioning(image_pil: Image.Image, kind: str) -> Image.Image:
    kind = (kind or "lq").lower()
    if kind == "lq":
        return image_pil.convert("RGB")
    if kind == "gray":
        return image_pil.convert("L").convert("RGB")
    if kind == "canny":
        return canny_edge(image_pil)
    raise ValueError(f"Unknown conditioning type '{kind}'. Valid: {VALID_CONDITIONING_TYPES}")


@dataclass
class _ResolvedSample:
    gt_path: Path
    lq_path: Path
    weather: str


class PairedWeatherDataset(torch_data.Dataset):
    """Read paired (LQ, GT) images for one or more weather degradations.

    Args:
        dataset_root: Root directory containing ``{weather}/{split}/{GT,LQ}/`` folders.
        weather_types: Weather sub-folders to include. ``None`` discovers all.
        splits: Which sub-splits to include (``train``, ``val``, ...). ``None`` -> ``["train"]``.
        resolution: Square resolution used for resize (short-side resize + center crop).
        conditioning_type: One of ``lq`` / ``gray`` / ``canny``.
        null_text_ratio: Probability of replacing the prompt with an empty string (CFG training).
        use_prompt: When ``False`` every prompt becomes an empty string.
        prompt_ratio: Probability of using the per-weather prompt (vs empty) when ``use_prompt=True``.
        weather_prompts: Optional override mapping weather -> prompt string.
        weather_num_samples: Optional cap per weather type, e.g. ``{"rain": 100}``.
        interpolation: torchvision interpolation string (default ``bilinear``).
        augment: When ``True`` apply random horizontal flip and 90-degree rotations.
    """

    def __init__(
        self,
        dataset_root: str | os.PathLike,
        weather_types: Optional[Sequence[str]] = None,
        splits: Optional[Sequence[str]] = None,
        resolution: int = 1024,
        conditioning_type: str = "lq",
        null_text_ratio: float = 0.5,
        use_prompt: bool = False,
        prompt_ratio: float = 0.2,
        weather_prompts: Optional[Dict[str, str]] = None,
        weather_num_samples: Optional[Dict[str, int]] = None,
        interpolation: str = "bilinear",
        augment: bool = True,
        preload: bool = False,
        cache_dir: Optional[str] = None,
        cache_format: str = "jpg",
        cache_jpeg_quality: int = 95,
        force_rebuild_cache: bool = False,
        cache_num_workers: int = 0,
    ) -> None:
        super().__init__()

        self.dataset_root = Path(dataset_root)
        self.resolution = int(resolution)
        self.conditioning_type = conditioning_type.lower()
        if self.conditioning_type not in VALID_CONDITIONING_TYPES:
            raise ValueError(
                f"conditioning_type must be one of {VALID_CONDITIONING_TYPES}, got {self.conditioning_type}"
            )
        self.null_text_ratio = float(null_text_ratio)
        self.use_prompt = bool(use_prompt)
        self.prompt_ratio = max(0.0, min(1.0, float(prompt_ratio)))
        self.weather_num_samples = dict(weather_num_samples or {})

        self.weather_types = list(weather_types) if weather_types else None
        self.splits = list(splits) if splits else ["train"]

        self.weather_prompts: Dict[str, str] = dict(DEFAULT_WEATHER_PROMPTS)
        if weather_prompts:
            self.weather_prompts.update(weather_prompts)

        self.samples: List[_ResolvedSample] = self._discover()

        try:
            self._interp = getattr(transforms.InterpolationMode, interpolation.upper())
        except (AttributeError, KeyError) as exc:
            raise ValueError(f"Unsupported interpolation '{interpolation}'") from exc

        self.preprocess = transforms.Compose(
            [
                transforms.Resize(self.resolution, interpolation=self._interp),
                transforms.CenterCrop(self.resolution),
            ]
        )
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        self.augment = bool(augment)
        self._cache: Optional[List[Tuple[Image.Image, Image.Image]]] = None
        self._tensor_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None

        self.cache_dir: Optional[Path] = Path(cache_dir) if cache_dir else None
        self.cache_format = cache_format
        self.cache_jpeg_quality = int(cache_jpeg_quality)
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            tag = f"res{self.resolution}_{self.conditioning_type}_{self._interp.name.lower()}"
            self.cache_subdir = self.cache_dir / tag
            self.cache_subdir.mkdir(parents=True, exist_ok=True)
            if force_rebuild_cache or not self._cache_complete():
                self.build_cache(num_workers=cache_num_workers, force=force_rebuild_cache)
            else:
                print(f"[数据集] 磁盘缓存已存在: {self.cache_subdir} (跳过重建)")

        if preload:
            self._preload_to_memory()

    def _preload_to_memory(self) -> None:
        if self._tensor_cache is not None:
            return
        if self.cache_dir is not None and self._cache_complete():
            print(f"[数据集] preload: 从磁盘缓存加载 {len(self.samples)} 张 ...")
            cache: List[Tuple[torch.Tensor, torch.Tensor]] = []
            for i in range(len(self.samples)):
                cache.append(self._load_cached(i))
            self._tensor_cache = cache
            return
        print(f"[数据集] preload: 预解码 {len(self.samples)} 张图像到内存 ...")
        cache_list: List[Tuple[Image.Image, Image.Image]] = []
        for i, sample in enumerate(self.samples):
            gt = Image.open(sample.gt_path).convert("RGB")
            gt.load()
            lq = Image.open(sample.lq_path).convert("RGB")
            lq.load()
            cache_list.append((gt, lq))
        self._cache = cache_list
        print(f"[数据集] preload 完成")

    def _discover(self) -> List[_ResolvedSample]:
        if not self.dataset_root.is_dir():
            raise FileNotFoundError(f"dataset_root does not exist: {self.dataset_root}")

        if self.weather_types is None:
            weather_dirs = [p for p in self.dataset_root.iterdir() if p.is_dir()]
            weather_dirs = [
                d
                for d in weather_dirs
                if any((d / split / sub).is_dir() for split in self.splits for sub in ("GT", "LQ"))
            ]
            self.weather_types = sorted(d.name for d in weather_dirs)

        if not self.weather_types:
            raise FileNotFoundError(
                f"No weather sub-folders discovered under {self.dataset_root}. "
                f"Expected structure: {self.dataset_root}/<weather>/<split>/{{GT,LQ}}/"
            )

        samples: List[_ResolvedSample] = []
        for weather in self.weather_types:
            for split in self.splits:
                gt_dir = self.dataset_root / weather / split / "GT"
                lq_dir = self.dataset_root / weather / split / "LQ"
                if not gt_dir.is_dir() or not lq_dir.is_dir():
                    print(f"[数据集] 跳过缺失目录: {gt_dir} 或 {lq_dir}")
                    continue
                gt_map = {p.stem: p for p in gt_dir.iterdir() if p.is_file() and _is_image(p)}
                lq_map = {p.stem: p for p in lq_dir.iterdir() if p.is_file() and _is_image(p)}
                matched = 0
                for stem in sorted(gt_map.keys() & lq_map.keys()):
                    samples.append(_ResolvedSample(gt_map[stem], lq_map[stem], weather))
                    matched += 1
                print(f"[数据集] {weather}/{split}: 匹配 {matched} 对")

        if not samples:
            raise FileNotFoundError(
                f"未在 {self.dataset_root} 下找到任何匹配的 (GT, LQ) 图像对。"
                f"请确认目录结构: {{weather}}/{{split}}/{{GT,LQ}}/"
            )

        if self.weather_num_samples:
            grouped: Dict[str, List[_ResolvedSample]] = {w: [] for w in self.weather_types}
            for s in samples:
                grouped.setdefault(s.weather, []).append(s)
            capped: List[_ResolvedSample] = []
            for weather in self.weather_types:
                bucket = grouped.get(weather, [])
                limit = int(self.weather_num_samples.get(weather, -1))
                if limit > 0 and limit < len(bucket):
                    print(f"[数据集] {weather}: 截断 {len(bucket)} -> {limit} 样本")
                    capped.extend(bucket[:limit])
                else:
                    capped.extend(bucket)
            samples = capped

        print(f"[数据集] 最终训练样本数: {len(samples)}")
        for w in self.weather_types:
            cnt = sum(1 for s in samples if s.weather == w)
            limit = int(self.weather_num_samples.get(w, -1)) if self.weather_num_samples else -1
            limit_str = f"/{limit}" if limit > 0 else ""
            print(f"  - {w}: {cnt}{limit_str}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _make_prompt(self, weather: str) -> str:
        if not self.use_prompt:
            return ""
        if random.random() >= self.prompt_ratio:
            return ""
        return self.weather_prompts.get(weather, "")

    def _augment(self, *imgs: Image.Image) -> List[Image.Image]:
        if not self.augment:
            return list(imgs)
        if random.random() < 0.5:
            imgs = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in imgs]
        k = random.randint(0, 3)
        if k:
            imgs = [img.rotate(90 * k, expand=True) for img in imgs]
        return imgs

    @staticmethod
    def _augment_tensor(gt: torch.Tensor, cond: torch.Tensor, augment: bool):
        if not augment:
            return gt, cond
        if random.random() < 0.5:
            gt = torch.flip(gt, dims=(-1,))
            cond = torch.flip(cond, dims=(-1,))
        k = random.randint(0, 3)
        if k:
            gt = torch.rot90(gt, k=k, dims=(-2, -1))
            cond = torch.rot90(cond, k=k, dims=(-2, -1))
        return gt, cond

    def _cache_paths(self, index: int) -> Tuple[Path, Path, Optional[Path]]:
        sample = self.samples[index]
        stem = f"{sample.weather}__{sample.gt_path.stem}"
        ext = self.cache_format
        gt_p = self.cache_subdir / f"{stem}__gt.{ext}"
        lq_p = self.cache_subdir / f"{stem}__lq.{ext}"
        cond_p = self.cache_subdir / f"{stem}__cond.{ext}" if self.conditioning_type != "lq" else None
        return gt_p, lq_p, cond_p

    def _cache_complete(self) -> bool:
        if self.cache_dir is None:
            return False
        for i in range(len(self.samples)):
            gt_p, lq_p, cond_p = self._cache_paths(i)
            if not gt_p.is_file() or not lq_p.is_file():
                return False
            if cond_p is not None and not cond_p.is_file():
                return False
        return True

    def _load_cached(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        gt_p, lq_p, cond_p = self._cache_paths(index)
        gt = self._load_image_tensor(gt_p)
        if cond_p is not None:
            cond = self._load_image_tensor(cond_p)
        else:
            cond = self._load_image_tensor(lq_p)
        return gt, cond

    def _load_image_tensor(self, path: Path) -> torch.Tensor:
        if self.cache_format == "pt":
            return torch.load(path, map_location="cpu")
        with Image.open(path) as img:
            img.load()
        arr = np.array(img)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return tensor

    def _save_image_tensor(self, path: Path, tensor_u8: torch.Tensor) -> int:
        """Save a uint8 [3,H,W] tensor to disk. Returns file size in bytes."""
        if self.cache_format == "pt":
            torch.save(tensor_u8.contiguous(), path)
            return path.stat().st_size
        arr = tensor_u8.permute(1, 2, 0).cpu().numpy()
        img = Image.fromarray(arr, mode="RGB")
        if self.cache_format in ("jpg", "jpeg"):
            img.save(path, format="JPEG", quality=self.cache_jpeg_quality, optimize=False)
        else:
            img.save(path, format="PNG", optimize=False)
        return path.stat().st_size

    def _save_cached(
        self,
        index: int,
        gt_tensor_u8: torch.Tensor,
        cond_tensor_u8: torch.Tensor,
        save_cond_separately: bool,
    ) -> Tuple[int, int, int]:
        gt_p, lq_p, cond_p = self._cache_paths(index)
        gt_size = self._save_image_tensor(gt_p, gt_tensor_u8)
        if save_cond_separately:
            cond_size = self._save_image_tensor(cond_p, cond_tensor_u8)
            lq_size = cond_size
        else:
            cond_size = self._save_image_tensor(lq_p, cond_tensor_u8)
            lq_size = cond_size
        return gt_size, lq_size, cond_size

    def _process_one_to_tensor(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        t0 = time.perf_counter()
        with Image.open(sample.gt_path) as gt_img_raw:
            gt_img = gt_img_raw.convert("RGB")
            gt_img.load()
        with Image.open(sample.lq_path) as lq_img_raw:
            lq_img = lq_img_raw.convert("RGB")
            lq_img.load()
        t_decode = time.perf_counter() - t0

        t0 = time.perf_counter()
        cond_pil = make_conditioning(lq_img, self.conditioning_type)
        t_cond = time.perf_counter() - t0

        t0 = time.perf_counter()
        gt_img = self.preprocess(gt_img)
        cond_pil = self.preprocess(cond_pil)
        t_resize = time.perf_counter() - t0

        gt_t = (self.to_tensor(gt_img) * 255).to(torch.uint8)
        cond_t = (self.to_tensor(cond_pil) * 255).to(torch.uint8)
        return gt_t, cond_t, t_decode, t_cond, t_resize

    def build_cache(self, num_workers: int = 0, force: bool = False) -> None:
        if self.cache_dir is None:
            raise RuntimeError("cache_dir is not set")
        if self.cache_format not in ("pt", "jpg", "jpeg", "png"):
            raise ValueError(f"cache_format must be pt|jpg|jpeg|png, got {self.cache_format!r}")
        self.cache_subdir.mkdir(parents=True, exist_ok=True)
        save_cond = self.conditioning_type != "lq"
        n = len(self.samples)
        if force:
            for i in range(n):
                gt_p, lq_p, cond_p = self._cache_paths(i)
                for p in (gt_p, lq_p, cond_p):
                    if p is not None and p.is_file():
                        p.unlink()
        t_total = time.perf_counter()
        if num_workers and num_workers > 0:
            from multiprocessing import get_context

            ctx = get_context("spawn")
            args_list = [(i,) for i in range(n)]
            with ctx.Pool(processes=num_workers) as pool:
                results = pool.starmap(self._process_one_to_tensor, args_list)
        else:
            results = [self._process_one_to_tensor(i) for i in range(n)]
        decode_t = sum(r[2] for r in results)
        cond_t = sum(r[3] for r in results)
        resize_t = sum(r[4] for r in results)
        total_bytes = 0
        for i, (gt_t, cond_t_t, *_rest) in enumerate(results):
            g, l, _c = self._save_cached(i, gt_t.to(torch.uint8), cond_t_t.to(torch.uint8), save_cond)
            total_bytes += g + l
        elapsed = time.perf_counter() - t_total
        gb = total_bytes / 1024 ** 3
        per_sample = total_bytes / max(n, 1) / 1024 ** 2
        print(
            f"[数据集] 磁盘缓存构建完成: {self.cache_subdir}\n"
            f"  samples={n}  total={elapsed:.1f}s  "
            f"avg/sample={elapsed/n*1000:.1f}ms\n"
            f"  decode={decode_t:.1f}s  conditioning={cond_t:.1f}s  resize={resize_t:.1f}s\n"
            f"  format={self.cache_format}  total={gb:.2f} GB  "
            f"per_sample={per_sample:.2f} MB"
        )

    def __getitem__(self, index: int) -> Dict[str, object]:
        if self._tensor_cache is not None:
            gt_u8, cond_u8 = self._tensor_cache[index]
        elif self.cache_dir is not None and self._cache_complete():
            gt_u8, cond_u8 = self._load_cached(index)
        elif self._cache is not None:
            gt_img, lq_img = self._cache[index]
            cond_pil = make_conditioning(lq_img, self.conditioning_type)
            gt_img = self.preprocess(gt_img)
            cond_pil = self.preprocess(cond_pil)
            gt_img, cond_pil = self._augment(gt_img, cond_pil)
            gt_u8 = (self.to_tensor(gt_img) * 255).to(torch.uint8)
            cond_u8 = (self.to_tensor(cond_pil) * 255).to(torch.uint8)
        else:
            gt_u8, cond_u8, *_ = self._process_one_to_tensor(index)

        gt_u8, cond_u8 = self._augment_tensor(gt_u8, cond_u8, self.augment)
        gt_tensor = gt_u8.float().div_(255.0)
        cond_tensor = cond_u8.float().div_(255.0)
        gt_tensor = self.normalize(gt_tensor)

        sample = self.samples[index]
        prompt = self._make_prompt(sample.weather)
        if random.random() < self.null_text_ratio:
            prompt = ""

        return {
            "pixel_values": gt_tensor,
            "conditioning_pixel_values": cond_tensor,
            "prompt": prompt,
            "weather": sample.weather,
            "index": index,
        }


def encode_weather_prompts(
    dataset: PairedWeatherDataset,
    text_encoders: Sequence[torch.nn.Module],
    tokenizers: Sequence,
    device: torch.device,
    original_size: Tuple[int, int],
    crops_coords_top_left: Tuple[int, int],
    target_size: Tuple[int, int],
    proportion_empty_prompts: float = 0.0,
    batch_size: int = 8,
    skip_prompt_extraction: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pre-compute SDXL text embeddings for the full dataset.

    Args:
        skip_prompt_extraction: When ``True``, skip the per-sample ``dataset[i]`` walk
            (which triggers full PIL decode + resize + center-crop just to read the
            ``prompt`` field). Useful when every prompt is known to be empty
            (``use_prompt=False`` and ``proportion_empty_prompts=0``), saving minutes
            of startup time on medium/large datasets.

    Returns:
        prompt_embeds: ``[N, 77, 2048]``
        text_embeds: ``[N, 1280]`` (pooled, from text_encoder_2)
        time_ids: ``[N, 6]`` SDXL add-time-ids
    """
    if len(text_encoders) != 2 or len(tokenizers) != 2:
        raise ValueError("SDXL requires exactly 2 text encoders and 2 tokenizers")

    add_time_ids = torch.tensor(
        list(original_size + crops_coords_top_left + target_size), dtype=torch.float32
    )

    if skip_prompt_extraction:
        prompt_list = [""] * len(dataset)
        print(f"[数据集] 跳过 prompt 提取, 直接使用 {len(dataset)} 个空 prompt (use_prompt=False)")
    else:
        # Fast path: 不走 __getitem__ (会触发 PIL decode + resize + center_crop,
        # 单线程遍历 30k 样本约 15 分钟), 直接从 dataset.samples 读取 weather 字段.
        # _make_prompt 调用顺序与原代码一致, RNG 序列保持不变, prompt_list 字节级一致.
        prompt_list = [dataset._make_prompt(s.weather) for s in dataset.samples]

    if proportion_empty_prompts > 0:
        prompt_list = ["" if random.random() < proportion_empty_prompts else p for p in prompt_list]

    text_encoders = [t.to(device) for t in text_encoders]
    for t in text_encoders:
        t.eval()

    all_prompt_embeds: List[torch.Tensor] = []
    all_text_embeds: List[torch.Tensor] = []

    with torch.no_grad():
        for start in range(0, len(prompt_list), batch_size):
            chunk = prompt_list[start : start + batch_size]
            embeds_list = []
            pooled_list = []
            for tokenizer, text_encoder in zip(tokenizers, text_encoders):
                text_inputs = tokenizer(
                    chunk,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                input_ids = text_inputs.input_ids.to(device)
                outputs = text_encoder(input_ids, output_hidden_states=True)
                pooled = outputs[0]
                hidden = outputs.hidden_states[-2]
                embeds_list.append(hidden)
                pooled_list.append(pooled)
            prompt_embeds = torch.cat(embeds_list, dim=-1).cpu()
            pooled_embeds = pooled_list[-1].cpu()
            all_prompt_embeds.append(prompt_embeds)
            all_text_embeds.append(pooled_embeds)

    prompt_embeds = torch.cat(all_prompt_embeds, dim=0)
    text_embeds = torch.cat(all_text_embeds, dim=0)
    time_ids = add_time_ids.unsqueeze(0).repeat(prompt_embeds.shape[0], 1)
    return prompt_embeds, text_embeds, time_ids


class PrecomputedEmbeddingDataset(torch_data.Dataset):
    """Wraps a :class:`PairedWeatherDataset` with pre-computed text embeddings."""

    def __init__(
        self,
        base: PairedWeatherDataset,
        prompt_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        time_ids: torch.Tensor,
    ) -> None:
        self.base = base
        self.prompt_embeds = prompt_embeds
        self.text_embeds = text_embeds
        self.time_ids = time_ids
        if prompt_embeds.shape[0] != len(base):
            raise ValueError("prompt_embeds length must match dataset size")
        if text_embeds.shape[0] != len(base):
            raise ValueError("text_embeds length must match dataset size")
        if time_ids.shape[0] != len(base):
            raise ValueError("time_ids length must match dataset size")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = self.base[index]
        return {
            "pixel_values": item["pixel_values"],
            "conditioning_pixel_values": item["conditioning_pixel_values"],
            "prompt_embeds": self.prompt_embeds[index],
            "text_embeds": self.text_embeds[index],
            "time_ids": self.time_ids[index],
            "weather": item["weather"],
        }


def build_precomputed_dataset(
    dataset: PairedWeatherDataset,
    text_encoders: Sequence[torch.nn.Module],
    tokenizers: Sequence,
    device: torch.device,
    original_size: Tuple[int, int],
    crops_coords_top_left: Tuple[int, int],
    target_size: Tuple[int, int],
    proportion_empty_prompts: float = 0.0,
    batch_size: int = 8,
    skip_prompt_extraction: bool = False,
) -> PrecomputedEmbeddingDataset:
    pe, te, ti = encode_weather_prompts(
        dataset=dataset,
        text_encoders=text_encoders,
        tokenizers=tokenizers,
        device=device,
        original_size=original_size,
        crops_coords_top_left=crops_coords_top_left,
        target_size=target_size,
        proportion_empty_prompts=proportion_empty_prompts,
        batch_size=batch_size,
        skip_prompt_extraction=skip_prompt_extraction,
    )
    return PrecomputedEmbeddingDataset(dataset, pe, te, ti)
