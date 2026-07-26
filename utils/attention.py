"""Attention backend selection + DataLoader / training-loop micro-optimizations.

The default ``auto`` policy tries the most efficient backend available:
  1. xformers (if installed + flag set)
  2. PyTorch SDPA flash-attention (Ampere+)
  3. PyTorch SDPA memory-efficient
  4. math (always available, no acceleration)
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Optional

import torch

logger = logging.getLogger(__name__)


def _has_xformers() -> bool:
    try:
        from diffusers.utils.import_utils import is_xformers_available

        return is_xformers_available()
    except Exception:  # noqa: BLE001
        return False


def configure_sdpa_backends(enable_flash: bool = True, enable_mem_efficient: bool = True, enable_math: bool = True) -> None:
    if not torch.cuda.is_available():
        return
    try:
        torch.backends.cuda.enable_flash_sdp(enable_flash)
        torch.backends.cuda.enable_mem_efficient_sdp(enable_mem_efficient)
        torch.backends.cuda.enable_math_sdp(enable_math)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to configure SDPA backends: %s", exc)


def enable_efficient_attention(
    model: torch.nn.Module,
    backend: str = "auto",
) -> str:
    """Enable a memory-efficient attention backend on ``model`` (UNet / ControlNet / pipeline).

    Returns the name of the backend that was actually enabled.
    """
    backend = (backend or "auto").lower()

    if backend in ("auto", "xformers") and _has_xformers():
        try:
            model.enable_xformers_memory_efficient_attention()
            logger.info("[attn] enabled xformers on %s", type(model).__name__)
            return "xformers"
        except Exception as exc:  # noqa: BLE001
            logger.warning("[attn] xformers failed (%s), falling back", exc)

    if backend in ("auto", "sdpa", "flash"):
        configure_sdpa_backends(enable_flash=True, enable_mem_efficient=True, enable_math=True)
        try:
            from diffusers.models.attention_processor import AttnProcessor2_0

            model.set_attn_processor(AttnProcessor2_0())
            logger.info("[attn] enabled SDPA (flash/mem-efficient) on %s", type(model).__name__)
            return "sdpa"
        except Exception as exc:  # noqa: BLE001
            logger.warning("[attn] SDPA failed (%s), falling back to math", exc)

    logger.info("[attn] using math backend on %s", type(model).__name__)
    return "math"


def move_optimistic(x: Any, **kwargs: Any) -> Any:
    """``x.to(**kwargs)`` with ``non_blocking=True`` when target device is CUDA."""
    if "non_blocking" not in kwargs:
        kwargs["non_blocking"] = True
    return x.to(**kwargs)
