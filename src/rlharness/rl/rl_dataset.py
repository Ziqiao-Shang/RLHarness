"""MapTab dataset adapter using the same image resize order as SFT and evaluation."""

from __future__ import annotations

import asyncio
import math
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image
from verl.utils.dataset.rl_dataset import RLHFDataset


def _load_image(source: Any) -> Image.Image:
    if isinstance(source, Image.Image):
        return source.copy()
    if isinstance(source, bytes):
        return Image.open(BytesIO(source)).copy()
    path = str(source)
    if path.startswith("file://"):
        path = path[7:]
    with Image.open(Path(path)) as image:
        return image.copy()


def _resize_by_area(image: Image.Image, min_pixels: int, max_pixels: int) -> Image.Image:
    """Apply the area-preserving resize used before Qwen smart resize."""
    area = image.width * image.height
    if area > max_pixels:
        factor = math.sqrt(max_pixels / area)
        image = image.resize((int(image.width * factor), int(image.height * factor)))
    elif area < min_pixels:
        factor = math.sqrt(min_pixels / area)
        image = image.resize((int(image.width * factor), int(image.height * factor)))
    return image.convert("RGB")


class MapTabRLHFDataset(RLHFDataset):
    """Pre-resize MapTab images consistently across SFT, evaluation, and RL."""

    def _build_messages(self, example, key):
        return RLHFDataset._build_messages(self, example, key)

    @classmethod
    def _process_multi_modal_info(cls, messages, image_patch_size, config):
        del image_patch_size
        processor_kwargs = config.get("mm_processor_kwargs", {}) or {}
        max_pixels = int(processor_kwargs.get("max_pixels", 1_000_000))
        min_pixels = int(processor_kwargs.get("min_pixels", 3_136))

        images = []
        has_video = False
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "video":
                    has_video = True
                if item.get("type") != "image":
                    continue
                source = item.get("image")
                if source is None and "bytes" in item:
                    source = item["bytes"]
                if source is None:
                    raise ValueError("Image item must contain 'image' or 'bytes'.")
                item_max = int(item.get("max_pixels", max_pixels))
                item_min = int(item.get("min_pixels", min_pixels))
                images.append(_resize_by_area(_load_image(source), item_min, item_max))

        videos = None
        if has_video:
            _, videos, _ = super()._process_multi_modal_info(messages, 16, config)
        audios = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "audio":
                    continue
                audios.append(item.get("audio", item.get("audio_url")))
        return images or None, videos, audios or None

    @classmethod
    async def process_multi_modal_info(cls, messages, image_patch_size, config):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: cls._process_multi_modal_info(messages, image_patch_size, config),
        )
