"""Дедупликация кадров по перцептивному хешу."""

from __future__ import annotations

import shutil
from pathlib import Path

import imagehash
from PIL import Image


def _frame_seconds(name: str, fps: float) -> float:
    """Время кадра frame_%06d.jpg в секундах от начала видео."""
    index = int(Path(name).stem.split("_")[1])
    return (index - 1) / fps


def _format_timestamp(seconds: float) -> str:
    """Секунды -> строка HH:MM:SS (округляется до секунды)."""
    total = int(round(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def dedup_frames(frames: list[Path], images_dir: Path, fps: float, threshold: int) -> list[dict]:
    """Отобрать репрезентативные кадры по pHash-расстоянию и скопировать их в images_dir.

    Кадр сохраняется, если расстояние Хэмминга его pHash от последнего
    сохранённого кадра больше threshold. Возвращает записи
    {"image", "seconds", "timestamp"} в порядке следования кадров.
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    keyframes: list[dict] = []
    last_hash: imagehash.ImageHash | None = None
    for frame in frames:
        current_hash = imagehash.phash(Image.open(frame))
        if last_hash is not None and (current_hash - last_hash) <= threshold:
            continue
        last_hash = current_hash
        seconds = _frame_seconds(frame.name, fps)
        timestamp = _format_timestamp(seconds)
        target = images_dir / f"frame_{timestamp.replace(':', '_')}.jpg"
        shutil.copy2(frame, target)
        keyframes.append({"image": target.name, "seconds": seconds, "timestamp": timestamp})
    return keyframes
