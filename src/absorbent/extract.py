"""Извлечение кадров из видео через ffmpeg."""

from __future__ import annotations

import subprocess
from pathlib import Path


class ExtractionError(Exception):
    """Ошибка запуска ffmpeg при извлечении кадров."""


def extract_frames(input_path: Path, frames_dir: Path, fps: float, force: bool = False) -> list[Path]:
    """Извлечь кадры с частотой fps в frames_dir; вернуть отсортированный список файлов.

    Повторный запуск без force переиспользует уже извлечённые кадры.
    """
    frames_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(frames_dir.glob("frame_*.jpg"))
    if existing and not force:
        return existing

    pattern = str(frames_dir / "frame_%06d.jpg")
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(input_path), "-vf", f"fps={fps}", "-q:v", "2", pattern],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise ExtractionError(f"ffmpeg завершился с кодом {proc.returncode}: {proc.stderr.strip()}")
    return sorted(frames_dir.glob("frame_*.jpg"))
