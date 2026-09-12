"""Отбор ключевых кадров по перцептивному хешу: сцены и их ядра."""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

# Размер блока при подсчёте сумм расстояний: ограничивает матрицу расстояний сцены.
MEDOID_BLOCK_FRAMES = 512


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


def _frame_hashes(frames: list[Path]) -> np.ndarray:
    """pHash каждого кадра как одно 64-битное число.

    Порядок бит внутри числа значения не имеет: дальше считаются только
    расстояния Хэмминга.
    """
    return np.array([
        np.packbits(imagehash.phash(Image.open(frame)).hash.reshape(-1)).view(np.uint64)[0]
        for frame in frames
    ], dtype=np.uint64)


def _scenes(hashes: np.ndarray, threshold: int) -> list[tuple[int, int]]:
    """Отрезки подряд идущих кадров: каждый кадр не дальше threshold бит от первого в отрезке.

    Порог играет роль границы смены слайда. Цепное связывание здесь
    невозможно: сравнивается не с предыдущим кадром, а с первым кадром
    отрезка, поэтому сцена не растягивается на всю лекцию (см. BACKLOG.md,
    раздел «Отвергнуто»).
    """
    if not len(hashes):
        return []
    scenes: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(hashes)):
        if int(np.bitwise_count(hashes[index] ^ hashes[start])) > threshold:
            scenes.append((start, index - 1))
            start = index
    scenes.append((start, len(hashes) - 1))
    return scenes


def _medoid(start: int, end: int, hashes: np.ndarray) -> int:
    """Индекс ядра сцены: минимальная сумма расстояний до остальных кадров сцены.

    Анимационные кадры и артефакты (курсор, расфокус, промежуточный кадр
    перехода) дальше от центра сцены и ядром не становятся. Суммы считаются
    блоками, чтобы не держать в памяти матрицу расстояний всей сцены.
    """
    members = hashes[start:end + 1]
    totals = np.zeros(len(members), dtype=np.int64)
    for offset in range(0, len(members), MEDOID_BLOCK_FRAMES):
        block = members[offset:offset + MEDOID_BLOCK_FRAMES]
        totals[offset:offset + len(block)] = np.bitwise_count(
            block[:, None] ^ members[None, :]).sum(axis=1)
    return start + int(np.argmin(totals))


def _drop_similar(indices: list[int], hashes: np.ndarray, threshold: int) -> list[int]:
    """Убрать кадр, похожий (не дальше threshold бит) на предыдущий оставленный."""
    kept: list[int] = []
    for index in indices:
        if kept and int(np.bitwise_count(hashes[index] ^ hashes[kept[-1]])) <= threshold:
            continue
        kept.append(index)
    return kept


def select_keyframes(frames: list[Path], images_dir: Path, fps: float, threshold: int,
                     min_seconds: float) -> list[dict]:
    """Отобрать ключевые кадры: по одному ядру на каждую достаточно долгую сцену.

    Кадры разбиваются на сцены по pHash (см. _scenes); сцена короче
    min_seconds секунд (min_seconds * fps кадров, не меньше одного) считается
    анимацией, промежуточным состоянием или шумом и отбрасывается; из
    оставшихся сцен берётся ядро (см. _medoid), а ядра, похожие на соседа по
    времени, не дублируются (см. _drop_similar). Если сцен длиннее порога нет
    вовсе, берётся ядро самого длинного отрезка, чтобы конспект не остался без
    иллюстраций. Ядра копируются в images_dir как frame_HH_MM_SS.jpg, а
    каталог приводится к текущему отбору: кадры прошлого отбора удаляются.
    Возвращаются записи {"image", "seconds", "timestamp"} по хронологии.
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    hashes = _frame_hashes(frames)
    min_frames = max(1, math.ceil(min_seconds * fps))
    scenes = _scenes(hashes, threshold)
    kept = [scene for scene in scenes if scene[1] - scene[0] + 1 >= min_frames]
    if not kept and scenes:
        print(f"внимание: нет сцен длиной {min_seconds} с и более, взят самый длинный отрезок")
        kept = [max(scenes, key=lambda scene: scene[1] - scene[0])]
    chosen = []
    for index in _drop_similar([_medoid(start, end, hashes) for start, end in kept],
                               hashes, threshold):
        seconds = _frame_seconds(frames[index].name, fps)
        timestamp = _format_timestamp(seconds)
        chosen.append((index, f"frame_{timestamp.replace(':', '_')}.jpg", seconds, timestamp))
    names = {name for _, name, _, _ in chosen}
    for stale in images_dir.glob("frame_*.jpg"):
        if stale.name not in names:
            stale.unlink()
    keyframes: list[dict] = []
    for index, name, seconds, timestamp in chosen:
        shutil.copy2(frames[index], images_dir / name)
        keyframes.append({"image": name, "seconds": seconds, "timestamp": timestamp})
    return keyframes
