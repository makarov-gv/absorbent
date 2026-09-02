"""Тесты дедупликации кадров по pHash.

Зафиксировано эмпирическое поведение imagehash: pHash (DCT) не различает
однотонные заливки — три разных сплошных цвета дают один и тот же хеш,
поэтому все они схлопываются в один кадр. Мелкие геометрические детали
(узкие полосы) на маленьких кадрах тоже теряются в 32x32-даунсемпле.
Надёжно различаются только крупные паттерны: диагональные треугольники
и центральный квадрат дают >= 26 бит расстояния при 64x64 JPEG.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from absorbent.dedup import _format_timestamp, _frame_seconds, dedup_frames


def _make_frame(path: Path, bg: tuple[int, int, int], pattern: str) -> None:
    """Создать кадр 64x64 с крупным белым паттерном на цветном фоне."""
    img = Image.new("RGB", (64, 64), bg)
    draw = ImageDraw.Draw(img)
    if pattern == "diag_lr":  # заливка левого нижнего треугольника.
        draw.polygon([(0, 0), (64, 0), (0, 64)], fill="white")
    elif pattern == "diag_rl":  # заливка правого нижнего треугольника.
        draw.polygon([(0, 0), (64, 64), (64, 0)], fill="white")
    elif pattern == "center":  # белый квадрат в центре.
        draw.rectangle([16, 16, 48, 48], fill="white")
    else:
        raise AssertionError(f"неизвестный паттерн: {pattern}")
    img.save(path, "JPEG")


class TestTimestampHelpers:
    def test_frame_seconds(self) -> None:
        assert _frame_seconds("frame_000001.jpg", 1.0) == 0.0
        assert _frame_seconds("frame_000005.jpg", 1.0) == 4.0
        assert _frame_seconds("frame_000003.jpg", 0.5) == 4.0  # (3-1)/0.5.
        assert _frame_seconds("frame_000010.jpg", 1.0) == 9.0

    def test_format_timestamp(self) -> None:
        assert _format_timestamp(0.0) == "00:00:00"
        assert _format_timestamp(65.4) == "00:01:05"
        assert _format_timestamp(3661.9) == "01:01:02"  # 3661.9 -> 3662 с.
        assert _format_timestamp(86399.6) == "24:00:00"  # округление вверх через границу часов.


class TestDedupFrames:
    def test_solid_colors_collapse_to_one(self, tmp_path: Path) -> None:
        """pHash не видит разницы между сплошными цветами — остаётся один кадр."""
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        for index, color in enumerate([(220, 0, 0), (0, 180, 0), (0, 0, 200)], start=1):
            _make_frame(frames_dir / f"frame_{index:06d}.jpg", color, "diag_lr")
        images_dir = tmp_path / "images"
        keyframes = dedup_frames(sorted(frames_dir.glob("frame_*.jpg")), images_dir, 1.0, 10)
        assert len(keyframes) == 1
        assert keyframes[0]["timestamp"] == "00:00:00"

    def test_distinct_patterns_all_kept(self, tmp_path: Path) -> None:
        """Крупные различные паттерны (треугольники/квадрат) сохраняются все."""
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        patterns = [("diag_lr", (150, 30, 30)), ("diag_rl", (30, 90, 30)), ("center", (30, 30, 120))]
        for index, (pattern, bg) in enumerate(patterns, start=1):
            _make_frame(frames_dir / f"frame_{index:06d}.jpg", bg, pattern)
        images_dir = tmp_path / "images"
        keyframes = dedup_frames(sorted(frames_dir.glob("frame_*.jpg")), images_dir, 1.0, 10)
        assert len(keyframes) == 3
        # Имена по таймкодам: 0, 1, 2 секунды.
        assert [k["image"] for k in keyframes] == [
            "frame_00_00_00.jpg",
            "frame_00_00_01.jpg",
            "frame_00_00_02.jpg",
        ]
        for k in keyframes:
            assert (images_dir / k["image"]).is_file()

    def test_duplicate_pattern_dropped(self, tmp_path: Path) -> None:
        """Третий кадр дублирует второй -> остаются два ключевых кадра."""
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        _make_frame(frames_dir / "frame_000001.jpg", (10, 10, 10), "diag_lr")
        _make_frame(frames_dir / "frame_000002.jpg", (10, 10, 10), "diag_rl")
        _make_frame(frames_dir / "frame_000003.jpg", (10, 10, 10), "diag_rl")  # дубль второго.
        images_dir = tmp_path / "images"
        keyframes = dedup_frames(sorted(frames_dir.glob("frame_*.jpg")), images_dir, 1.0, 10)
        assert len(keyframes) == 2
        assert sorted(p.name for p in images_dir.iterdir()) == ["frame_00_00_00.jpg", "frame_00_00_01.jpg"]

    def test_zero_threshold_keeps_all_distinct(self, tmp_path: Path) -> None:
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        _make_frame(frames_dir / "frame_000001.jpg", (1, 1, 1), "diag_lr")
        _make_frame(frames_dir / "frame_000002.jpg", (2, 2, 2), "diag_rl")
        images_dir = tmp_path / "images"
        keyframes = dedup_frames(sorted(frames_dir.glob("frame_*.jpg")), images_dir, 1.0, 0)
        assert len(keyframes) == 2

    def test_high_threshold_keeps_first_only(self, tmp_path: Path) -> None:
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        _make_frame(frames_dir / "frame_000001.jpg", (1, 1, 1), "diag_lr")
        _make_frame(frames_dir / "frame_000002.jpg", (2, 2, 2), "diag_rl")
        images_dir = tmp_path / "images"
        # Порог 64: любое расстояние <= 64 отбрасывается.
        keyframes = dedup_frames(sorted(frames_dir.glob("frame_*.jpg")), images_dir, 1.0, 64)
        assert len(keyframes) == 1

    def test_seconds_recorded_from_fps(self, tmp_path: Path) -> None:
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        _make_frame(frames_dir / "frame_000001.jpg", (1, 1, 1), "diag_lr")
        _make_frame(frames_dir / "frame_000003.jpg", (2, 2, 2), "diag_rl")
        images_dir = tmp_path / "images"
        keyframes = dedup_frames(sorted(frames_dir.glob("frame_*.jpg")), images_dir, 0.5, 10)
        # (3-1)/0.5 = 4 секунды.
        assert keyframes[1]["seconds"] == 4.0
        assert keyframes[1]["timestamp"] == "00:00:04"

    def test_copy2_used(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Кадры копируются через shutil.copy2 (сохраняет метаданные)."""
        import absorbent.dedup as dedup_module

        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        _make_frame(frames_dir / "frame_000001.jpg", (1, 1, 1), "diag_lr")
        images_dir = tmp_path / "images"
        real_copy2 = shutil.copy2
        copied: list[tuple] = []

        def _spy(src, dst):
            copied.append((src, dst))
            return real_copy2(src, dst)

        monkeypatch.setattr(dedup_module.shutil, "copy2", _spy)
        dedup_frames([frames_dir / "frame_000001.jpg"], images_dir, 1.0, 10)
        assert len(copied) == 1
        assert Path(copied[0][1]).name == "frame_00_00_00.jpg"
