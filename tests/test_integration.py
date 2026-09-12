"""Интеграционный тест: реальные ffmpeg/ffprobe/pHash, сетевые вызовы подменены.

Проверяет весь конвейер process_file end-to-end на синтетическом видео с
тремя разными статичными слайдами: реальные извлечение кадров и
дедупликация должны дать три ключевых кадра (0/5/10 с), а подменённые
STT/vision/LLM — собрать notes.md, где имена изображений ровно совпадают с
файлами на диске (контракт шага синтеза).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from absorbent.config import load_config
from absorbent.pipeline import process_file

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                reason="ffmpeg/ffprobe не установлены")


def _make_slide(path: Path, color: tuple[int, int, int], pattern: str) -> None:
    """Слайд 320x180 с крупным различимым паттерном (pHash-различие > 10 бит)."""
    img = Image.new("RGB", (320, 180), color)
    draw = ImageDraw.Draw(img)
    if pattern == "diag_lr":
        draw.polygon([(0, 0), (320, 0), (0, 180)], fill="white")
    elif pattern == "diag_rl":
        draw.polygon([(0, 0), (320, 180), (320, 0)], fill="white")
    elif pattern == "center":
        draw.rectangle([80, 45, 240, 135], fill="white")
    else:
        raise AssertionError(pattern)
    img.save(path, "JPEG")


def _build_video(tmp_path: Path) -> Path:
    """Три статичных слайда по 5 с + тихий тон: 15-секундное видео."""
    slides = [tmp_path / f"slide_{i}.png" for i in range(1, 4)]
    _make_slide(slides[0], (150, 30, 30), "diag_lr")
    _make_slide(slides[1], (30, 90, 30), "diag_rl")
    _make_slide(slides[2], (30, 30, 120), "center")
    video = tmp_path / "fixture.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-loop", "1", "-t", "5", "-i", str(slides[0]),
         "-loop", "1", "-t", "5", "-i", str(slides[1]),
         "-loop", "1", "-t", "5", "-i", str(slides[2]),
         "-f", "lavfi", "-i", "sine=frequency=440:duration=15",
         "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0,format=yuv420p[v]",
         "-map", "[v]", "-map", "3:a", "-c:v", "libx264", "-c:a", "aac", "-shortest",
         str(video)],
        check=True,
        capture_output=True,
    )
    return video


@pytest.fixture
def offline_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, args_namespace):
    """process_file с реальными extract/dedup и подменёнными сетевыми этапами."""
    import absorbent.pipeline as pipeline_module

    video = _build_video(tmp_path)
    out_dir = tmp_path / "out"

    def _fake_stt(spec, provider, input_path, language, out_json, work_dir, force=False):
        segments = [
            {"start": 0.0, "end": 2.0, "text": "Введение в тему", "speaker": None},
            {"start": 5.5, "end": 7.0, "text": "Теперь формулы", "speaker": None},
            {"start": 10.5, "end": 12.0, "text": "Подводим итоги", "speaker": None},
        ]
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(segments, ensure_ascii=False), encoding="utf-8")
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "language.txt").write_text("ru", encoding="utf-8")
        return "ru"

    def _fake_describe(vision_spec, provider, keyframes, transcript, images_dir, output_language,
                       out_json, force=False):
        data = {
            kf["image"]: {"timestamp": kf["timestamp"], "description": f"слайд на {kf['timestamp']}"}
            for kf in keyframes
        }
        out_json.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return data

    def _fake_synthesize(llm_spec, provider, transcript, descriptions, output_language, out_md,
                         images_prefix="images/"):
        # Реалистичный Markdown: подпись + ссылка на каждый кадр (по контракту
        # имена берутся только из списка описаний).
        parts = ["# Конспект интеграционного теста\n"]
        for name in sorted(descriptions):
            parts.append(f"![Кадр {descriptions[name]['timestamp']}]({images_prefix}{name})\n")
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text("\n".join(parts), encoding="utf-8")
        return out_md

    monkeypatch.setattr(pipeline_module, "run_stt", _fake_stt)
    monkeypatch.setattr(pipeline_module, "describe_keyframes", _fake_describe)
    monkeypatch.setattr(pipeline_module, "run_synthesize", _fake_synthesize)
    def _skip_render(notes_md, out_html=None, out_pdf=None, lang=None):
        return notes_md.with_name("notes.html"), notes_md.with_name("notes.pdf")

    monkeypatch.setattr(pipeline_module, "render_all", _skip_render)
    cfg = load_config(str(REPO_CONFIG))
    return video, out_dir, cfg, args_namespace


class TestEndToEnd:
    def test_three_keyframes_and_image_contract(self, offline_pipeline) -> None:
        video, out_dir, cfg, args = offline_pipeline
        args.output_dir = str(out_dir)
        notes_path = process_file(video, cfg, args)

        # 1. Ровно три ключевых кадра на 0/5/10 секундах.
        images = sorted(p.name for p in (out_dir / "images").glob("frame_*.jpg"))
        assert images == ["frame_00_00_00.jpg", "frame_00_00_05.jpg", "frame_00_00_10.jpg"]

        # 2. Транскрипт на месте, с языком.
        transcript = json.loads((out_dir / ".absorbent" / "transcript.json").read_text(encoding="utf-8"))
        assert len(transcript) == 3
        assert (out_dir / ".absorbent" / "language.txt").read_text(encoding="utf-8") == "ru"

        # 3. notes.md существует и ссылается ровно на файлы с диска.
        assert notes_path.exists()
        text = notes_path.read_text(encoding="utf-8")
        refs = re.findall(r"images/(frame_[0-9_]+\.jpg)", text)
        assert sorted(refs) == images  # нет выдуманных имён, нет пропущенных.

        # 4. Повторный прогон из кэша: файлы результата переживают.
        notes_path.unlink()
        args.force = False
        process_file(video, cfg, args)
        assert notes_path.exists()

    def test_reproducible_dedup_on_rerun(self, offline_pipeline) -> None:
        """Повторная дедупликация из кэша кадров даёт те же ключевые кадры."""
        video, out_dir, cfg, args = offline_pipeline
        args.output_dir = str(out_dir)
        process_file(video, cfg, args)
        first = sorted(p.name for p in (out_dir / "images").glob("frame_*.jpg"))
        # Пересборка keyframes.json с другим порогом из тех же кадров не
        # переизвлекает видео (extract из кэша), но изображения обновляются.
        args.force = False
        args.hash_threshold = 0
        process_file(video, cfg, args)
        # Порог 0 всё равно оставляет три разных паттерна (расстояния > 0).
        second = sorted(p.name for p in (out_dir / "images").glob("frame_*.jpg"))
        assert second == first


class TestAudioFlatMode:
    """Аудиовход в плоском режиме: настоящие ffmpeg/ffprobe, сетевые этапы подменены."""

    def test_audio_input_flat_layout(self, offline_pipeline, tmp_path: Path) -> None:
        _, _, cfg, args = offline_pipeline
        audio = tmp_path / "lecture.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
             "-i", "sine=frequency=440:duration=3", "-ac", "1", "-ar", "16000", str(audio)],
            check=True,
            capture_output=True,
        )
        args.flat = True
        notes = process_file(audio, cfg, args)

        # Результат и сырой транскрипт — рядом с медиафайлом.
        assert notes == tmp_path / "lecture.md"
        transcript = json.loads((tmp_path / "lecture.json").read_text(encoding="utf-8"))
        assert [seg["text"] for seg in transcript] == ["Введение в тему", "Теперь формулы",
                                                       "Подводим итоги"]
        # Кадров нет: у wav нет видеопотока, каталог изображений пуст.
        assert not list((tmp_path / "lecture.images").glob("frame_*.jpg"))
        assert "lecture.images/" not in notes.read_text(encoding="utf-8")
        # Кэш — в <каталог>/.absorbent/<имя>/, каталога <имя>/ нет.
        cache = tmp_path / ".absorbent" / "lecture"
        assert (cache / "language.txt").read_text(encoding="utf-8") == "ru"
        assert not (cache / "transcript.json").exists()
        assert not (tmp_path / "lecture").exists()
