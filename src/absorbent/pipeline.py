"""Оркестрация конвейера: файл -> конспект Markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from absorbent.config import Config
from absorbent.dedup import dedup_frames
from absorbent.describe import describe_keyframes
from absorbent.extract import extract_frames
from absorbent.render import render_all
from absorbent.synthesize import run_synthesize
from absorbent.transcribe import run_stt


def _cache_dir(output_dir: Path) -> Path:
    """Служебный каталог промежуточных результатов."""
    return output_dir / ".absorbent"


def _load_json(path: Path) -> list | dict:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def process_file(input_path: Path, cfg: Config, args: argparse.Namespace) -> Path:
    """Прогнать все этапы конвейера для одного файла; вернуть путь notes.md.

    Порядок этапов: извлечение кадров -> дедупликация -> распознавание речи
    -> описание кадров -> синтез Markdown -> постпроцессинг (таблицы,
    HTML, PDF). Каждый этап уважает свой `--skip-*` флаг и кэш в
    `.absorbent/` (обход кэша — `--force`); постпроцессинг отключается
    флагом `--skip-render`.

    Язык распознавания речи (`--language`) и язык результата
    (`--output-language`) — независимые оси: первый лишь подсказывает whisper
    (для двуязычных лекций всегда 'auto'), второй явно задаёт язык промптов
    описания и синтеза (по умолчанию 'ru').
    """
    stem = input_path.stem
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_path.with_name(stem)
    images_dir = output_dir / "images"
    cache = _cache_dir(output_dir)
    frames_dir = cache / "frames"
    for directory in (output_dir, images_dir, cache, frames_dir):
        directory.mkdir(parents=True, exist_ok=True)

    keyframes_path = cache / "keyframes.json"
    transcript_path = cache / "transcript.json"
    descriptions_path = cache / "descriptions.json"
    notes_path = output_dir / "notes.md"

    output_language = getattr(args, "output_language", "ru")

    print(f"[{stem}] извлечение кадров...")
    keyframes: list[dict] = []
    if not args.skip_frames:
        frames = extract_frames(input_path, frames_dir, args.fps, force=args.force)
        if args.force or not keyframes_path.exists():
            keyframes = dedup_frames(frames, images_dir, args.fps, args.hash_threshold)
            keyframes_path.write_text(json.dumps(keyframes, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
        else:
            keyframes = json.loads(keyframes_path.read_text(encoding="utf-8"))

    print(f"[{stem}] распознавание речи...")
    if not args.skip_stt:
        stt_provider, stt_spec = cfg.resolve("stt", args.stt)
        run_stt(stt_spec, stt_provider, input_path, args.language, transcript_path,
                force=args.force)
    transcript = _load_json(transcript_path)

    descriptions: dict = {}
    if not args.skip_describe:
        print(f"[{stem}] описание {len(keyframes)} ключевых кадров...")
        if keyframes:
            vision_provider, vision_spec = cfg.resolve("vision", args.vision_model)
            descriptions = describe_keyframes(vision_spec, vision_provider, keyframes, transcript,
                                              images_dir, output_language, descriptions_path,
                                              force=args.force)
        else:
            descriptions = _load_json(descriptions_path)
            if isinstance(descriptions, list):
                descriptions = {}
    elif descriptions_path.exists():
        loaded = _load_json(descriptions_path)
        descriptions = loaded if isinstance(loaded, dict) else {}

    if not args.skip_synthesize:
        print(f"[{stem}] синтез Markdown...")
        llm_provider, llm_spec = cfg.resolve("llm", args.llm_model)
        run_synthesize(llm_spec, llm_provider, transcript, descriptions, output_language,
                       notes_path)

    if not getattr(args, "skip_render", False) and notes_path.exists():
        print(f"[{stem}] постпроцессинг: таблицы, HTML, PDF...")
        render_all(notes_path)
        print(f"[{stem}] готово: {notes_path} (+ notes.html, notes.pdf)")
    return notes_path
