"""Оркестрация конвейера: файл -> конспект Markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from absorbent.config import Config
from absorbent.dedup import select_keyframes
from absorbent.describe import describe_keyframes
from absorbent.extract import extract_frames, has_video_stream
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
    """Прогнать все этапы конвейера для одного файла; вернуть путь конспекта.

    Раскладка по умолчанию: каталог <имя>/ рядом с входом, внутри notes.md,
    images/ и .absorbent/ с кэшем. Плоская раскладка (--flat): <имя>.md,
    <имя>.json и <имя>.images/ ложатся рядом с медиафайлом, а кэш — в
    <каталог входа>/.absorbent/<имя>/.

    Порядок этапов: извлечение кадров -> отбор ключевых кадров ->
    распознавание речи -> описание кадров -> синтез Markdown ->
    постпроцессинг (таблицы, HTML, PDF). Вход без видеопотока (аудиофайл)
    пропускает все этапы кадров: остаётся транскрипт и конспект без
    иллюстраций. Каждый этап уважает свой `--skip-*` флаг и кэш в
    `.absorbent/` (обход кэша — `--force`); постпроцессинг отключается
    флагом `--skip-render`.

    Язык распознавания речи (`--language`) и язык результата
    (`--output-language`) — независимые оси: первый лишь подсказывает whisper
    (для двуязычных лекций всегда 'auto'), второй явно задаёт язык промптов
    описания и синтеза (по умолчанию 'ru').
    """
    stem = input_path.stem
    flat = getattr(args, "flat", False)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif flat:
        output_dir = input_path.parent
    else:
        output_dir = input_path.with_name(stem)
    if flat:
        cache = input_path.parent / ".absorbent" / stem
        notes_path = output_dir / f"{stem}.md"
        transcript_path = output_dir / f"{stem}.json"
        images_dir = output_dir / f"{stem}.images"
        images_prefix = f"{stem}.images/"
    else:
        cache = _cache_dir(output_dir)
        notes_path = output_dir / "notes.md"
        transcript_path = cache / "transcript.json"
        images_dir = output_dir / "images"
        images_prefix = "images/"
    frames_dir = cache / "frames"
    keyframes_path = cache / "keyframes.json"
    descriptions_path = cache / "descriptions.json"
    for directory in (output_dir, images_dir, cache, frames_dir):
        directory.mkdir(parents=True, exist_ok=True)

    output_language = getattr(args, "output_language", "ru")
    min_scene_seconds = getattr(args, "min_scene_seconds", 5.0)

    print(f"[{stem}] извлечение кадров...")
    keyframes: list[dict] = []
    if not args.skip_frames and has_video_stream(input_path):
        frames = extract_frames(input_path, frames_dir, args.fps, force=args.force)
        if args.force or not keyframes_path.exists():
            keyframes = select_keyframes(frames, images_dir, args.fps, args.hash_threshold,
                                         min_scene_seconds)
            keyframes_path.write_text(json.dumps(keyframes, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
        else:
            keyframes = json.loads(keyframes_path.read_text(encoding="utf-8"))
    elif not args.skip_frames:
        print(f"[{stem}] кадров нет (аудио без видеопотока)")

    print(f"[{stem}] распознавание речи...")
    if not args.skip_stt:
        stt_provider, stt_spec = cfg.resolve("stt", args.stt)
        run_stt(stt_spec, stt_provider, input_path, args.language, transcript_path, cache,
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

    # Описания кадров прошлого отбора в синтез не идут: иначе модель вставит ссылки на чужие кадры.
    if keyframes:
        wanted = {kf["image"] for kf in keyframes}
        descriptions = {name: entry for name, entry in descriptions.items() if name in wanted}

    if not args.skip_synthesize:
        print(f"[{stem}] синтез Markdown...")
        llm_provider, llm_spec = cfg.resolve("llm", args.llm_model)
        run_synthesize(llm_spec, llm_provider, transcript, descriptions, output_language,
                       notes_path, images_prefix=images_prefix)

    if not getattr(args, "skip_render", False) and notes_path.exists():
        print(f"[{stem}] постпроцессинг: таблицы, HTML, PDF...")
        if flat:
            render_all(notes_path, output_dir / f"{stem}.html", output_dir / f"{stem}.pdf",
                       lang=output_language)
            print(f"[{stem}] готово: {notes_path}")
        else:
            render_all(notes_path, lang=output_language)
            print(f"[{stem}] готово: {notes_path} (+ notes.html, notes.pdf)")
    return notes_path
