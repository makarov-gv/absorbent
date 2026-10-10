"""Оркестрация конвейера: файл -> конспект Markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from absorbent.config import Config
from absorbent.dedup import select_keyframes
from absorbent.describe import describe_keyframes, describe_slides
from absorbent.events import stage
from absorbent.extract import extract_frames, has_video_stream
from absorbent.render import outputs_up_to_date, render_all
from absorbent.slides import prepare_slides
from absorbent.synthesize import run_synthesize
from absorbent.transcribe import run_stt


def _cache_dir(output_dir: Path) -> Path:
    """Служебный каталог промежуточных результатов."""
    return output_dir / ".absorbent"


def _load_json(path: Path) -> list | dict:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _filter_by_images(items: dict, images: set[str]) -> dict:
    """Оставить описания текущего набора картинок: чужой отбор в синтез не идёт."""
    return {name: entry for name, entry in items.items() if name in images}


def process_file(input_path: Path, cfg: Config, args: argparse.Namespace) -> Path:
    """Прогнать все этапы конвейера для одного файла; вернуть путь конспекта.

    Раскладка по умолчанию: каталог <имя>/ рядом с входом, внутри notes.md,
    images/ и .absorbent/ с кэшем. Плоская раскладка (--flat): <имя>.md,
    <имя>.json и <имя>.images/ ложатся рядом с медиафайлом, а кэш — в
    <каталог входа>/.absorbent/<имя>/.

    Порядок этапов: извлечение кадров -> отбор ключевых кадров ->
    подготовка презентаций и фотографий доски -> распознавание речи ->
    описание кадров и материалов -> синтез Markdown -> постпроцессинг
    (таблицы, HTML, PDF). Вход без видеопотока (аудиофайл) пропускает все
    этапы кадров: остаётся транскрипт и конспект без иллюстраций, а каталогов
    кадров (frames/ и images/) не появляется вовсе — так же ведёт себя прогон
    без `--slides`.

    Материалы к лекции задаются `--slides` (файл или каталог, флаг можно
    повторять): страницы презентации растрируются в `--slides-dpi`, а
    фотографии доски копируются как есть; и то и другое описывается той же
    vision-моделью, что кадры. Описания уходят в синтез отдельными разделами:
    страницы — «Слайды презентации» с меткой «слайд N», снимки — «Фотографии
    доски». Автоподбор презентации по имени записи не делается: рядом с
    записью лежит PDF, собранный самим absorbent из предыдущего конспекта.

    Этапы выбирает кэш, а не флаги: признак готового этапа — его артефакт
    (transcript.json, keyframes.json, descriptions.json,
    slide_descriptions.json, notes.md), и без `--force` такой этап не
    пересобирается. Пересобрать один этап — удалить его артефакт; полный
    перезапуск даёт `--force`, он же нужен, если сменились настройки этапа
    (например, `--fps` или `--output-language`). Материалы принимаются по
    подписи источника в каталоге `slides/` (путь, размер, mtime и dpi),
    поэтому правка колоды, замена снимка или смена `--slides-dpi` заставляют
    подготовить материал заново. Постпроцессинг пропускается, когда HTML и
    PDF не старше notes.md, и отключается флагом `--skip-render`. В синтез
    уходят описания только тех картинок, что подготовлены в этом прогоне.

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
    html_path = notes_path.with_suffix(".html")
    pdf_path = notes_path.with_suffix(".pdf")
    frames_dir = cache / "frames"
    keyframes_path = cache / "keyframes.json"
    descriptions_path = cache / "descriptions.json"
    slides_dir = cache / "slides"
    slide_descriptions_path = cache / "slide_descriptions.json"
    # Конвейер заводит только каталоги результата и кэша: каталоги картинок
    # (frames/ и images/) создают сами этапы кадров и материалов, и только
    # тогда, когда картинки есть, поэтому аудиовход без слайдов не оставляет
    # пустых каталогов.
    for directory in (output_dir, cache):
        directory.mkdir(parents=True, exist_ok=True)

    output_language = args.output_language
    min_scene_seconds = args.min_scene_seconds

    print(f"[{stem}] извлечение кадров...")
    keyframes: list[dict] = []
    with stage("extract_frames") as status:
        video = has_video_stream(input_path)
        if video:
            if not args.force and any(frames_dir.glob("frame_*.jpg")):
                status.skip()
            frames = extract_frames(input_path, frames_dir, args.fps, force=args.force)
        else:
            status.skip()
            print(f"[{stem}] кадров нет (аудио без видеопотока)")
    with stage("select_keyframes") as status:
        if not video:
            status.skip()
        elif args.force or not keyframes_path.exists():
            keyframes = select_keyframes(frames, images_dir, args.fps, args.hash_threshold,
                                         min_scene_seconds)
            keyframes_path.write_text(json.dumps(keyframes, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
        else:
            status.skip()
            keyframes = json.loads(keyframes_path.read_text(encoding="utf-8"))

    slides: list[dict] = []
    with stage("prepare_slides") as status:
        if args.slides:
            print(f"[{stem}] подготовка материалов к лекции: {len(args.slides)}...")
            slides = prepare_slides(args.slides, images_dir, slides_dir,
                                    args.slides_dpi, force=args.force,
                                    max_pages=getattr(args, "max_material_pages", None))
        else:
            status.skip()

    print(f"[{stem}] распознавание речи...")
    with stage("stt"):
        stt_provider, stt_spec = cfg.resolve("stt", args.stt)
        run_stt(stt_spec, stt_provider, input_path, args.language, transcript_path, cache,
                force=args.force)
        transcript = _load_json(transcript_path)

    descriptions: dict = {}
    slide_descriptions: dict = {}
    if keyframes or slides:
        vision_provider, vision_spec = cfg.resolve("vision", args.vision_model)
    with stage("describe_keyframes") as status:
        if keyframes:
            print(f"[{stem}] описание {len(keyframes)} ключевых кадров...")
            descriptions = describe_keyframes(vision_spec, vision_provider, keyframes, transcript,
                                              images_dir, output_language, descriptions_path,
                                              force=args.force)
        else:
            status.skip()
    with stage("describe_slides") as status:
        if slides:
            print(f"[{stem}] описание материалов к лекции: {len(slides)}...")
            slide_descriptions = describe_slides(vision_spec, vision_provider, slides, images_dir,
                                                 output_language, slide_descriptions_path,
                                                 force=args.force)
        else:
            status.skip()
    # Описания картинок прошлого отбора в синтез не идут: иначе модель вставит
    # ссылки на чужие файлы, которых на диске нет.
    descriptions = _filter_by_images(descriptions, {kf["image"] for kf in keyframes})
    slide_descriptions = _filter_by_images(slide_descriptions,
                                           {entry["image"] for entry in slides})

    with stage("synthesize") as status:
        if notes_path.exists() and not args.force:
            status.skip()
            print(f"[{stem}] синтез пропущен: конспект уже есть (--force для пересборки)")
        else:
            print(f"[{stem}] синтез Markdown...")
            llm_provider, llm_spec = cfg.resolve("llm", args.llm_model)
            run_synthesize(llm_spec, llm_provider, transcript, descriptions, output_language,
                           notes_path, images_prefix=images_prefix, slides=slide_descriptions)

    with stage("render") as status:
        if not args.skip_render and notes_path.exists():
            if args.force or not outputs_up_to_date(notes_path, html_path, pdf_path):
                print(f"[{stem}] постпроцессинг: таблицы, HTML, PDF...")
                render_all(notes_path, html_path, pdf_path, lang=output_language)
            else:
                status.skip()
            print(f"[{stem}] готово: {notes_path} (+ {html_path.name}, {pdf_path.name})")
        else:
            status.skip()
    return notes_path
