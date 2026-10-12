"""Оркестрация конвейера: файл -> конспект Markdown."""

from __future__ import annotations

import argparse
import ipaddress
import json
from pathlib import Path
from urllib.parse import urlsplit

from absorbent.config import Config, ModelSpec, Provider, batch_size_for
from absorbent.dedup import select_keyframes
from absorbent.describe import describe_keyframes, describe_slides
from absorbent.events import stage
from absorbent.extract import extract_frames, has_video_stream
from absorbent.parallel import start_in_thread
from absorbent.render import outputs_up_to_date, render_all
from absorbent.slides import prepare_slides
from absorbent.summarize import run_summarize
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


def _local_provider(provider: Provider) -> bool:
    """Провайдер на этой же машине: его модель делит видеопамять с whisper."""
    host = urlsplit(provider.base_url).hostname
    if host is None:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _may_describe_slides_early(stt_spec: ModelSpec, vision_provider: Provider) -> bool:
    """Можно ли описывать материалы, пока идёт распознавание речи.

    Облачная vision-модель видеокарту не занимает вовсе, поэтому её вызовы
    идут параллельно whisper. Локальная поднимает свои веса на той же карте и
    должна подождать: иначе llama.cpp делит память с уже загруженным whisper.
    """
    return not (_local_provider(vision_provider) and stt_spec.backend == "whisper")


def _run_stt_stage(stt_spec: ModelSpec, stt_provider: Provider, input_path: Path, language: str,
                   transcript_path: Path, cache: Path, force: bool) -> str | None:
    """Распознать речь в отдельном потоке: этап журнала живёт вместе с потоком.

    Свой `stage('stt')` открывается внутри потока, поэтому длительность этапа в
    журнале — это длительность распознавания, а не всего перекрытия с
    подготовкой кадров.
    """
    with stage("stt"):
        return run_stt(stt_spec, stt_provider, input_path, language, transcript_path, cache,
                       force=force)


def process_file(input_path: Path, cfg: Config, args: argparse.Namespace) -> Path:
    """Прогнать все этапы конвейера для одного файла; вернуть путь конспекта.

    Раскладка по умолчанию: каталог <имя>/ рядом с входом, внутри notes.md,
    images/ и .absorbent/ с кэшем. Плоская раскладка (--flat): <имя>.md,
    <имя>.json и <имя>.images/ ложатся рядом с медиафайлом, а кэш — в
    <каталог входа>/.absorbent/<имя>/.

    Порядок этапов: распознавание речи идёт одновременно с подготовкой кадров
    и материалов (извлечение кадров -> отбор ключевых кадров -> растеризация
    презентаций и копирование фотографий доски), затем выжимка транскрипта ->
    описание кадров и материалов -> синтез Markdown -> постпроцессинг
    (таблицы, HTML, PDF). Вход без видеопотока (аудиофайл) пропускает все
    этапы кадров: остаётся транскрипт и конспект без иллюстраций, а каталогов
    кадров (frames/ и images/) не появляется вовсе — так же ведёт себя прогон
    без `--slides`.

    Параллелизм нужен там, где этапы не зависят друг от друга. Whisper считает
    минуты, а ffmpeg и pdftoppm — секунды, поэтому распознавание речи уходит в
    отдельный поток, и ожидание речи не удлиняет прогон. Описание картинок
    идёт окнами по `batch_size` выбранной модели (значение модели перекрывает
    значение провайдера, см. `config.batch_size_for`). Материалы не зависят от
    речи вовсе, поэтому при облачной vision-модели они описываются ещё до
    конца распознавания; локальная модель этого не делает — её веса поднимаются
    на карте, занятой whisper (`_may_describe_slides_early`).

    Описание кадров опирается на речь: каждый кадр получает выжимку всей
    лекции (отдельный вызов LLM, этап `summarize`) и окно транскрипта ±60
    секунд вокруг своего таймкода.

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
    summary_path = cache / "summary.md"
    # Конвейер заводит только каталоги результата и кэша: каталоги картинок
    # (frames/ и images/) создают сами этапы кадров и материалов, и только
    # тогда, когда картинки есть, поэтому аудиовход без слайдов не оставляет
    # пустых каталогов.
    for directory in (output_dir, cache):
        directory.mkdir(parents=True, exist_ok=True)

    output_language = args.output_language
    min_scene_seconds = args.min_scene_seconds

    keyframes: list[dict] = []
    slides: list[dict] = []
    descriptions: dict = {}
    slide_descriptions: dict = {}
    slides_described = False
    vision_provider: Provider | None = None
    vision_spec: ModelSpec | None = None
    vision_batch_size = 1

    print(f"[{stem}] распознавание речи...")
    stt_provider, stt_spec = cfg.resolve("stt", args.stt)
    # Распознавание речи занимает минуты, а извлечение кадров, их отбор и
    # растеризация материалов — секунды, поэтому они идут одновременно:
    # whisper работает в отдельном потоке, а главный поток готовит картинки.
    stt_future = start_in_thread(_run_stt_stage, stt_spec, stt_provider, input_path,
                                 args.language, transcript_path, cache, args.force)
    print(f"[{stem}] извлечение кадров...")
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
    with stage("prepare_slides") as status:
        if args.slides:
            print(f"[{stem}] подготовка материалов к лекции: {len(args.slides)}...")
            slides = prepare_slides(args.slides, images_dir, slides_dir, args.slides_dpi,
                                    force=args.force,
                                    max_pages=getattr(args, "max_material_pages", None))
        else:
            status.skip()
    if keyframes or slides:
        vision_provider, vision_spec = cfg.resolve("vision", args.vision_model)
        vision_batch_size = batch_size_for(vision_provider, vision_spec)
    # Материалы не зависят от речи, поэтому при облачной vision-модели их
    # описание идёт, пока whisper ещё считает, и распознавание речи перестаёт
    # быть единственным длинным этапом прогона.
    if slides and _may_describe_slides_early(stt_spec, vision_provider):
        print(f"[{stem}] описание материалов к лекции: {len(slides)}...")
        with stage("describe_slides"):
            slide_descriptions = describe_slides(vision_spec, vision_provider, slides, images_dir,
                                                 output_language, slide_descriptions_path,
                                                 force=args.force, batch_size=vision_batch_size)
        slides_described = True
    # Ошибка распознавания всплывает здесь и прекращает прогон: кадры без речи
    # описывать не по чему.
    stt_future.result()
    transcript = _load_json(transcript_path)

    summary = ""
    with stage("summarize") as status:
        if transcript:
            print(f"[{stem}] выжимка транскрипта...")
            llm_provider, llm_spec = cfg.resolve("llm", args.llm_model)
            summary = run_summarize(llm_spec, llm_provider, transcript, output_language,
                                    summary_path, transcript_path=transcript_path,
                                    force=args.force)
        else:
            status.skip()
            print(f"[{stem}] выжимка транскрипта не нужна: транскрипт пуст")

    with stage("describe_keyframes") as status:
        if keyframes:
            print(f"[{stem}] описание {len(keyframes)} ключевых кадров...")
            descriptions = describe_keyframes(vision_spec, vision_provider, keyframes, transcript,
                                              images_dir, output_language, descriptions_path,
                                              force=args.force, batch_size=vision_batch_size,
                                              summary=summary)
        else:
            status.skip()
    if not slides:
        with stage("describe_slides") as status:
            status.skip()
    elif not slides_described:
        print(f"[{stem}] описание материалов к лекции: {len(slides)}...")
        with stage("describe_slides"):
            slide_descriptions = describe_slides(vision_spec, vision_provider, slides, images_dir,
                                                 output_language, slide_descriptions_path,
                                                 force=args.force, batch_size=vision_batch_size)
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
