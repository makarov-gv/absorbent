"""Тесты оркестрации конвейера: порядок этапов, кэш, --force и --skip-render."""

from __future__ import annotations

import json
import os
from concurrent.futures import Future
from pathlib import Path

import pytest

from absorbent.config import ModelSpec, Provider, load_config
from absorbent.pipeline import (_load_json, _local_provider, _may_describe_slides_early,
                                process_file)


@pytest.fixture
def cfg(minimal_config: Path):
    return load_config(str(minimal_config))


@pytest.fixture
def fake_stages(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Подменяет функции этапов записывающими заглушками.

    Фоновый поток распознавания подменён синхронным `start_in_thread`: порядок
    обращений становится детерминированным, а само перекрытие этапов проверяется
    отдельно (`_may_describe_slides_early`, tests/test_parallel.py).
    """
    import absorbent.pipeline as pipeline_module

    state: dict = {"calls": []}

    def _start_in_thread(function, *args):
        state["calls"].append("thread")
        future: Future = Future()
        try:
            future.set_result(function(*args))
        except BaseException as exc:  # ошибка этапа доезжает через Future, как из потока.
            future.set_exception(exc)
        return future

    def _extract(input_path, frames_dir, fps, force=False):
        state["calls"].append("extract")
        frames_dir.mkdir(parents=True, exist_ok=True)
        for i in range(3):
            (frames_dir / f"frame_{i+1:06d}.jpg").write_bytes(b"jpeg")
        return sorted(frames_dir.glob("frame_*.jpg"))

    def _select(frames, images_dir, fps, threshold, min_seconds):
        state["calls"].append(f"select:{threshold}:{min_seconds}")
        images_dir.mkdir(parents=True, exist_ok=True)
        (images_dir / "frame_00_00_00.jpg").write_bytes(b"jpeg")
        return [{"image": "frame_00_00_00.jpg", "seconds": 0.0, "timestamp": "00:00:00"}]

    def _run_stt(spec, provider, input_path, language, out_json, work_dir, force=False):
        state["calls"].append(f"stt:{language}")
        out_json.write_text(json.dumps([{"start": 0.0, "end": 1.0, "text": "привет",
                                         "speaker": None}]), encoding="utf-8")
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "language.txt").write_text("ru", encoding="utf-8")
        return "ru"

    def _summarize(llm_spec, provider, transcript, output_language, out_md,
                   transcript_path=None, force=False):
        state["calls"].append(f"summarize:{output_language}")
        state["summary_transcript"] = transcript
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text("выжимка лекции", encoding="utf-8")
        return "выжимка лекции"

    def _describe(vision_spec, provider, keyframes, transcript, images_dir, output_language,
                  out_json, force=False, *, batch_size=1, summary=""):
        state["calls"].append(f"describe:{output_language}:{len(keyframes)}")
        state["describe_batch"] = batch_size
        state["describe_summary"] = summary
        data = {kf["image"]: {"timestamp": kf["timestamp"], "description": f"кадр {kf['image']}"}
                for kf in keyframes}
        out_json.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return data

    def _prepare_slides(sources, images_dir, work_dir, dpi, force=False, *, max_pages=None):
        state["calls"].append("slides")
        state["slides_dpi"] = dpi
        images_dir.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)
        entries = []
        for index, source in enumerate(sources, start=1):
            name = f"slide_{index:03d}.png"
            (images_dir / name).write_bytes(b"png")
            entries.append({"image": name, "page": index, "source": str(source),
                            "source_page": index})
        return entries

    def _describe_slides(vision_spec, provider, slides, images_dir, output_language, out_json,
                         force=False, *, batch_size=1):
        state["calls"].append(f"slides:{output_language}:{len(slides)}")
        state["slides_batch"] = batch_size
        data = {entry["image"]: {"page": entry["page"], "source": entry["source"],
                                 "description": f"слайд {entry['image']}"} for entry in slides}
        out_json.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return data

    def _synthesize(llm_spec, provider, transcript, descriptions, output_language, out_md,
                    images_prefix="images/", slides=None):
        state["calls"].append(f"synthesize:{output_language}:{len(descriptions)}")
        state["images_prefix"] = images_prefix
        state["slides"] = slides
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text("# Конспект\n", encoding="utf-8")
        return out_md

    def _render(notes_md, out_html=None, out_pdf=None, lang=None):
        state["calls"].append("render")
        state["render_paths"] = (out_html, out_pdf)
        state["render_lang"] = lang
        # Постпроцессинг в жизни создаёт обе версии: на их наличие и mtime
        # опирается пропуск уже собранного (см. render.outputs_up_to_date).
        for path in (out_html, out_pdf):
            path.write_text("готово", encoding="utf-8")
        return out_html, out_pdf

    def _has_video(input_path):
        # Наличие видеопотока не является этапом конвейера: в state["calls"]
        # его нет, порядок этапов остаётся прежним.
        return state.get("video", True)

    monkeypatch.setattr(pipeline_module, "has_video_stream", _has_video)
    monkeypatch.setattr(pipeline_module, "extract_frames", _extract)
    monkeypatch.setattr(pipeline_module, "select_keyframes", _select)
    monkeypatch.setattr(pipeline_module, "run_stt", _run_stt)
    monkeypatch.setattr(pipeline_module, "start_in_thread", _start_in_thread)
    monkeypatch.setattr(pipeline_module, "run_summarize", _summarize)
    monkeypatch.setattr(pipeline_module, "describe_keyframes", _describe)
    monkeypatch.setattr(pipeline_module, "prepare_slides", _prepare_slides)
    monkeypatch.setattr(pipeline_module, "describe_slides", _describe_slides)
    monkeypatch.setattr(pipeline_module, "run_synthesize", _synthesize)
    monkeypatch.setattr(pipeline_module, "render_all", _render)
    state["module"] = pipeline_module
    return state


def _input(tmp_path: Path, name: str = "Лекция.mkv") -> Path:
    path = tmp_path / name
    path.write_bytes(b"media")
    return path


class TestProcessFile:
    def test_full_pipeline_layout_and_stages(self, tmp_path: Path, cfg, fake_stages: dict,
                                             args_namespace) -> None:
        input_path = _input(tmp_path)
        notes = process_file(input_path, cfg, args_namespace)
        # Раскладка по умолчанию: <stem>/ рядом с входом.
        out_dir = tmp_path / "Лекция"
        assert notes == out_dir / "notes.md"
        assert notes.exists() and notes.read_text(encoding="utf-8") == "# Конспект\n"
        assert (out_dir / "images" / "frame_00_00_00.jpg").exists()
        assert (out_dir / ".absorbent" / "frames").is_dir()
        for name in ("keyframes.json", "transcript.json", "descriptions.json"):
            assert (out_dir / ".absorbent" / name).exists()
        # Порядок этапов: распознавание речи стартует первым (в отдельном
        # потоке), кадры и материалы готовятся параллельно ему, выжимка
        # транскрипта идёт после распознавания.
        assert fake_stages["calls"] == [
            "thread", "stt:auto", "extract", "select:10:5.0", "summarize:ru",
            "describe:ru:1", "synthesize:ru:1", "render",
        ]

    def test_custom_output_dir_and_languages(self, tmp_path: Path, cfg, fake_stages: dict,
                                             args_namespace) -> None:
        args_namespace.output_dir = str(tmp_path / "custom")
        args_namespace.language = "en"  # подсказка whisper.
        args_namespace.output_language = "en"  # язык конспекта.
        input_path = _input(tmp_path)
        process_file(input_path, cfg, args_namespace)
        assert (tmp_path / "custom" / "notes.md").exists()
        assert "stt:en" in fake_stages["calls"]
        assert "describe:en:1" in fake_stages["calls"]
        assert "synthesize:en:1" in fake_stages["calls"]
        # Язык вёрстки конспекта следует за языком результата, а не за подсказкой whisper.
        assert fake_stages["render_lang"] == "en"
        assert (tmp_path / "custom" / ".absorbent" / "language.txt").read_text(encoding="utf-8") == "ru"

    def test_second_run_reuses_cache(self, tmp_path: Path, cfg, fake_stages: dict,
                                     args_namespace) -> None:
        input_path = _input(tmp_path)
        process_file(input_path, cfg, args_namespace)
        assert (tmp_path / "Лекция" / "notes.html").exists()
        assert (tmp_path / "Лекция" / "notes.pdf").exists()
        fake_stages["calls"].clear()
        process_file(input_path, cfg, args_namespace)
        # Без force отбор ключевых кадров пропускается (есть keyframes.json),
        # extract_frames/run_stt/summarize/describe вызываются, но возвращают
        # кэш внутри; синтез и постпроцессинг пропускаются целиком: notes.md на
        # месте, а HTML и PDF не старше него.
        assert fake_stages["calls"] == ["thread", "stt:auto", "extract", "summarize:ru",
                                        "describe:ru:1"]

    def test_synthesis_from_cache_kept(self, tmp_path: Path, cfg, fake_stages: dict,
                                       args_namespace, capsys: pytest.CaptureFixture) -> None:
        """Готовый конспект не пересобирается: правка notes.md вручную переживает прогон."""
        input_path = _input(tmp_path)
        notes = process_file(input_path, cfg, args_namespace)
        notes.write_text("# Правленый конспект\n", encoding="utf-8")
        fake_stages["calls"].clear()
        process_file(input_path, cfg, args_namespace)
        assert "synthesize" not in " ".join(fake_stages["calls"])
        assert notes.read_text(encoding="utf-8") == "# Правленый конспект\n"
        assert "синтез пропущен" in capsys.readouterr().out

    def test_render_reruns_when_notes_md_is_newer(self, tmp_path: Path, cfg, fake_stages: dict,
                                                 args_namespace) -> None:
        """Правка notes.md без --force пересобирает HTML и PDF."""
        input_path = _input(tmp_path)
        notes = process_file(input_path, cfg, args_namespace)
        stale = (tmp_path / "Лекция" / "notes.html").stat().st_mtime
        os.utime(notes, (stale + 10, stale + 10))
        fake_stages["calls"].clear()
        process_file(input_path, cfg, args_namespace)
        assert fake_stages["calls"] == ["thread", "stt:auto", "extract", "summarize:ru",
                                        "describe:ru:1", "render"]

    def test_force_reruns_all(self, tmp_path: Path, cfg, fake_stages: dict, args_namespace) -> None:
        args_namespace.force = True
        input_path = _input(tmp_path)
        process_file(input_path, cfg, args_namespace)
        assert fake_stages["calls"] == ["thread", "stt:auto", "extract", "select:10:5.0",
                                        "summarize:ru", "describe:ru:1", "synthesize:ru:1",
                                        "render"]

    def test_deleted_notes_md_rebuilds_synthesis_only(self, tmp_path: Path, cfg, fake_stages: dict,
                                                      args_namespace) -> None:
        """Пересборка одного этапа — удаление его артефакта: остальное берёт кэш."""
        input_path = _input(tmp_path)
        notes = process_file(input_path, cfg, args_namespace)
        notes.unlink()
        fake_stages["calls"].clear()
        process_file(input_path, cfg, args_namespace)
        # Кадры не отбираются заново (keyframes.json на месте), распознавание,
        # выжимка и описание идут по кэшу, синтез и постпроцессинг отрабатывают.
        assert fake_stages["calls"] == ["thread", "stt:auto", "extract", "summarize:ru",
                                        "describe:ru:1", "synthesize:ru:1", "render"]

    def test_hash_threshold_passed(self, tmp_path: Path, cfg, fake_stages: dict,
                                   args_namespace) -> None:
        args_namespace.hash_threshold = 20
        args_namespace.min_scene_seconds = 2.0
        args_namespace.force = True
        input_path = _input(tmp_path)
        process_file(input_path, cfg, args_namespace)
        assert "select:20:2.0" in fake_stages["calls"]

    def test_skip_render_skips_postprocess(self, tmp_path: Path, cfg, fake_stages: dict,
                                           args_namespace) -> None:
        args_namespace.skip_render = True
        input_path = _input(tmp_path)
        process_file(input_path, cfg, args_namespace)
        # Синтез отработал, но постпроцессинг не вызван.
        assert "render" not in fake_stages["calls"]

    def test_synthesis_error_skips_render(self, tmp_path: Path, cfg, fake_stages: dict,
                                          args_namespace, monkeypatch: pytest.MonkeyPatch) -> None:
        """Пустой ответ LLM: синтез падает, HTML и PDF не собираются."""
        import absorbent.pipeline as pipeline_module
        from absorbent.synthesize import SynthesisError

        def _fail(*args, **kwargs):
            raise SynthesisError("модель не вернула текст конспекта")

        monkeypatch.setattr(pipeline_module, "run_synthesize", _fail)
        with pytest.raises(SynthesisError):
            process_file(_input(tmp_path), cfg, args_namespace)
        assert "render" not in fake_stages["calls"]


class TestHelpers:
    def test_load_json_missing_returns_empty_list(self, tmp_path: Path) -> None:
        assert _load_json(tmp_path / "nope.json") == []

    def test_load_json_existing(self, tmp_path: Path) -> None:
        path = tmp_path / "data.json"
        path.write_text('[{"a": 1}]', encoding="utf-8")
        assert _load_json(path) == [{"a": 1}]

    def test_stale_frame_descriptions_dropped(self, tmp_path: Path, cfg, fake_stages: dict,
                                              args_namespace) -> None:
        """Кэш описаний прошлого отбора не доходит до синтеза."""
        import absorbent.pipeline as pipeline_module

        seen: dict = {}

        def _synthesize(llm_spec, provider, transcript, descriptions, output_language, out_md,
                        images_prefix="images/", slides=None):
            seen["descriptions"] = descriptions
            out_md.parent.mkdir(parents=True, exist_ok=True)
            out_md.write_text("x", encoding="utf-8")
            return out_md

        def _describe_cached(vision_spec, provider, keyframes, transcript, images_dir,
                             output_language, out_json, force=False, *, batch_size=1, summary=""):
            # Как настоящий кэш описаний: отдаёт и кадр прошлого отбора.
            return {kf["image"]: {"timestamp": kf["timestamp"], "description": "текущий"}
                    for kf in keyframes} | {
                "frame_00_00_09.jpg": {"timestamp": "00:00:09", "description": "прошлый отбор"}}

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(pipeline_module, "describe_keyframes", _describe_cached)
        monkeypatch.setattr(pipeline_module, "run_synthesize", _synthesize)
        process_file(_input(tmp_path), cfg, args_namespace)
        assert seen["descriptions"] == {
            "frame_00_00_00.jpg": {"timestamp": "00:00:00", "description": "текущий"}}
        monkeypatch.undo()


class TestFlatMode:
    """Плоская раскладка (--flat): результат рядом с медиафайлом, кэш в .absorbent/."""

    def test_flat_layout_and_prefix(self, tmp_path: Path, cfg, fake_stages: dict,
                                    args_namespace, capsys: pytest.CaptureFixture) -> None:
        args_namespace.flat = True
        input_path = _input(tmp_path, "lecture.mp4")
        notes = process_file(input_path, cfg, args_namespace)

        assert notes == tmp_path / "lecture.md"
        assert notes.read_text(encoding="utf-8") == "# Конспект\n"
        assert (tmp_path / "lecture.json").exists()
        assert (tmp_path / "lecture.images" / "frame_00_00_00.jpg").exists()
        cache = tmp_path / ".absorbent" / "lecture"
        assert (cache / "frames").is_dir()
        assert (cache / "language.txt").read_text(encoding="utf-8") == "ru"
        # Каталога <имя>/ с notes.md в плоском режиме нет.
        assert not (tmp_path / "lecture").exists()
        # Префикс картинок для синтеза — <имя>.images/, а не images/.
        assert fake_stages["images_prefix"] == "lecture.images/"
        # Постпроцессинг получает явные пути и печатает итог с их именами.
        assert fake_stages["render_paths"] == (tmp_path / "lecture.html", tmp_path / "lecture.pdf")
        out = capsys.readouterr().out
        assert f"готово: {notes} (+ lecture.html, lecture.pdf)" in out

    def test_flat_with_output_dir(self, tmp_path: Path, cfg, fake_stages: dict,
                                  args_namespace) -> None:
        """--flat --output-dir: результат в DIR, кэш всё равно у каталога входа."""
        args_namespace.flat = True
        args_namespace.output_dir = str(tmp_path / "res")
        input_path = _input(tmp_path, "lecture.mp4")
        notes = process_file(input_path, cfg, args_namespace)
        assert notes == tmp_path / "res" / "lecture.md"
        assert (tmp_path / "res" / "lecture.json").exists()
        assert (tmp_path / "res" / "lecture.images").is_dir()
        assert (tmp_path / ".absorbent" / "lecture" / "language.txt").exists()
        assert fake_stages["images_prefix"] == "lecture.images/"

    def test_default_layout_passes_render_paths(self, tmp_path: Path, cfg, fake_stages: dict,
                                                args_namespace, capsys: pytest.CaptureFixture) -> None:
        process_file(_input(tmp_path), cfg, args_namespace)
        assert fake_stages["images_prefix"] == "images/"
        assert fake_stages["render_paths"] == (tmp_path / "Лекция" / "notes.html",
                                               tmp_path / "Лекция" / "notes.pdf")
        assert "(+ notes.html, notes.pdf)" in capsys.readouterr().out

    def test_audio_input_skips_frame_stages(self, tmp_path: Path, cfg, fake_stages: dict,
                                           args_namespace, capsys: pytest.CaptureFixture) -> None:
        """Аудиофайл без видеопотока: кадры не извлекаются, конспект без картинок."""
        fake_stages["video"] = False
        args_namespace.flat = True
        notes = process_file(_input(tmp_path, "lecture.mp3"), cfg, args_namespace)
        assert "extract" not in fake_stages["calls"]
        assert not any(c.startswith("select") for c in fake_stages["calls"])
        assert not any(c.startswith("describe") for c in fake_stages["calls"])
        assert fake_stages["calls"] == ["thread", "stt:auto", "summarize:ru",
                                        "synthesize:ru:0", "render"]
        assert "кадров нет (аудио без видеопотока)" in capsys.readouterr().out
        assert notes.exists()

    def test_audio_input_leaves_no_image_dirs(self, tmp_path: Path, cfg, fake_stages: dict,
                                              args_namespace) -> None:
        """У аудиовхода не появляется каталогов кадров — ни images/, ни frames/."""
        fake_stages["video"] = False
        process_file(_input(tmp_path, "lecture.mp3"), cfg, args_namespace)
        out_dir = tmp_path / "lecture"
        assert not (out_dir / "images").exists()
        assert not (out_dir / ".absorbent" / "frames").exists()
        # Служебный кэш при этом на месте: транскрипт и язык пишутся туда.
        assert (out_dir / ".absorbent" / "language.txt").exists()


class TestSlides:
    """Этап презентаций: порядок, отсев чужого кэша, аудиовход."""

    def _deck(self, tmp_path: Path, name: str = "deck.pdf") -> Path:
        path = tmp_path / name
        path.write_bytes(b"%PDF")
        return path

    def test_stage_order_and_synthesis_input(self, tmp_path: Path, cfg, fake_stages: dict,
                                             args_namespace) -> None:
        deck = self._deck(tmp_path)
        args_namespace.slides = [deck]
        process_file(_input(tmp_path), cfg, args_namespace)
        # Материалы растеризуются, пока идёт распознавание речи, и описываются
        # сразу после: облачная vision-модель GPU не занимает, поэтому ждать
        # whisper ей не нужно.
        assert fake_stages["calls"] == [
            "thread", "stt:auto", "extract", "select:10:5.0", "slides", "slides:ru:1",
            "summarize:ru", "describe:ru:1", "synthesize:ru:1", "render",
        ]
        # Описания страниц уходят в синтез вместе с описаниями кадров.
        assert set(fake_stages["slides"]) == {"slide_001.png"}
        assert (tmp_path / "Лекция" / "images" / "slide_001.png").exists()
        assert (tmp_path / "Лекция" / ".absorbent" / "slide_descriptions.json").exists()

    def test_stale_slide_descriptions_do_not_reach_synthesis(self, tmp_path: Path, cfg,
                                                            fake_stages: dict,
                                                            args_namespace,
                                                            capsys: pytest.CaptureFixture) -> None:
        """Материалы прошлого прогона не всплывают, если слайды не переданы."""
        fake_stages["video"] = False
        cache = tmp_path / "lecture" / ".absorbent"
        cache.mkdir(parents=True)
        (cache / "slide_descriptions.json").write_text(
            json.dumps({"slide_001.png": {"page": 1, "source": "/old/deck.pdf",
                                          "description": "прошлый прогон"}}, ensure_ascii=False),
            encoding="utf-8")
        process_file(_input(tmp_path, "lecture.mp3"), cfg, args_namespace)
        assert fake_stages["slides"] == {}
        assert "synthesize:ru:0" in fake_stages["calls"]
        assert "подготовка материалов" not in capsys.readouterr().out

    def test_audio_with_slides_has_no_frame_dirs(self, tmp_path: Path, cfg, fake_stages: dict,
                                                 args_namespace) -> None:
        """Аудиовход со слайдами: картинки слайдов есть, каталогов кадров нет."""
        fake_stages["video"] = False
        deck = self._deck(tmp_path)
        args_namespace.slides = [deck]
        process_file(_input(tmp_path, "lecture.mp3"), cfg, args_namespace)
        out_dir = tmp_path / "lecture"
        assert fake_stages["calls"] == ["thread", "stt:auto", "slides", "slides:ru:1",
                                        "summarize:ru", "synthesize:ru:0", "render"]
        assert (out_dir / "images" / "slide_001.png").exists()
        assert not (out_dir / ".absorbent" / "frames").exists()

    def test_slides_dpi_reaches_stage(self, tmp_path: Path, cfg, fake_stages: dict,
                                      args_namespace) -> None:
        args_namespace.slides = [self._deck(tmp_path)]
        args_namespace.slides_dpi = 300
        process_file(_input(tmp_path), cfg, args_namespace)
        assert fake_stages["slides_dpi"] == 300

    def test_stale_slide_descriptions_dropped(self, tmp_path: Path, cfg, fake_stages: dict,
                                              args_namespace) -> None:
        """Кэш описаний прошлого набора страниц не доходит до синтеза."""
        import absorbent.pipeline as pipeline_module

        deck = self._deck(tmp_path)
        args_namespace.slides = [deck]

        def _describe_cached(vision_spec, provider, slides, images_dir, output_language, out_json,
                             force=False, *, batch_size=1):
            # Как настоящий кэш страниц: отдаёт и страницу прошлого отбора.
            return {entry["image"]: {"page": entry["page"], "source": entry["source"],
                                     "description": "текущая"} for entry in slides} | {
                "slide_009.png": {"page": 9, "source": str(deck), "description": "прошлый отбор"}}

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(pipeline_module, "describe_slides", _describe_cached)
        process_file(_input(tmp_path), cfg, args_namespace)
        assert fake_stages["slides"] == {
            "slide_001.png": {"page": 1, "source": str(deck), "description": "текущая"}}
        monkeypatch.undo()


def test_journal_audio_stage_boundaries(tmp_path: Path, cfg, fake_stages: dict, args_namespace) -> None:
    """Аудиовход: каждый этап закрывается, распознавание речи идёт до выжимки.

    Поток распознавания подменён синхронным (см. фикстуру), поэтому порядок
    событий детерминирован; настоящее перекрытие этапов проверяет
    `TestParallelLayout` и tests/test_parallel.py.
    """
    from absorbent.events import Journal

    fake_stages["video"] = False
    path = tmp_path / "events.jsonl"
    with Journal(path):
        process_file(_input(tmp_path, "lecture.wav"), cfg, args_namespace)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    starts = [row["stage"] for row in rows if row["event"] == "stage_start"]
    ends = [row for row in rows if row["event"] == "stage_end"]
    assert starts == [row["stage"] for row in ends] == [
        "stt", "extract_frames", "select_keyframes", "prepare_slides", "summarize",
        "describe_keyframes", "describe_slides", "synthesize", "render"]
    assert {row["stage"]: row["status"] for row in ends} == {
        "stt": "ok", "extract_frames": "skipped", "select_keyframes": "skipped",
        "prepare_slides": "skipped", "summarize": "ok", "describe_keyframes": "skipped",
        "describe_slides": "skipped", "synthesize": "ok", "render": "ok"}


def test_journal_cached_frame_selection_synthesis_and_render(tmp_path: Path, cfg, fake_stages: dict, args_namespace) -> None:
    from absorbent.events import Journal

    input_path = _input(tmp_path)
    process_file(input_path, cfg, args_namespace)
    path = tmp_path / "events.jsonl"
    with Journal(path):
        process_file(input_path, cfg, args_namespace)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    ends = {row["stage"]: row["status"] for row in rows if row["event"] == "stage_end"}
    assert {name: ends[name] for name in ("extract_frames", "select_keyframes", "synthesize", "render")} == {
        "extract_frames": "skipped", "select_keyframes": "skipped", "synthesize": "skipped", "render": "skipped"}


class TestLocalProvider:
    """Локальность провайдера: по ней решается, ждать ли whisper."""

    @pytest.mark.parametrize(("base_url", "expected"), [
        ("http://127.0.0.1:12434/v1", True),
        ("http://127.0.0.2:9/v1", True),
        ("http://localhost:12434/v1", True),
        ("http://[::1]:12434/v1", True),
        ("https://api.deepseek.com", False),
        ("http://192.168.1.5:12434/v1", False),
        ("", False),
    ])
    def test_local_provider(self, base_url: str, expected: bool) -> None:
        assert _local_provider(Provider(name="p", base_url=base_url)) is expected


class TestSlidesOverlap:
    """Описание материалов перекрывается с распознаванием речи не всегда."""

    def _vision(self, base_url: str) -> Provider:
        return Provider(name="v", base_url=base_url)

    @pytest.mark.parametrize(("backend", "base_url", "expected"), [
        ("whisper", "https://api.deepseek.com", True),   # облачная модель GPU не занимает.
        ("whisper", "http://127.0.0.1:12434/v1", False),  # whisper и локальная модель — одна карта.
        ("voxtral", "http://127.0.0.1:12434/v1", True),   # whisper не запускается вовсе.
        ("voxtral", "https://api.deepseek.com", True),
    ])
    def test_may_describe_slides_early(self, backend: str, base_url: str, expected: bool) -> None:
        spec = ModelSpec(name="stt", backend=backend)
        assert _may_describe_slides_early(spec, self._vision(base_url)) is expected


class TestParallelLayout:
    """Порядок этапов при разной локальности моделей и размер батча."""

    def _config(self, tmp_path: Path, *, stt: str, vision_local: bool, provider_batch: int = 1,
                model_batch: int | None = None, name: str = "config.yaml") -> Path:
        vision_base = "http://127.0.0.1:12434/v1" if vision_local else "https://api.example.test/v1"
        stt_block = ("whisper-local:\n    backend: whisper\n    whisper_model: tiny\n"
                     if stt == "whisper" else
                     "voxtral-cloud:\n    backend: voxtral\n    provider: mistral\n    model: v\n")
        model_line = "" if model_batch is None else f"    batch_size: {model_batch}\n"
        path = tmp_path / name
        path.write_text(f"""providers:
  vision:
    base_url: {vision_base}
    api_key_env: TEST_KEY_A
    batch_size: {provider_batch}
  mistral:
    base_url: https://api.mistral.ai/v1
    api_key_env: TEST_KEY_A
stt:
  {stt_block}    default: true
vision:
  vis-a:
    provider: vision
    default: true
{model_line}llm:
  llm-a:
    provider: vision
    default: true
""", encoding="utf-8")
        return path

    def _deck(self, tmp_path: Path) -> Path:
        deck = tmp_path / "deck.pdf"
        deck.write_bytes(b"%PDF")
        return deck

    def test_local_vision_waits_for_whisper(self, tmp_path: Path, fake_stages: dict,
                                            args_namespace) -> None:
        """Локальная vision-модель делит карту с whisper: материалы описываются после."""
        cfg_path = self._config(tmp_path, stt="whisper", vision_local=True)
        args_namespace.slides = [self._deck(tmp_path)]
        process_file(_input(tmp_path), load_config(str(cfg_path)), args_namespace)
        assert fake_stages["calls"] == [
            "thread", "stt:auto", "extract", "select:10:5.0", "slides", "summarize:ru",
            "describe:ru:1", "slides:ru:1", "synthesize:ru:1", "render",
        ]

    def test_local_vision_with_cloud_stt_describes_early(self, tmp_path: Path, fake_stages: dict,
                                                         args_namespace) -> None:
        """С облачным распознаванием карта свободна: материалы описываются сразу."""
        cfg_path = self._config(tmp_path, stt="voxtral", vision_local=True)
        args_namespace.slides = [self._deck(tmp_path)]
        process_file(_input(tmp_path), load_config(str(cfg_path)), args_namespace)
        assert fake_stages["calls"] == [
            "thread", "stt:auto", "extract", "select:10:5.0", "slides", "slides:ru:1",
            "summarize:ru", "describe:ru:1", "synthesize:ru:1", "render",
        ]

    def test_provider_batch_size_reaches_describe(self, tmp_path: Path, fake_stages: dict,
                                                  args_namespace) -> None:
        cfg_path = self._config(tmp_path, stt="whisper", vision_local=False, provider_batch=5)
        args_namespace.slides = [self._deck(tmp_path)]
        process_file(_input(tmp_path), load_config(str(cfg_path)), args_namespace)
        assert fake_stages["describe_batch"] == 5
        assert fake_stages["slides_batch"] == 5

    def test_model_batch_size_overrides_provider(self, tmp_path: Path, fake_stages: dict,
                                                 args_namespace) -> None:
        cfg_path = self._config(tmp_path, stt="whisper", vision_local=False, provider_batch=5,
                                model_batch=2)
        args_namespace.slides = [self._deck(tmp_path)]
        process_file(_input(tmp_path), load_config(str(cfg_path)), args_namespace)
        assert fake_stages["describe_batch"] == 2
        assert fake_stages["slides_batch"] == 2

    def test_summary_reaches_frame_prompt(self, tmp_path: Path, cfg, fake_stages: dict,
                                          args_namespace) -> None:
        """Выжимка транскрипта уходит в промпт каждого кадра."""
        process_file(_input(tmp_path), cfg, args_namespace)
        assert fake_stages["describe_summary"] == "выжимка лекции"
        assert fake_stages["summary_transcript"] == [{"start": 0.0, "end": 1.0, "text": "привет",
                                                     "speaker": None}]
        assert (tmp_path / "Лекция" / ".absorbent" / "summary.md").read_text(
            encoding="utf-8") == "выжимка лекции"

    def test_empty_transcript_skips_summary(self, tmp_path: Path, cfg, fake_stages: dict,
                                            args_namespace, monkeypatch: pytest.MonkeyPatch,
                                            capsys: pytest.CaptureFixture) -> None:
        """Без речи выжимка не выпрашивается: модель не вызывается вовсе."""

        def _empty_stt(spec, provider, input_path, language, out_json, work_dir, force=False):
            fake_stages["calls"].append(f"stt:{language}")
            out_json.write_text("[]", encoding="utf-8")
            work_dir.mkdir(parents=True, exist_ok=True)
            return None

        monkeypatch.setattr(fake_stages["module"], "run_stt", _empty_stt)
        process_file(_input(tmp_path), cfg, args_namespace)
        assert not any(call.startswith("summarize") for call in fake_stages["calls"])
        assert fake_stages["describe_summary"] == ""
        assert "выжимка транскрипта не нужна" in capsys.readouterr().out

    def test_stt_error_stops_after_frame_preparation(self, tmp_path: Path, cfg, fake_stages: dict,
                                                     args_namespace, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ошибка распознавания всплывает после подготовки кадров и роняет прогон."""
        from absorbent.transcribe import STTError

        def _failing_stt(spec, provider, input_path, language, out_json, work_dir, force=False):
            fake_stages["calls"].append(f"stt:{language}")
            raise STTError("модель распознавания недоступна")

        monkeypatch.setattr(fake_stages["module"], "run_stt", _failing_stt)
        with pytest.raises(STTError, match="недоступна"):
            process_file(_input(tmp_path), cfg, args_namespace)
        # Кадры успели подготовиться: они шли параллельно распознаванию речи.
        assert "extract" in fake_stages["calls"]
        assert "select:10:5.0" in fake_stages["calls"]
        assert not any(call.startswith(("describe", "summarize", "synthesize"))
                       for call in fake_stages["calls"])
