"""Тесты оркестрации конвейера: порядок этапов, кэш, skip-флаги, force."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from absorbent.config import load_config
from absorbent.pipeline import _load_json, process_file


@pytest.fixture
def cfg(minimal_config: Path):
    return load_config(str(minimal_config))


@pytest.fixture
def fake_stages(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Подменяет функции этапов записывающими заглушками."""
    import absorbent.pipeline as pipeline_module

    state: dict = {"calls": []}

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

    def _describe(vision_spec, provider, keyframes, transcript, images_dir, output_language,
                  out_json, force=False):
        state["calls"].append(f"describe:{output_language}:{len(keyframes)}")
        data = {kf["image"]: {"timestamp": kf["timestamp"], "description": f"кадр {kf['image']}"}
                for kf in keyframes}
        out_json.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return data

    def _synthesize(llm_spec, provider, transcript, descriptions, output_language, out_md,
                    images_prefix="images/"):
        state["calls"].append(f"synthesize:{output_language}:{len(descriptions)}")
        state["images_prefix"] = images_prefix
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
    monkeypatch.setattr(pipeline_module, "describe_keyframes", _describe)
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
        # Порядок этапов: язык речи (auto) и язык результата (ru) независимы.
        assert fake_stages["calls"] == [
            "extract", "select:10:5.0", "stt:auto", "describe:ru:1", "synthesize:ru:1", "render",
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
        # extract_frames/run_stt/describe вызываются, но возвращают кэш внутри;
        # синтез и постпроцессинг пропускаются целиком: notes.md на месте, а
        # HTML и PDF не старше него.
        assert fake_stages["calls"] == ["extract", "stt:auto", "describe:ru:1"]

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
        assert fake_stages["calls"] == ["extract", "stt:auto", "describe:ru:1", "render"]

    def test_force_reruns_all(self, tmp_path: Path, cfg, fake_stages: dict, args_namespace) -> None:
        args_namespace.force = True
        input_path = _input(tmp_path)
        process_file(input_path, cfg, args_namespace)
        assert fake_stages["calls"] == ["extract", "select:10:5.0", "stt:auto", "describe:ru:1",
                                        "synthesize:ru:1", "render"]

    def test_skip_all_flags(self, tmp_path: Path, cfg, fake_stages: dict, args_namespace) -> None:
        args_namespace.skip_stt = True
        args_namespace.skip_frames = True
        args_namespace.skip_describe = True
        args_namespace.skip_synthesize = True
        input_path = _input(tmp_path)
        notes = process_file(input_path, cfg, args_namespace)
        assert fake_stages["calls"] == []
        # Каталог результата и кэш созданы, notes.md отсутствует.
        assert notes == tmp_path / "Лекция" / "notes.md"
        assert notes.parent.is_dir()
        assert not notes.exists()

    def test_skip_stt_still_synthesizes(self, tmp_path: Path, cfg, fake_stages: dict,
                                        args_namespace) -> None:
        # Первый прогон кэширует transcript.json (синтез отключён), второй —
        # с --skip-stt: синтез работает по кэшу транскрипта, язык результата
        # не зависит от STT.
        args_namespace.skip_synthesize = True
        input_path = _input(tmp_path)
        process_file(input_path, cfg, args_namespace)
        fake_stages["calls"].clear()
        args_namespace.skip_synthesize = False
        args_namespace.skip_stt = True
        process_file(input_path, cfg, args_namespace)
        assert not any(c.startswith("stt:") for c in fake_stages["calls"])
        assert "describe:ru:1" in fake_stages["calls"]
        assert "synthesize:ru:1" in fake_stages["calls"]

    def test_skip_frames_with_cached_keyframes(self, tmp_path: Path, cfg, fake_stages: dict,
                                               args_namespace) -> None:
        # Второй прогон без извлечения кадров: из этапов остаётся только STT,
        # остальное берётся из кэша.
        input_path = _input(tmp_path)
        process_file(input_path, cfg, args_namespace)
        fake_stages["calls"].clear()
        args_namespace.skip_frames = True
        process_file(input_path, cfg, args_namespace)
        # Кадры не извлекаются; keyframes.json при этом не читается (skip),
        # поэтому describe не запускается, а готовые описания берутся из кэша.
        assert fake_stages["calls"] == ["stt:auto"]

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

    def test_describe_from_list_cache_resets(self, tmp_path: Path, cfg, fake_stages: dict,
                                             args_namespace) -> None:
        """Сломанный (списочный) descriptions.json при пустых кадрах -> {}."""
        import absorbent.pipeline as pipeline_module

        # Перехватываем синтез, чтобы проверить, какие описания туда доехали.
        seen: dict = {}

        def _synthesize(llm_spec, provider, transcript, descriptions, output_language, out_md,
                        images_prefix="images/"):
            seen["descriptions"] = descriptions
            out_md.parent.mkdir(parents=True, exist_ok=True)
            out_md.write_text("x", encoding="utf-8")
            return out_md

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(pipeline_module, "run_stt", lambda *a, **k: None)  # noqa: E731
        monkeypatch.setattr(pipeline_module, "run_synthesize", _synthesize)
        out_dir = tmp_path / "Лекция"
        cache = out_dir / ".absorbent"
        cache.mkdir(parents=True)
        # Кэш кадров пуст, а descriptions.json — список (не словарь) -> сброс в {}.
        (cache / "descriptions.json").write_text("[]", encoding="utf-8")
        # skip_frames -> keyframes остаются пустыми; skip_describe=False,
        # поэтому отрабатывает ветка «кадров нет, читаем кэш описаний».
        args_namespace.skip_frames = True
        input_path = _input(tmp_path)
        process_file(input_path, cfg, args_namespace)
        assert seen["descriptions"] == {}
        monkeypatch.undo()

    def test_skip_describe_loads_existing_descriptions(self, tmp_path: Path, cfg,
                                                       fake_stages: dict, args_namespace) -> None:
        """--skip-describe: готовые описания читаются из кэша и уходят в синтез."""
        import absorbent.pipeline as pipeline_module

        seen: dict = {}

        def _synthesize(llm_spec, provider, transcript, descriptions, output_language, out_md,
                        images_prefix="images/"):
            seen["descriptions"] = descriptions
            out_md.parent.mkdir(parents=True, exist_ok=True)
            out_md.write_text("x", encoding="utf-8")
            return out_md

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(pipeline_module, "run_stt", lambda *a, **k: None)  # noqa: E731
        monkeypatch.setattr(pipeline_module, "run_synthesize", _synthesize)
        out_dir = tmp_path / "Лекция"
        cache = out_dir / ".absorbent"
        cache.mkdir(parents=True)
        # В кэше два описания: кадр текущего отбора и кадр прошлого отбора,
        # которого в keyframes уже нет; до синтеза доходит только первый.
        (cache / "descriptions.json").write_text(
            json.dumps({"frame_00_00_00.jpg": {"timestamp": "00:00:00", "description": "текущий"},
                        "frame_00_00_09.jpg": {"timestamp": "00:00:09", "description": "прошлый отбор"}}),
            encoding="utf-8")
        args_namespace.skip_describe = True
        input_path = _input(tmp_path)
        process_file(input_path, cfg, args_namespace)
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
        assert fake_stages["calls"] == ["stt:auto", "synthesize:ru:0", "render"]
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
