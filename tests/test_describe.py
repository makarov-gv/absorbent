"""Тесты описания ключевых кадров: окно транскрипта, data-URL, промпты, ошибки."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from absorbent.config import PLACEHOLDER_API_KEY, ConfigError, ModelSpec, Provider
from absorbent.describe import (
    _data_url,
    _prompt_language,
    _transcript_window,
    describe_keyframes,
    describe_slides,
)

VISION_SPEC = ModelSpec(name="vis", provider="p", default=True)


def _provider(api_key_env: str = "TEST_KEY_A") -> Provider:
    return Provider(name="p", base_url="https://api.example.test/v1", api_key_env=api_key_env)


def _make_image(images_dir: Path, name: str) -> Path:
    images_dir.mkdir(parents=True, exist_ok=True)
    path = images_dir / name
    path.write_bytes(b"\xff\xd8\xff\xe0fakejpeg\xff\xd9")
    return path


class TestPromptLanguage:
    def test_explicit_language_used(self) -> None:
        assert _prompt_language("ru") == "ru"
        assert _prompt_language("en") == "en"

    def test_unknown_falls_back_ru(self) -> None:
        # Язык промпта всегда явный: неизвестное значение -> запас на 'ru'.
        assert _prompt_language("de") == "ru"


class TestTranscriptWindow:
    def test_formats_lines(self) -> None:
        transcript = [
            {"start": 0.0, "end": 1.0, "text": "начало", "speaker": None},
            {"start": 61.0, "end": 62.0, "text": "далеко", "speaker": None},  # вне окна 60 с.
            {"start": 125.5, "end": 126.0, "text": "граница", "speaker": None},
        ]
        # Кадр на 65-й секунде: окно [5..125]; 125.5 не входит, 61.0 входит.
        window = _transcript_window(transcript, 65.0)
        assert "[01:01] далеко" in window
        assert "начало" not in window
        assert "граница" not in window

    def test_window_edges_inclusive(self) -> None:
        transcript = [{"start": 5.0, "end": 6.0, "text": "ровно на границе", "speaker": None}]
        assert "ровно на границе" in _transcript_window(transcript, 65.0)  # |5-65| = 60.
        assert "ровно на границе" in _transcript_window(transcript, 5.0)

    def test_empty_window_literal(self) -> None:
        assert _transcript_window([], 10.0) == "(no transcript near this frame)"


class TestDataUrl:
    def test_jpeg_mime_and_base64(self, tmp_path: Path) -> None:
        path = _make_image(tmp_path / "images", "frame_00_00_00.jpg")
        url = _data_url(path)
        assert url.startswith("data:image/jpeg;base64,")
        assert base64.b64decode(url.split(",", 1)[1]) == b"\xff\xd8\xff\xe0fakejpeg\xff\xd9"

    def test_png_mime_by_extension(self, tmp_path: Path) -> None:
        png = tmp_path / "images" / "frame.png"
        _make_image(tmp_path / "images", "frame.png")
        png.write_bytes(b"png-bytes")
        assert _data_url(png).startswith("data:image/png;base64,")

    def test_unknown_extension_defaults_jpeg(self, tmp_path: Path) -> None:
        blob = tmp_path / "frame.unknown"
        blob.write_bytes(b"data")
        assert _data_url(blob).startswith("data:image/jpeg;base64,")


class TestDescribeKeyframes:
    def _setup(self, tmp_path: Path) -> tuple[Path, Path]:
        images_dir = tmp_path / "images"
        _make_image(images_dir, "frame_00_00_00.jpg")
        _make_image(images_dir, "frame_00_00_05.jpg")
        return tmp_path, images_dir

    def _frame(self, seconds: float) -> dict:
        return {"image": f"frame_{int(seconds // 3600):02d}_"
                         f"{int(seconds // 60 % 60):02d}_{int(seconds % 60):02d}.jpg",
                "seconds": seconds,
                "timestamp": f"{int(seconds // 3600):02d}:{int(seconds // 60 % 60):02d}:"
                             f"{int(seconds % 60):02d}"}

    def test_describes_and_writes_json(self, tmp_path: Path, fake_openai,
                                       monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_KEY_A", "k")
        _, images_dir = self._setup(tmp_path)
        fake_openai.set_replies(["описание первого", "описание второго"])
        keyframes = [self._frame(0.0), self._frame(5.0)]
        transcript = [{"start": 2.0, "end": 3.0, "text": "речь на кадре", "speaker": None}]
        out = tmp_path / "descriptions.json"
        result = describe_keyframes(VISION_SPEC, _provider(), keyframes, transcript, images_dir,
                                    "ru", out)
        assert set(result) == {"frame_00_00_00.jpg", "frame_00_00_05.jpg"}
        assert result["frame_00_00_00.jpg"]["description"] == "описание первого"
        saved = json.loads(out.read_text(encoding="utf-8"))
        assert saved["frame_00_00_05.jpg"]["description"] == "описание второго"

    def test_user_message_contains_timestamp_and_transcript(self, tmp_path: Path, fake_openai,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_KEY_A", "k")
        _, images_dir = self._setup(tmp_path)
        fake_openai.set_replies(["описание"])
        keyframes = [self._frame(5.0)]
        transcript = [{"start": 4.0, "end": 4.5, "text": "слова лектора", "speaker": None}]
        describe_keyframes(VISION_SPEC, _provider(), keyframes, transcript, images_dir, "ru",
                           tmp_path / "descriptions.json")
        call = fake_openai.chat.completions.calls[0]
        messages = call["messages"]
        assert messages[0]["role"] == "system"
        user = messages[1]
        assert user["content"][0]["text"] == (
            "Кадр на таймкоде 00:00:05.\n"
            "Фрагмент транскрипта вокруг этого момента:\n[00:04] слова лектора\n\n"
            "Опиши содержимое кадра."
        )
        assert user["content"][1]["type"] == "image_url"
        assert user["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    def test_en_prompt_used(self, tmp_path: Path, fake_openai, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_KEY_A", "k")
        _, images_dir = self._setup(tmp_path)
        fake_openai.set_replies(["desc"])
        keyframes = [self._frame(0.0)]
        describe_keyframes(VISION_SPEC, _provider(), keyframes, [], images_dir, "en",
                           tmp_path / "descriptions.json")
        system = fake_openai.chat.completions.calls[0]["messages"][0]["content"]
        assert "screencast frame" in system

    def test_unknown_output_language_falls_back_ru(self, tmp_path: Path, fake_openai,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_KEY_A", "k")
        _, images_dir = self._setup(tmp_path)
        fake_openai.set_replies(["описание"])
        keyframes = [self._frame(0.0)]
        # Неизвестный язык -> запас на русский промпт (никогда не авто/детект).
        describe_keyframes(VISION_SPEC, _provider(), keyframes, [], images_dir, "xx",
                           tmp_path / "descriptions.json")
        system = fake_openai.chat.completions.calls[0]["messages"][0]["content"]
        assert "извлекаешь содержимое слайда" in system

    def test_skips_existing_without_force(self, tmp_path: Path, fake_openai,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_KEY_A", "k")
        _, images_dir = self._setup(tmp_path)
        out = tmp_path / "descriptions.json"
        out.write_text(json.dumps({"frame_00_00_00.jpg": {"timestamp": "00:00:00",
                                                          "description": "старое"}}),
                       encoding="utf-8")
        fake_openai.set_replies(["новое"])
        keyframes = [self._frame(0.0), self._frame(5.0)]
        result = describe_keyframes(VISION_SPEC, _provider(), keyframes, [], images_dir, "ru", out)
        assert result["frame_00_00_00.jpg"]["description"] == "старое"  # не переописано.
        assert result["frame_00_00_05.jpg"]["description"] == "новое"
        # Был сделан ровно один API-вызов.
        assert len(fake_openai.chat.completions.calls) == 1

    def test_force_redescribes_all(self, tmp_path: Path, fake_openai,
                                   monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_KEY_A", "k")
        _, images_dir = self._setup(tmp_path)
        out = tmp_path / "descriptions.json"
        out.write_text(json.dumps({"frame_00_00_00.jpg": {"timestamp": "00:00:00",
                                                          "description": "старое"}}),
                       encoding="utf-8")
        fake_openai.set_replies(["новое"])
        keyframes = [self._frame(0.0)]
        result = describe_keyframes(VISION_SPEC, _provider(), keyframes, [], images_dir, "ru", out,
                                    force=True)
        assert result["frame_00_00_00.jpg"]["description"] == "новое"

    def test_api_error_tolerated_per_frame(self, tmp_path: Path, fake_openai,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_KEY_A", "k")
        _, images_dir = self._setup(tmp_path)
        fake_openai.set_replies(["второй кадр"])
        keyframes = [self._frame(0.0), self._frame(5.0)]
        # Первый вызов падает, второй — успешен: ошибка кадра не роняет прогон.
        completions = fake_openai.chat.completions
        original = completions.create
        state = {"n": 0}

        def _flaky(**kwargs):
            state["n"] += 1
            if state["n"] == 1:
                raise RuntimeError("network down")
            return original(**kwargs)

        monkeypatch.setattr(completions, "create", _flaky)
        result = describe_keyframes(VISION_SPEC, _provider(), keyframes, [], images_dir, "ru",
                                    tmp_path / "descriptions.json")
        assert "network down" in result["frame_00_00_00.jpg"]["description"]
        assert result["frame_00_00_05.jpg"]["description"] == "второй кадр"

    def test_missing_api_key_raises(self, tmp_path: Path, fake_openai,
                                    monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TEST_KEY_A", raising=False)
        _, images_dir = self._setup(tmp_path)
        keyframes = [self._frame(0.0)]
        with pytest.raises(ConfigError, match="TEST_KEY_A"):
            describe_keyframes(VISION_SPEC, _provider(), keyframes, [], images_dir, "ru",
                               tmp_path / "descriptions.json")

    def test_no_key_for_empty_env_name(self, tmp_path: Path, fake_openai) -> None:
        # api_key_env == "" -> клиенту уходит заглушка: OpenAI SDK не принимает
        # api_key=None, а подставлять чужой OPENAI_API_KEY из окружения нельзя.
        _, images_dir = self._setup(tmp_path)
        fake_openai.set_replies(["описание"])
        keyframes = [self._frame(0.0)]
        provider = Provider(name="local", base_url="http://127.0.0.1:9/v1", api_key_env="")
        describe_keyframes(VISION_SPEC, provider, keyframes, [], images_dir, "ru",
                           tmp_path / "descriptions.json")
        assert fake_openai.instances[-1].api_key == PLACEHOLDER_API_KEY


class TestDescribeSlides:
    """Описание страниц презентации: без транскрипта и таймкодов, свой промпт."""

    def _setup(self, tmp_path: Path) -> Path:
        images_dir = tmp_path / "images"
        _make_image(images_dir, "slide_001.png")
        _make_image(images_dir, "slide_002.png")
        return images_dir

    def _slide(self, index: int) -> dict:
        return {"image": f"slide_{index:03d}.png", "page": index,
                "source": "/decks/Лекция 4.pdf", "source_page": index}

    def test_describes_and_writes_json(self, tmp_path: Path, fake_openai,
                                       monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_KEY_A", "k")
        images_dir = self._setup(tmp_path)
        fake_openai.set_replies(["первый слайд", "второй слайд"])
        slides = [self._slide(1), self._slide(2)]
        out = tmp_path / "slide_descriptions.json"
        result = describe_slides(VISION_SPEC, _provider(), slides, images_dir, "ru", out)
        assert result == {
            "slide_001.png": {"page": 1, "source": "/decks/Лекция 4.pdf", "kind": "slide",
                              "description": "первый слайд"},
            "slide_002.png": {"page": 2, "source": "/decks/Лекция 4.pdf", "kind": "slide",
                              "description": "второй слайд"},
        }
        saved = json.loads(out.read_text(encoding="utf-8"))
        assert saved == result

    def test_prompt_has_page_number_and_no_timestamp(self, tmp_path: Path, fake_openai,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_KEY_A", "k")
        images_dir = self._setup(tmp_path)
        fake_openai.set_replies(["описание"])
        describe_slides(VISION_SPEC, _provider(), [self._slide(2)], images_dir, "ru",
                        tmp_path / "slide_descriptions.json")
        call = fake_openai.chat.completions.calls[0]
        system = call["messages"][0]["content"]
        user = call["messages"][1]["content"]
        # Промпт страницы отличается от промпта кадра: ни транскрипта, ни таймкода.
        assert "содержимое страницы презентации" in system
        assert "транскрип" not in system.lower()
        assert user[0]["text"] == "Слайд 2 из 1 (файл Лекция 4.pdf).\nОпиши содержимое слайда."
        assert "транскрип" not in user[0]["text"].lower()
        assert user[1]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_en_prompt_used(self, tmp_path: Path, fake_openai,
                            monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_KEY_A", "k")
        images_dir = self._setup(tmp_path)
        fake_openai.set_replies(["desc"])
        describe_slides(VISION_SPEC, _provider(), [self._slide(1)], images_dir, "en",
                        tmp_path / "slide_descriptions.json")
        system = fake_openai.chat.completions.calls[0]["messages"][0]["content"]
        assert "lecture presentation page" in system

    def test_unknown_language_falls_back_ru(self, tmp_path: Path, fake_openai,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_KEY_A", "k")
        images_dir = self._setup(tmp_path)
        fake_openai.set_replies(["описание"])
        describe_slides(VISION_SPEC, _provider(), [self._slide(1)], images_dir, "xx",
                        tmp_path / "slide_descriptions.json")
        system = fake_openai.chat.completions.calls[0]["messages"][0]["content"]
        assert "содержимое страницы презентации" in system

    def test_photo_described_with_board_prompt(self, tmp_path: Path, fake_openai,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
        """Снимок доски описывается промптом доски, а не промптом слайда."""
        monkeypatch.setenv("TEST_KEY_A", "k")
        images_dir = tmp_path / "images"
        _make_image(images_dir, "photo_001.jpg")
        fake_openai.set_replies(["формула на доске"])
        photo = {"image": "photo_001.jpg", "page": 1, "kind": "photo",
                 "source": "/снимки/доска.jpg", "source_page": 1}
        out = tmp_path / "slide_descriptions.json"
        result = describe_slides(VISION_SPEC, _provider(), [photo], images_dir, "ru", out)
        assert result == {"photo_001.jpg": {"page": 1, "source": "/снимки/доска.jpg",
                                            "kind": "photo", "description": "формула на доске"}}
        call = fake_openai.chat.completions.calls[0]
        assert "содержимое фотографии доски" in call["messages"][0]["content"]
        assert call["messages"][1]["content"][0]["text"] == (
            "Фотография доски (файл доска.jpg), снятая на лекции.\nОпиши содержимое снимка.")
        assert call["messages"][1]["content"][1]["image_url"]["url"].startswith(
            "data:image/jpeg;base64,")

    def test_skips_existing_without_force(self, tmp_path: Path, fake_openai,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_KEY_A", "k")
        images_dir = self._setup(tmp_path)
        out = tmp_path / "slide_descriptions.json"
        out.write_text(json.dumps({"slide_001.png": {"page": 1, "source": "/decks/deck.pdf",
                                                    "description": "старое"}}),
                       encoding="utf-8")
        fake_openai.set_replies(["новое"])
        result = describe_slides(VISION_SPEC, _provider(), [self._slide(1), self._slide(2)],
                                 images_dir, "ru", out)
        assert result["slide_001.png"]["description"] == "старое"  # не переописано.
        assert result["slide_002.png"]["description"] == "новое"
        assert len(fake_openai.chat.completions.calls) == 1

    def test_force_redescribes_all(self, tmp_path: Path, fake_openai,
                                   monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_KEY_A", "k")
        images_dir = self._setup(tmp_path)
        out = tmp_path / "slide_descriptions.json"
        out.write_text(json.dumps({"slide_001.png": {"page": 1, "source": "/decks/deck.pdf",
                                                    "description": "старое"}}),
                       encoding="utf-8")
        fake_openai.set_replies(["новое"])
        result = describe_slides(VISION_SPEC, _provider(), [self._slide(1)], images_dir, "ru",
                                 out, force=True)
        assert result["slide_001.png"]["description"] == "новое"

    def test_api_error_tolerated_per_slide(self, tmp_path: Path, fake_openai,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_KEY_A", "k")
        images_dir = self._setup(tmp_path)
        fake_openai.set_replies(["второй слайд"])
        completions = fake_openai.chat.completions
        original = completions.create
        state = {"n": 0}

        def _flaky(**kwargs):
            state["n"] += 1
            if state["n"] == 1:
                raise RuntimeError("network down")
            return original(**kwargs)

        monkeypatch.setattr(completions, "create", _flaky)
        result = describe_slides(VISION_SPEC, _provider(), [self._slide(1), self._slide(2)],
                                 images_dir, "ru", tmp_path / "slide_descriptions.json")
        assert "(ошибка описания: network down)" in result["slide_001.png"]["description"]
        assert result["slide_002.png"]["description"] == "второй слайд"

    def test_missing_api_key_raises(self, tmp_path: Path, fake_openai,
                                    monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TEST_KEY_A", raising=False)
        images_dir = self._setup(tmp_path)
        with pytest.raises(ConfigError, match="TEST_KEY_A"):
            describe_slides(VISION_SPEC, _provider(), [self._slide(1)], images_dir, "ru",
                            tmp_path / "slide_descriptions.json")
