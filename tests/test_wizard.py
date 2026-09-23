"""Тесты мастера: выбор файлов и слайдов, сборка аргументов, диалоги prompt_toolkit."""

from __future__ import annotations

from pathlib import Path

import pytest
from prompt_toolkit.document import Document
from prompt_toolkit.validation import ValidationError
from prompt_toolkit.widgets import Button, CheckboxList, TextArea

from absorbent.config import load_config
from absorbent.wizard import (
    MANUAL_PATH,
    OTHER_DIRECTORY,
    Console,
    TtyConsole,
    _ask_number,
    _button_app,
    _checkbox_app,
    _choose_input,
    _choose_slides,
    _EnterConfirmsRadioList,
    _input_app,
    _message_app,
    _number_check,
    _radio_app,
    run_wizard,
)

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


class ScriptedConsole(Console):
    """Подделка Console: заранее заданные ответы на диалоги и запись вопросов."""

    def __init__(self, selects: list = (), many: list = (), texts: list = (),
                 confirms: list = ()) -> None:
        self._selects = list(selects)
        self._many = list(many)
        self._texts = list(texts)
        self._confirms = list(confirms)
        self.questions: list[tuple] = []
        self.messages: list[tuple[str, str]] = []

    def _pop(self, queue: list):
        """Следующий ответ; экземпляр исключения в сценарии — это его бросок."""
        answer = queue.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def select(self, title, text, values, default=None):
        self.questions.append(("select", title, text, values, default))
        return self._pop(self._selects)

    def select_many(self, title, text, values, defaults=()):
        self.questions.append(("select_many", title, text, values, list(defaults)))
        return self._pop(self._many)

    def text(self, title, text, default="", only_directories=False, validator=None,
             error_message=""):
        self.questions.append(("text", title, text, default, only_directories))
        return self._pop(self._texts)

    def confirm(self, title, text, default=True):
        self.questions.append(("confirm", title, text, default))
        return self._pop(self._confirms)

    def message(self, title, text):
        self.messages.append((title, text))

    def questions_of(self, kind: str) -> list[tuple]:
        return [question for question in self.questions if question[0] == kind]


def _media(tmp_path: Path, name: str = "lecture.m4a") -> Path:
    path = tmp_path / name
    path.write_bytes(b"media")
    return path


def _slides(tmp_path: Path, name: str = "deck.pdf") -> Path:
    path = tmp_path / name
    path.write_bytes(b"%PDF")
    return path


class TestChooseInput:
    def test_picks_file_from_listing(self, tmp_path: Path) -> None:
        first = _media(tmp_path, "a.m4a")
        second = _media(tmp_path, "b.m4a")
        console = ScriptedConsole(selects=[str(first)])
        assert _choose_input(console, tmp_path) == first
        _, title, _, values, default = console.questions[0]
        assert title == "Запись лекции"
        assert values[:2] == [(str(first), str(first)), (str(second), str(second))]
        assert values[2:] == [(OTHER_DIRECTORY, "Другой каталог…"),
                              (MANUAL_PATH, "Указать путь к файлу…")]
        assert default == str(first)

    def test_switches_to_other_directory(self, tmp_path: Path) -> None:
        other = tmp_path / "other"
        other.mkdir()
        target = _media(other, "there.m4a")
        console = ScriptedConsole(selects=[OTHER_DIRECTORY, str(target)], texts=[str(other)])
        assert _choose_input(console, tmp_path) == target
        # Второй список — уже содержимое выбранного каталога.
        assert console.questions_of("select")[1][3][0] == (str(target), str(target))
        assert console.questions_of("text")[0][4] is True  # only_directories.

    def test_missing_directory_message(self, tmp_path: Path) -> None:
        target = _media(tmp_path)
        missing = tmp_path / "nope"
        console = ScriptedConsole(selects=[OTHER_DIRECTORY, str(target)], texts=[str(missing)])
        assert _choose_input(console, tmp_path) == target
        assert console.messages == [("Каталог не найден", f"Нет такого каталога: {missing}")]

    def test_manual_file_path(self, tmp_path: Path) -> None:
        target = _media(tmp_path, "manual.m4a")
        console = ScriptedConsole(selects=[MANUAL_PATH], texts=[str(target)])
        assert _choose_input(console, tmp_path) == target

    def test_manual_directory_path_shows_listing(self, tmp_path: Path) -> None:
        other = tmp_path / "other"
        other.mkdir()
        target = _media(other, "deep.m4a")
        console = ScriptedConsole(selects=[MANUAL_PATH, str(target)], texts=[str(other)])
        assert _choose_input(console, tmp_path) == target
        assert console.questions_of("select")[1][3][0] == (str(target), str(target))

    def test_missing_file_message_and_retry(self, tmp_path: Path) -> None:
        target = _media(tmp_path)
        missing = tmp_path / "ghost.m4a"
        console = ScriptedConsole(selects=[MANUAL_PATH, str(target)], texts=[str(missing)])
        assert _choose_input(console, tmp_path) == target
        assert console.messages == [("Файл не найден", f"Нет такого файла: {missing}")]

    def test_empty_directory_defaults_to_manual_path(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        target = _media(tmp_path, "later.m4a")
        console = ScriptedConsole(selects=[MANUAL_PATH, str(target)], texts=[str(target)])
        assert _choose_input(console, empty) == target
        _, _, _, values, default = console.questions[0]
        assert values == [(OTHER_DIRECTORY, "Другой каталог…"),
                          (MANUAL_PATH, "Указать путь к файлу…")]
        assert default == MANUAL_PATH

    def test_cancel_in_list(self, tmp_path: Path) -> None:
        console = ScriptedConsole(selects=[None])
        assert _choose_input(console, tmp_path) is None

    def test_cancel_in_text_dialog(self, tmp_path: Path) -> None:
        console = ScriptedConsole(selects=[MANUAL_PATH], texts=[None])
        assert _choose_input(console, tmp_path) is None

    def test_cancel_in_other_directory_dialog(self, tmp_path: Path) -> None:
        console = ScriptedConsole(selects=[OTHER_DIRECTORY], texts=[None])
        assert _choose_input(console, tmp_path) is None


class TestChooseSlides:
    def test_no_marks_means_no_slides(self, tmp_path: Path) -> None:
        input_path = _media(tmp_path)
        console = ScriptedConsole(many=[[]])
        assert _choose_slides(console, input_path, tmp_path) == []

    def test_marked_candidates_returned(self, tmp_path: Path) -> None:
        input_path = _media(tmp_path)
        deck = _slides(tmp_path)
        console = ScriptedConsole(many=[[str(deck)]])
        assert _choose_slides(console, input_path, tmp_path) == [deck]

    def test_board_photo_listed_as_candidate(self, tmp_path: Path) -> None:
        """Снимок доски предлагается в списке материалов наравне с колодой."""
        input_path = _media(tmp_path)
        photo = tmp_path / "доска.jpg"
        photo.write_bytes(b"\xff\xd8\xff\xe0")
        console = ScriptedConsole(many=[[str(photo)]])
        assert _choose_slides(console, input_path, tmp_path) == [photo]
        values = console.questions_of("select_many")[0][3]
        assert (str(photo), str(photo)) in values

    def test_same_stem_candidate_preselected(self, tmp_path: Path) -> None:
        input_path = _media(tmp_path, "Lecture.m4a")
        deck = _slides(tmp_path, "Lecture.pdf")
        console = ScriptedConsole(many=[[]])
        assert _choose_slides(console, input_path, tmp_path) == []
        assert console.questions_of("select_many")[0][4] == [str(deck)]

    def test_other_directory_adds_files_to_next_listing(self, tmp_path: Path) -> None:
        input_path = _media(tmp_path)
        other = tmp_path / "slides"
        other.mkdir()
        deck = _slides(other, "extra.pdf")
        console = ScriptedConsole(many=[[OTHER_DIRECTORY], [str(deck)]], texts=[str(other)])
        assert _choose_slides(console, input_path, tmp_path) == [deck]
        second = console.questions_of("select_many")[1][3]
        assert (str(deck), str(deck)) in second
        assert console.questions_of("text")[0][4] is True  # only_directories.

    def test_manual_file_and_directory(self, tmp_path: Path) -> None:
        input_path = _media(tmp_path)
        other = tmp_path / "slides"
        other.mkdir()
        deck = _slides(other, "one.pdf")
        console = ScriptedConsole(many=[[MANUAL_PATH], [MANUAL_PATH], [str(deck)]],
                                  texts=[str(deck), str(other)])
        assert _choose_slides(console, input_path, tmp_path) == [deck]
        assert console.questions_of("text")[0][4] is False

    def test_slides_error_message_and_retry(self, tmp_path: Path) -> None:
        input_path = _media(tmp_path)
        deck = _slides(tmp_path)
        missing = tmp_path / "nowhere.pdf"
        console = ScriptedConsole(many=[[MANUAL_PATH], [str(deck)]], texts=[str(missing)])
        assert _choose_slides(console, input_path, tmp_path) == [deck]
        assert console.messages[0][0] == "Материалы не найдены"
        assert "не существует" in console.messages[0][1]

    def test_repeated_path_not_duplicated(self, tmp_path: Path) -> None:
        input_path = _media(tmp_path)
        deck = _slides(tmp_path)
        console = ScriptedConsole(many=[[MANUAL_PATH], [MANUAL_PATH], [str(deck)]],
                                  texts=[str(deck), str(deck)])
        assert _choose_slides(console, input_path, tmp_path) == [deck]
        values = console.questions_of("select_many")[1][3]
        assert values.count((str(deck), str(deck))) == 1

    def test_cancel_in_list(self, tmp_path: Path) -> None:
        console = ScriptedConsole(many=[None])
        assert _choose_slides(console, _media(tmp_path), tmp_path) is None

    def test_cancel_in_text_dialog(self, tmp_path: Path) -> None:
        console = ScriptedConsole(many=[[OTHER_DIRECTORY]], texts=[None])
        assert _choose_slides(console, _media(tmp_path), tmp_path) is None


class TestRunWizard:
    @pytest.fixture
    def cfg(self):
        return load_config(str(REPO_CONFIG))

    def _essentials_console(self, input_path: Path, config_path: str | None = None, flat: bool = True,
                            output_language: str = "ru", llm: str | None = None,
                            **kwargs) -> ScriptedConsole:
        """Ответы основного пути: запись, слайды, раскладка, язык, модель синтеза."""
        return ScriptedConsole(
            selects=[str(input_path), output_language, llm],
            many=[[]],
            confirms=[flat, False, True],  # плоская, без «дополнительно», запуск.
            **kwargs,
        )

    def test_essentials_tokens_and_summary(self, tmp_path: Path, cfg,
                                          capsys: pytest.CaptureFixture) -> None:
        input_path = _media(tmp_path)
        console = self._essentials_console(input_path, llm=cfg.default_name("llm"))
        tokens = run_wizard(cfg, console=console, start_dir=tmp_path)
        # Дефолтная модель синтеза в команду не пишется.
        assert tokens == ["--flat", "--output-language", "ru", str(input_path)]
        out = capsys.readouterr().out
        assert f"Команда: absorbent --flat --output-language ru {input_path}" in out
        # Список моделей синтеза помечает дефолтную модель.
        values = console.questions_of("select")[2][3]
        assert (cfg.default_name("llm"), f"{cfg.default_name('llm')}  (по умолчанию)") in values

    def test_config_path_is_first_token(self, tmp_path: Path, cfg) -> None:
        input_path = _media(tmp_path)
        console = self._essentials_console(input_path, llm=cfg.default_name("llm"))
        tokens = run_wizard(cfg, str(REPO_CONFIG), console, tmp_path)
        assert tokens[:2] == ["--config", str(REPO_CONFIG)]
        assert tokens[-1] == str(input_path)

    def test_separate_output_directory(self, tmp_path: Path, cfg) -> None:
        input_path = _media(tmp_path)
        console = self._essentials_console(input_path, flat=False, llm=cfg.default_name("llm"))
        tokens = run_wizard(cfg, console=console, start_dir=tmp_path)
        assert "--flat" not in tokens

    def test_non_default_llm_model_added(self, tmp_path: Path, cfg) -> None:
        input_path = _media(tmp_path)
        console = self._essentials_console(input_path, llm="deepseek-v4-pro")
        tokens = run_wizard(cfg, console=console, start_dir=tmp_path)
        assert tokens == ["--flat", "--output-language", "ru", "--llm-model", "deepseek-v4-pro",
                          str(input_path)]

    def test_slides_and_path_with_spaces_are_quoted(self, tmp_path: Path, cfg,
                                                    capsys: pytest.CaptureFixture) -> None:
        input_path = _media(tmp_path, "lecture one.m4a")
        deck = _slides(tmp_path, "deck one.pdf")
        console = ScriptedConsole(selects=[str(input_path), "ru", cfg.default_name("llm")],
                                  many=[[str(deck)]], confirms=[True, False, True])
        tokens = run_wizard(cfg, console=console, start_dir=tmp_path)
        assert tokens == ["--slides", str(deck), "--flat", "--output-language", "ru",
                          str(input_path)]
        # Сводка пригодна для копирования в терминал: путь с пробелом в кавычках.
        assert f"'{deck}'" in capsys.readouterr().out

    def test_advanced_block_tokens(self, tmp_path: Path, cfg) -> None:
        input_path = _media(tmp_path)
        console = ScriptedConsole(
            selects=[str(input_path), "ru", cfg.default_name("llm"), "en",
                     "voxtral-transcribe-2", cfg.default_name("vision")],
            many=[[], ["--skip-stt", "--skip-describe"]],
            texts=["2", "10", "5", "300"],
            confirms=[False, True, True, True],  # не плоская, «дополнительно», --force, запуск.
        )
        tokens = run_wizard(cfg, console=console, start_dir=tmp_path)
        assert tokens == [
            "--output-language", "ru",
            "--language", "en",
            "--stt", "voxtral-transcribe-2",
            "--fps", "2.0",
            "--hash-threshold", "10",
            "--min-scene-seconds", "5.0",
            "--slides-dpi", "300",
            "--force",
            "--skip-stt", "--skip-describe",
            str(input_path),
        ]
        # Дефолтная модель описания в команду не попала, а язык конспекта спрошен.
        assert "--vision-model" not in tokens
        # Числовые вопросы получили значения по умолчанию и подсказку об ошибке.
        numbers = console.questions_of("text")
        assert [question[1] for question in numbers] == ["--fps", "--hash-threshold",
                                                        "--min-scene-seconds", "--slides-dpi"]
        assert [question[3] for question in numbers] == ["1", "10", "5", "150"]

    def test_advanced_block_accepted_by_defaults(self, tmp_path: Path, cfg) -> None:
        """Enter на числовых вопросах возвращает значения по умолчанию."""
        input_path = _media(tmp_path)
        console = ScriptedConsole(
            selects=[str(input_path), "ru", cfg.default_name("llm"), "auto",
                     cfg.default_name("stt"), cfg.default_name("vision")],
            many=[[], []],
            texts=["1", "10", "5", "150"],
            confirms=[True, True, False, True],
        )
        tokens = run_wizard(cfg, console=console, start_dir=tmp_path)
        assert tokens == ["--flat", "--output-language", "ru", "--language", "auto",
                          "--fps", "1.0", "--hash-threshold", "10", "--min-scene-seconds", "5.0",
                          "--slides-dpi", "150", str(input_path)]

    def test_cancel_at_input(self, tmp_path: Path, cfg) -> None:
        console = ScriptedConsole(selects=[None])
        assert run_wizard(cfg, console=console, start_dir=tmp_path) is None

    def test_cancel_at_slides(self, tmp_path: Path, cfg) -> None:
        console = ScriptedConsole(selects=[str(_media(tmp_path))], many=[None])
        assert run_wizard(cfg, console=console, start_dir=tmp_path) is None

    def test_cancel_at_output_language(self, tmp_path: Path, cfg) -> None:
        console = ScriptedConsole(selects=[str(_media(tmp_path)), None], many=[[]],
                                  confirms=[True])
        assert run_wizard(cfg, console=console, start_dir=tmp_path) is None

    def test_cancel_at_llm_model(self, tmp_path: Path, cfg) -> None:
        console = ScriptedConsole(selects=[str(_media(tmp_path)), "ru", None], many=[[]],
                                  confirms=[True])
        assert run_wizard(cfg, console=console, start_dir=tmp_path) is None

    def test_cancel_at_language(self, tmp_path: Path, cfg) -> None:
        console = ScriptedConsole(
            selects=[str(_media(tmp_path)), "ru", cfg.default_name("llm"), None], many=[[]],
            confirms=[True, True])
        assert run_wizard(cfg, console=console, start_dir=tmp_path) is None

    def test_cancel_at_stt_model(self, tmp_path: Path, cfg) -> None:
        console = ScriptedConsole(
            selects=[str(_media(tmp_path)), "ru", cfg.default_name("llm"), "auto", None],
            many=[[]], confirms=[True, True])
        assert run_wizard(cfg, console=console, start_dir=tmp_path) is None

    def test_cancel_at_vision_model(self, tmp_path: Path, cfg) -> None:
        console = ScriptedConsole(
            selects=[str(_media(tmp_path)), "ru", cfg.default_name("llm"), "auto",
                     cfg.default_name("stt"), None],
            many=[[]], confirms=[True, True])
        assert run_wizard(cfg, console=console, start_dir=tmp_path) is None

    def test_cancel_at_number(self, tmp_path: Path, cfg) -> None:
        console = ScriptedConsole(
            selects=[str(_media(tmp_path)), "ru", cfg.default_name("llm"), "auto",
                     cfg.default_name("stt"), cfg.default_name("vision")],
            many=[[]], texts=[None], confirms=[True, True])
        assert run_wizard(cfg, console=console, start_dir=tmp_path) is None

    def test_cancel_at_skip_list(self, tmp_path: Path, cfg) -> None:
        console = ScriptedConsole(
            selects=[str(_media(tmp_path)), "ru", cfg.default_name("llm"), "auto",
                     cfg.default_name("stt"), cfg.default_name("vision")],
            many=[[], None], texts=["1", "10", "5", "150"], confirms=[True, True, False])
        assert run_wizard(cfg, console=console, start_dir=tmp_path) is None

    def test_refusal_at_final_confirmation(self, tmp_path: Path, cfg,
                                           capsys: pytest.CaptureFixture) -> None:
        input_path = _media(tmp_path)
        console = ScriptedConsole(selects=[str(input_path), "ru", cfg.default_name("llm")],
                                  many=[[]], confirms=[True, False, False])
        assert run_wizard(cfg, console=console, start_dir=tmp_path) is None
        assert "Команда: absorbent" in capsys.readouterr().out

    def test_keyboard_interrupt_cancels(self, tmp_path: Path, cfg) -> None:
        console = ScriptedConsole(selects=[KeyboardInterrupt()])
        assert run_wizard(cfg, console=console, start_dir=tmp_path) is None

    def test_end_of_input_cancels(self, tmp_path: Path, cfg) -> None:
        console = ScriptedConsole(selects=[str(_media(tmp_path))], many=[EOFError()])
        assert run_wizard(cfg, console=console, start_dir=tmp_path) is None

    def test_default_start_dir_and_console(self, tmp_path: Path, cfg,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
        """Без явных аргументов мастер берёт текущий каталог и настоящую консоль."""
        import absorbent.wizard as wizard_module

        monkeypatch.chdir(tmp_path)
        input_path = _media(tmp_path)
        seen: dict = {}

        class _Console(TtyConsole):
            def select(self, title, text, values, default=None):
                if title != "Запись лекции":
                    return default  # остальные списки отвечают вариантом по умолчанию.
                seen["values"] = values
                return str(input_path)

            def select_many(self, title, text, values, defaults=()):
                return []

            def confirm(self, title, text, default=True):
                return title != "Дополнительные параметры"

            def text(self, title, text, default="", only_directories=False, validator=None,
                     error_message=""):
                return text

        monkeypatch.setattr(wizard_module, "TtyConsole", _Console)
        tokens = run_wizard(cfg)
        assert tokens is not None and tokens[-1] == str(input_path)
        assert (str(input_path), str(input_path)) in seen["values"]


class _Dialog:
    """Подделка диалога prompt_toolkit: run() отдаёт заранее заданный результат."""

    def __init__(self, result: object) -> None:
        self.result = result

    def run(self) -> object:
        return self.result


class _Recorder:
    """Подделка диалоговой функции: запоминает вызов и отдаёт готовый диалог."""

    def __init__(self, result: object = None) -> None:
        self.result = result
        self.calls: list[tuple[tuple, dict]] = []

    def __call__(self, *args, **kwargs) -> _Dialog:
        self.calls.append((args, kwargs))
        return _Dialog(self.result)

    def kwargs(self, index: int = 0) -> dict:
        return self.calls[index][1]


class TestTtyConsole:
    def test_select_uses_radio_app(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import absorbent.wizard as wizard_module

        app = _Recorder("ru")
        monkeypatch.setattr(wizard_module, "_radio_app", app)
        values = [("ru", "Русский"), ("en", "English")]
        assert TtyConsole().select("Язык конспекта", "текст", values, default="ru") == "ru"
        assert app.calls[0][0] == ("Язык конспекта", "текст", values, "ru")

    def test_select_many_uses_checkbox_app(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import absorbent.wizard as wizard_module

        app = _Recorder(["a"])
        monkeypatch.setattr(wizard_module, "_checkbox_app", app)
        values = [("a", "A"), ("b", "B")]
        assert TtyConsole().select_many("Слайды", "текст", values, defaults=["b"]) == ["a"]
        assert app.calls[0][0] == ("Слайды", "текст", values, ["b"])

    def test_text_wires_path_completer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import absorbent.wizard as wizard_module

        app = _Recorder("/tmp")
        monkeypatch.setattr(wizard_module, "_input_app", app)
        assert TtyConsole().text("Файл", "текст", default="/tmp/", only_directories=True) == "/tmp"
        args = app.calls[0][0]
        assert args[:3] == ("Файл", "текст", "/tmp/")
        assert args[3].expanduser is True
        assert args[3].only_directories is True
        assert args[4] is None  # валидатора нет.

    def test_text_validator_accepts_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Валидатор проверяет и значение по умолчанию: Enter на пустом поле проходит."""
        import absorbent.wizard as wizard_module

        app = _Recorder("5")
        monkeypatch.setattr(wizard_module, "_input_app", app)
        TtyConsole().text("Число", "текст", default="5", validator=lambda text: text.isdigit(),
                          error_message="нужно число")
        validator = app.calls[0][0][4]
        validator.validate(Document(""))  # пусто -> проверяется default "5".
        validator.validate(Document("7"))
        with pytest.raises(ValidationError):
            validator.validate(Document("abc"))

    def test_confirm_button_order_follows_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import absorbent.wizard as wizard_module

        yes_first = _Recorder(True)
        monkeypatch.setattr(wizard_module, "_button_app", yes_first)
        assert TtyConsole().confirm("Запуск", "текст", default=True) is True
        assert yes_first.calls[0][0] == ("Запуск", "текст", [("Да", True), ("Нет", False)])
        no_first = _Recorder(False)
        monkeypatch.setattr(wizard_module, "_button_app", no_first)
        assert TtyConsole().confirm("Запуск", "текст", default=False) is False
        assert no_first.calls[0][0] == ("Запуск", "текст", [("Нет", False), ("Да", True)])

    def test_message_uses_message_app(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import absorbent.wizard as wizard_module

        app = _Recorder(None)
        monkeypatch.setattr(wizard_module, "_message_app", app)
        TtyConsole().message("Ошибка", "текст ошибки")
        assert app.calls[0][0] == ("Ошибка", "текст ошибки")


class RecordingButton(Button):
    """Button, запоминающий экземпляры: по ним тест находит обработчики кнопок."""

    instances: list["RecordingButton"] = []

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        RecordingButton.instances.append(self)


class RecordingCheckboxList(CheckboxList):
    """CheckboxList, запоминающий экземпляры: виджеты в дереве контейнеров не видны."""

    instances: list["RecordingCheckboxList"] = []

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        RecordingCheckboxList.instances.append(self)


class RecordingRadioList(_EnterConfirmsRadioList):
    """Список мастера, запоминающий экземпляры: виджеты не видны в дереве контейнеров."""

    instances: list["RecordingRadioList"] = []

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        RecordingRadioList.instances.append(self)


class RecordingTextArea(TextArea):
    """TextArea, запоминающий экземпляры, чтобы добраться до его буфера."""

    instances: list["RecordingTextArea"] = []

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        RecordingTextArea.instances.append(self)


class _FakeApp:
    """Подделка Application: запоминает результаты и исключения exit()."""

    def __init__(self) -> None:
        self.results: list = []
        self.exceptions: list = []

    def exit(self, result=None, exception=None) -> None:
        self.results.append(result)
        self.exceptions.append(exception)


@pytest.fixture
def buttons(monkeypatch: pytest.MonkeyPatch) -> list[RecordingButton]:
    import absorbent.wizard as wizard_module

    RecordingButton.instances = []
    monkeypatch.setattr(wizard_module, "Button", RecordingButton)
    return RecordingButton.instances


@pytest.fixture
def listboxes(monkeypatch: pytest.MonkeyPatch) -> list[RecordingCheckboxList]:
    import absorbent.wizard as wizard_module

    RecordingCheckboxList.instances = []
    monkeypatch.setattr(wizard_module, "CheckboxList", RecordingCheckboxList)
    return RecordingCheckboxList.instances


@pytest.fixture
def radios(monkeypatch: pytest.MonkeyPatch) -> list[RecordingRadioList]:
    import absorbent.wizard as wizard_module

    RecordingRadioList.instances = []
    monkeypatch.setattr(wizard_module, "_EnterConfirmsRadioList", RecordingRadioList)
    return RecordingRadioList.instances


@pytest.fixture
def fields(monkeypatch: pytest.MonkeyPatch) -> list[RecordingTextArea]:
    import absorbent.wizard as wizard_module

    RecordingTextArea.instances = []
    monkeypatch.setattr(wizard_module, "TextArea", RecordingTextArea)
    return RecordingTextArea.instances


class TestCheckboxApp:
    def test_checkbox_holds_values_and_defaults(self, buttons: list[RecordingButton],
                                                listboxes: list[RecordingCheckboxList]) -> None:
        values = [("a", "A"), ("b", "B")]
        _checkbox_app("Слайды", "текст", values, defaults=["b"])
        assert [listbox.values for listbox in listboxes] == [values]
        assert [list(listbox.current_values) for listbox in listboxes] == [["b"]]
        assert [button.text for button in buttons] == ["Готово", "Отмена"]

    def test_escape_confirms_marks(self, buttons: list[RecordingButton],
                                   listboxes: list[RecordingCheckboxList],
                                   monkeypatch: pytest.MonkeyPatch) -> None:
        import absorbent.wizard as wizard_module

        values = [("a", "A"), ("b", "B")]
        app = _checkbox_app("Слайды", "текст", values, defaults=["b"])
        fake = _FakeApp()
        monkeypatch.setattr(wizard_module, "get_app", lambda: fake)
        binding = app.key_bindings.get_bindings_for_keys(("escape",))[0]
        binding.handler(None)
        assert fake.results == [["b"]]

    def test_ready_button_confirms_marks(self, buttons: list[RecordingButton],
                                         listboxes: list[RecordingCheckboxList],
                                         monkeypatch: pytest.MonkeyPatch) -> None:
        import absorbent.wizard as wizard_module

        values = [("a", "A"), ("b", "B")]
        _checkbox_app("Слайды", "текст", values, defaults=["a", "b"])
        fake = _FakeApp()
        monkeypatch.setattr(wizard_module, "get_app", lambda: fake)
        buttons[0].handler()
        assert fake.results == [["a", "b"]]

    def test_cancel_button_returns_none(self, buttons: list[RecordingButton],
                                        listboxes: list[RecordingCheckboxList],
                                        monkeypatch: pytest.MonkeyPatch) -> None:
        import absorbent.wizard as wizard_module

        _checkbox_app("Слайды", "текст", [("a", "A")])
        fake = _FakeApp()
        monkeypatch.setattr(wizard_module, "get_app", lambda: fake)
        buttons[1].handler()
        assert fake.results == [None]


class TestNumberInput:
    def test_number_check_bounds(self) -> None:
        check = _number_check(1, 10, int)
        assert check("5") is True
        assert check("1") is True
        assert check("10") is True
        assert check("abc") is False
        assert check("50") is False
        assert check("0") is False

    def test_ask_number_normalizes(self) -> None:
        console = ScriptedConsole(texts=["5"])
        assert _ask_number(console, "--fps", "текст", "1", 0.01, 100, float) == "5.0"
        assert console.questions[0] == ("text", "--fps", "текст", "1", False)

    def test_ask_number_cancel(self) -> None:
        console = ScriptedConsole(texts=[None])
        assert _ask_number(console, "--fps", "текст", "1", 0.01, 100, float) is None

    def test_number_questions_table_shape(self) -> None:
        """Таблица числовых вопросов: флаг, текст, дефолт, границы, приведение."""
        from absorbent.wizard import NUMBER_QUESTIONS

        assert [question[0] for question in NUMBER_QUESTIONS] == [
            "--fps", "--hash-threshold", "--min-scene-seconds", "--slides-dpi"]
        assert all(len(question) == 6 for question in NUMBER_QUESTIONS)


class TestRadioApp:
    def test_list_holds_values_and_default(self, buttons: list[RecordingButton],
                                           radios: list[RecordingRadioList]) -> None:
        values = [("ru", "Русский"), ("en", "English")]
        _radio_app("Язык", "текст", values, default="en")
        assert radios[0].values == values
        assert radios[0].current_value == "en"
        assert [button.text for button in buttons] == ["Отмена"]

    def test_enter_confirms_focused_item(self, buttons: list[RecordingButton],
                                         radios: list[RecordingRadioList],
                                         monkeypatch: pytest.MonkeyPatch) -> None:
        import absorbent.wizard as wizard_module

        values = [("ru", "Русский"), ("en", "English")]
        _radio_app("Язык", "текст", values, default="ru")
        fake = _FakeApp()
        monkeypatch.setattr(wizard_module, "get_app", lambda: fake)
        radios[0]._selected_index = 1
        radios[0]._handle_enter()  # обработчик Enter в списке.
        assert fake.results == ["en"]

    def test_escape_cancels(self, buttons: list[RecordingButton],
                            radios: list[RecordingRadioList],
                            monkeypatch: pytest.MonkeyPatch) -> None:
        import absorbent.wizard as wizard_module

        app = _radio_app("Язык", "текст", [("ru", "Русский")], default="ru")
        fake = _FakeApp()
        monkeypatch.setattr(wizard_module, "get_app", lambda: fake)
        binding = app.key_bindings.get_bindings_for_keys(("escape",))[0]
        binding.handler(None)
        assert fake.results == [None]

    def test_cancel_button_returns_none(self, buttons: list[RecordingButton],
                                        radios: list[RecordingRadioList],
                                        monkeypatch: pytest.MonkeyPatch) -> None:
        import absorbent.wizard as wizard_module

        _radio_app("Язык", "текст", [("ru", "Русский")], default="ru")
        fake = _FakeApp()
        monkeypatch.setattr(wizard_module, "get_app", lambda: fake)
        buttons[0].handler()
        assert fake.results == [None]


class TestInputApp:
    def test_field_is_single_line_with_completer(self, buttons: list[RecordingButton],
                                                 fields: list[RecordingTextArea]) -> None:
        from prompt_toolkit.completion import PathCompleter
        from prompt_toolkit.validation import Validator

        completer = PathCompleter(expanduser=True)
        validator = Validator.from_callable(lambda text: text.isdigit(), "нужно число")
        _input_app("Число", "текст", "5", completer, validator)
        field = fields[0]
        # Поле не предзаполняется: иначе набранное приклеилось бы к «5».
        assert field.text == ""
        assert field.completer is completer
        assert field.validator is validator
        assert [button.text for button in buttons] == ["Готово", "Отмена"]

    def test_enter_returns_typed_text(self, buttons: list[RecordingButton],
                                      fields: list[RecordingTextArea],
                                      monkeypatch: pytest.MonkeyPatch) -> None:
        from prompt_toolkit.completion import PathCompleter

        import absorbent.wizard as wizard_module

        _input_app("Число", "текст", "5", PathCompleter(), None)
        fake = _FakeApp()
        monkeypatch.setattr(wizard_module, "get_app", lambda: fake)
        fields[0].text = "7"
        fields[0].buffer.accept_handler(fields[0].buffer)  # обработчик Enter в поле.
        assert fake.results == ["7"]

    def test_enter_with_empty_field_keeps_default(self, buttons: list[RecordingButton],
                                                  fields: list[RecordingTextArea],
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
        from prompt_toolkit.completion import PathCompleter

        import absorbent.wizard as wizard_module

        _input_app("Число", "текст", "5", PathCompleter(), None)
        fake = _FakeApp()
        monkeypatch.setattr(wizard_module, "get_app", lambda: fake)
        fields[0].buffer.accept_handler(fields[0].buffer)
        assert fake.results == ["5"]

    def test_ready_button_returns_field_text(self, buttons: list[RecordingButton],
                                             fields: list[RecordingTextArea],
                                             monkeypatch: pytest.MonkeyPatch) -> None:
        from prompt_toolkit.completion import PathCompleter

        import absorbent.wizard as wizard_module

        _input_app("Путь", "текст", "/tmp/", PathCompleter(), None)
        fake = _FakeApp()
        monkeypatch.setattr(wizard_module, "get_app", lambda: fake)
        fields[0].text = "/home/"
        buttons[0].handler()
        assert fake.results == ["/home/"]

    def test_escape_cancels(self, buttons: list[RecordingButton],
                            fields: list[RecordingTextArea],
                            monkeypatch: pytest.MonkeyPatch) -> None:
        from prompt_toolkit.completion import PathCompleter

        import absorbent.wizard as wizard_module

        app = _input_app("Путь", "текст", "/tmp/", PathCompleter(), None)
        fake = _FakeApp()
        monkeypatch.setattr(wizard_module, "get_app", lambda: fake)
        binding = app.key_bindings.get_bindings_for_keys(("escape",))[0]
        binding.handler(None)
        assert fake.results == [None]

    def test_cancel_button_returns_none(self, buttons: list[RecordingButton],
                                        fields: list[RecordingTextArea],
                                        monkeypatch: pytest.MonkeyPatch) -> None:
        from prompt_toolkit.completion import PathCompleter

        import absorbent.wizard as wizard_module

        _input_app("Путь", "текст", "/tmp/", PathCompleter(), None)
        fake = _FakeApp()
        monkeypatch.setattr(wizard_module, "get_app", lambda: fake)
        buttons[1].handler()
        assert fake.results == [None]


class TestDialogApps:
    """Клавиши общих диалогов мастера: Ctrl-C прерывает, Enter возвращает ответ."""

    def test_ctrl_c_aborts_every_dialog(self, buttons: list[RecordingButton],
                                        monkeypatch: pytest.MonkeyPatch) -> None:
        from prompt_toolkit.completion import PathCompleter

        import absorbent.wizard as wizard_module

        apps = [
            _radio_app("Язык", "текст", [("ru", "Русский")], default="ru"),
            _checkbox_app("Слайды", "текст", [("a", "A")]),
            _input_app("Путь", "текст", "/tmp/", PathCompleter(), None),
            _button_app("Запуск", "текст", [("Да", True)]),
            _message_app("Ошибка", "текст"),
        ]
        for app in apps:
            fake = _FakeApp()
            monkeypatch.setattr(wizard_module, "get_app", lambda fake=fake: fake)
            binding = app.key_bindings.get_bindings_for_keys(("c-c",))[0]
            binding.handler(None)
            assert fake.exceptions == [KeyboardInterrupt]
            assert fake.results == [None]

    def test_button_app_returns_focused_button(self, buttons: list[RecordingButton],
                                               monkeypatch: pytest.MonkeyPatch) -> None:
        import absorbent.wizard as wizard_module

        _button_app("Запуск", "текст", [("Да", True), ("Нет", False)])
        fake = _FakeApp()
        monkeypatch.setattr(wizard_module, "get_app", lambda: fake)
        assert [button.text for button in buttons] == ["Да", "Нет"]
        buttons[1].handler()
        assert fake.results == [False]

    def test_message_app_dismisses(self, buttons: list[RecordingButton],
                                   monkeypatch: pytest.MonkeyPatch) -> None:
        import absorbent.wizard as wizard_module

        _message_app("Ошибка", "текст ошибки")
        fake = _FakeApp()
        monkeypatch.setattr(wizard_module, "get_app", lambda: fake)
        assert [button.text for button in buttons] == ["Понятно"]
        buttons[0].handler()
        assert fake.results == [None]
