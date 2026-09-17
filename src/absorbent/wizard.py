"""Интерактивный мастер: диалоги prompt_toolkit, собирающие аргументы прогона.

Мастер обрабатывает одну запись за раз и возвращает готовую командную строку
(список аргументов без имени программы): cli.main переразбирает её тем же
парсером, поэтому напечатанную сводку можно скопировать в терминал.

Штатные диалоги prompt_toolkit не годятся в качестве управления мастером:
radiolist_dialog по Enter только запоминает пункт (закрывает его кнопка
«Готово»), input_dialog по Enter лишь переводит фокус на кнопку, а Esc не
отменяет ни то, ни другое. Поэтому диалоги списков и ввода собраны здесь:
Enter возвращает выбор сразу, Esc отменяет шаг (None), а Ctrl-C отменяет
мастер целиком. Диалоги «да/нет» и сообщений взяты штатными: там фокус сразу
на кнопке ответа, и Enter возвращает её.
"""

from __future__ import annotations

import shlex
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from pathlib import Path

from prompt_toolkit.application import Application, get_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import Completer, PathCompleter
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout
from prompt_toolkit.validation import Validator
from prompt_toolkit.widgets import (
    Button,
    CheckboxList,
    Dialog,
    Label,
    RadioList,
    TextArea,
    ValidationToolbar,
)

from absorbent.cli import MEDIA_EXTENSIONS
from absorbent.config import Config
from absorbent.slides import SLIDE_EXTENSIONS, SlidesError, resolve_slide_sources

# Значения-пункты в списках диалогов: сами по себе они не пути.
OTHER_DIRECTORY = "\x00каталог"
MANUAL_PATH = "\x00путь"

# Числовые вопросы блока «дополнительно»: флаг, текст вопроса, значение по
# умолчанию, границы отрезка и приведение типа.
NUMBER_QUESTIONS: tuple[tuple[str, str, str, float, float, Callable[[str], float]], ...] = (
    ("--fps", "Частота извлечения кадров (--fps):", "1", 0.01, 100, float),
    ("--hash-threshold", "Порог pHash-расстояния в битах (--hash-threshold):", "10", 0, 64, int),
    ("--min-scene-seconds", "Минимальная длительность сцены, с (--min-scene-seconds):", "5",
     0, 3600, float),
    ("--slides-dpi", "Разрешение растеризации страниц презентации (--slides-dpi):", "150",
     36, 600, int),
)

# Этапы, которые можно пропустить: значение флага и его пояснение.
SKIP_CHOICES: tuple[tuple[str, str], ...] = (
    ("--skip-stt", "--skip-stt — распознавание речи"),
    ("--skip-frames", "--skip-frames — кадры видео"),
    ("--skip-slides", "--skip-slides — презентации"),
    ("--skip-describe", "--skip-describe — описание кадров и слайдов"),
    ("--skip-synthesize", "--skip-synthesize — синтез конспекта"),
    ("--skip-render", "--skip-render — таблицы, HTML, PDF"),
)


class Console(ABC):
    """Единственная точка диалога с пользователем; в тестах подменяется."""

    @abstractmethod
    def select(self, title: str, text: str, values: list[tuple[str, str]],
               default: str | None = None) -> str | None:
        """Выбор одного варианта; None — отмена (Esc, Ctrl-C)."""

    @abstractmethod
    def select_many(self, title: str, text: str, values: list[tuple[str, str]],
                    defaults: Sequence[str] = ()) -> list[str] | None:
        """Выбор нескольких вариантов; None — отмена; пустой список — «ничего»."""

    @abstractmethod
    def text(self, title: str, text: str, default: str = "",
             only_directories: bool = False,
             validator: Callable[[str], bool] | None = None,
             error_message: str = "") -> str | None:
        """Ввод строки с автодополнением пути; пустой ввод — default; None — отмена."""

    @abstractmethod
    def confirm(self, title: str, text: str, default: bool = True) -> bool:
        """Вопрос да/нет; Enter подтверждает вариант по умолчанию (Esc не отменяет)."""

    @abstractmethod
    def message(self, title: str, text: str) -> None:
        """Сообщение об ошибке ввода (ждёт Enter)."""


def _abort() -> None:
    """Ctrl-C в диалоге: прервать мастер целиком (см. run_wizard)."""
    get_app().exit(exception=KeyboardInterrupt)


def _dialog_app(dialog: Dialog, keys: KeyBindings | None = None) -> Application:
    """Приложение диалога мастера: клавиши диалога плюс Ctrl-C везде.

    Ctrl-C в prompt_toolkit-диалоге сам по себе ничего не делает: терминал в
    raw-режиме не посылает SIGINT, а байт \\x03 остаётся необработанной
    клавишей, и мастер выглядел бы зависшим. Ctrl-C привязывается здесь для
    всех диалогов мастера, включая «да/нет» и сообщения: они тоже собраны на
    этом помощнике.
    """
    bindings = keys if keys is not None else KeyBindings()
    bindings.add("c-c")(lambda event: _abort())
    return Application(layout=Layout(dialog), key_bindings=bindings, mouse_support=True,
                       full_screen=True)


def _message_app(title: str, text: str) -> Application:
    """Сообщение об ошибке ввода: Enter (или кнопка) закрывает диалог."""

    def dismiss() -> None:
        get_app().exit(result=None)

    dialog = Dialog(
        title=title,
        body=HSplit([Label(text=text, dont_extend_height=True)]),
        buttons=[Button(text="Понятно", handler=dismiss)],
        with_background=True,
    )
    return _dialog_app(dialog)


def _button_app(title: str, text: str, buttons: list[tuple[str, bool]]) -> Application:
    """Вопрос с кнопками: Enter возвращает результат кнопки в фокусе."""

    def choose(result: bool) -> None:
        get_app().exit(result=result)

    dialog = Dialog(
        title=title,
        body=HSplit([Label(text=text, dont_extend_height=True)]),
        buttons=[Button(text=label, handler=lambda result=result: choose(result))
                 for label, result in buttons],
        with_background=True,
    )
    return _dialog_app(dialog)


def _checkbox_app(title: str, text: str, values: list[tuple[str, str]],
                  defaults: Sequence[str] = ()) -> Application:
    """Диалог множественного выбора: Enter и пробел — отметка, Esc — подтвердить.

    Штатный checkboxlist_dialog отменяет выбор по Esc: пользователь, нажавший
    Esc после отметок, потерял бы их. Здесь Esc подтверждает отметки, а
    отменяет мастер только Ctrl-C.
    """
    listbox = CheckboxList(values=values, default_values=list(defaults))

    def confirm() -> None:
        get_app().exit(result=list(listbox.current_values))

    def cancel() -> None:
        get_app().exit(result=None)

    dialog = Dialog(
        title=title,
        body=HSplit([
            Label(text=text, dont_extend_height=True),
            listbox,
            Label(text="Пробел — отметить, Esc — подтвердить выбор.", dont_extend_height=True),
        ]),
        buttons=[Button(text="Готово", handler=confirm), Button(text="Отмена", handler=cancel)],
        with_background=True,
    )
    keys = KeyBindings()
    keys.add("escape")(lambda event: confirm())
    return _dialog_app(dialog, keys)


class _EnterConfirmsRadioList(RadioList):
    """Список одиночного выбора, подтверждающий выбор по Enter.

    Штатный radiolist_dialog по Enter только запоминает текущий пункт, а
    закрывается кнопкой «Готово»: пользователь, нажавший Enter, видит, что
    ничего не произошло. Enter (как и пробел, и клик) возвращает выбранный
    пункт сразу.
    """

    def _handle_enter(self) -> None:
        super()._handle_enter()
        get_app().exit(result=self.current_value)


def _radio_app(title: str, text: str, values: list[tuple[str, str]],
               default: str | None = None) -> Application:
    """Диалог одиночного выбора: стрелки двигают, Enter подтверждает, Esc — отмена."""

    def cancel() -> None:
        get_app().exit(result=None)

    dialog = Dialog(
        title=title,
        body=HSplit([
            Label(text=text, dont_extend_height=True),
            _EnterConfirmsRadioList(values, default=default),
            Label(text="Стрелки — выбор, Enter — подтвердить, Esc — отмена.",
                  dont_extend_height=True),
        ]),
        buttons=[Button(text="Отмена", handler=cancel)],
        with_background=True,
    )
    keys = KeyBindings()
    keys.add("escape")(lambda event: cancel())
    return _dialog_app(dialog, keys)


def _input_app(title: str, text: str, default: str, completer: Completer,
               validator: Validator | None) -> Application:
    """Диалог ввода строки: Enter возвращает текст, Esc — отмена.

    Штатный input_dialog по Enter лишь переводит фокус на кнопку «Готово»
    (нужно нажать Enter дважды), а Esc не отменяет ввод. Неверный ввод сюда не
    проходит: валидатор показывает сообщение, диалог остаётся открытым.

    Поле не заполняется значением по умолчанию: курсор в TextArea стоит в
    начале, поэтому набранное приписывалось бы к готовому тексту и «5»
    превращалось бы в «15». Пустой ввод означает «оставить значение по
    умолчанию», о нём сообщает подсказка под вопросом.
    """
    def resolve(raw: str) -> None:
        get_app().exit(result=raw or default)

    def accept(buffer: Buffer) -> bool:
        resolve(buffer.text)
        return True

    def cancel() -> None:
        get_app().exit(result=None)

    field = TextArea(multiline=False, completer=completer, validator=validator,
                     accept_handler=accept)
    hint = f"Пусто — оставить «{default}»." if default else ""
    dialog = Dialog(
        title=title,
        body=HSplit([
            Label(text=text, dont_extend_height=True),
            field,
            Label(text=hint, dont_extend_height=True),
            ValidationToolbar(),
        ]),
        buttons=[Button(text="Готово", handler=lambda: resolve(field.text)),
                 Button(text="Отмена", handler=cancel)],
        with_background=True,
    )
    keys = KeyBindings()
    keys.add("escape")(lambda event: cancel())
    return _dialog_app(dialog, keys)


class TtyConsole(Console):
    """Диалоги prompt_toolkit: списки, чекбоксы, ввод пути с автодополнением."""

    def select(self, title, text, values, default=None):
        return _radio_app(title, text, values, default).run()

    def select_many(self, title, text, values, defaults=()):
        return _checkbox_app(title, text, values, defaults).run()

    def text(self, title, text, default="", only_directories=False, validator=None,
             error_message=""):
        completer = PathCompleter(expanduser=True, only_directories=only_directories)
        # Пустой ввод означает значение по умолчанию, поэтому проверять нужно и
        # его: иначе Enter на пустом поле не прошёл бы собственный валидатор.
        checked = (None if validator is None
                   else Validator.from_callable(lambda raw: bool(validator(raw or default)),
                                                error_message))
        return _input_app(title, text, default, completer, checked).run()

    def confirm(self, title, text, default=True):
        # Порядок кнопок задаёт вариант по умолчанию: первая кнопка в фокусе,
        # Enter возвращает её результат.
        yes = ("Да", True)
        no = ("Нет", False)
        buttons = [yes, no] if default else [no, yes]
        return bool(_button_app(title, text, buttons).run())

    def message(self, title, text):
        _message_app(title, text).run()


def _number_check(low: float, high: float,
                  cast: Callable[[str], float]) -> Callable[[str], bool]:
    """Проверка числового ввода: строка разбирается в число из отрезка [low, high]."""

    def _ok(text: str) -> bool:
        try:
            value = cast(text)
        except ValueError:
            return False
        return low <= value <= high

    return _ok


def _ask_number(console: Console, title: str, text: str, default: str, low: float,
                high: float, cast: Callable[[str], float]) -> str | None:
    """Спросить число в диапазоне; вернуть нормализованную строку или None при отмене."""
    raw = console.text(title, text, default=default,
                       validator=_number_check(low, high, cast),
                       error_message=f"нужно число от {low} до {high}")
    return None if raw is None else str(cast(raw))


def _choose_input(console: Console, start_dir: Path) -> Path | None:
    """Выбрать запись: список медиафайлов каталога, другой каталог или ручной путь."""
    directory = start_dir
    while True:
        files = sorted(p for p in directory.iterdir()
                       if p.is_file() and p.suffix.lower() in MEDIA_EXTENSIONS)
        values = [(str(p), str(p)) for p in files]
        values += [(OTHER_DIRECTORY, "Другой каталог…"), (MANUAL_PATH, "Указать путь к файлу…")]
        choice = console.select("Запись лекции", f"Каталог: {directory}\n"
                                 "Выберите медиафайл (стрелки, Enter).",
                                 values, default=str(files[0]) if files else MANUAL_PATH)
        if choice is None:
            return None
        if choice == OTHER_DIRECTORY:
            raw = console.text("Другой каталог", "Путь к каталогу с записями:",
                               default=str(directory), only_directories=True)
            if raw is None:
                return None
            candidate = Path(raw).expanduser()
            if candidate.is_dir():
                directory = candidate
            else:
                console.message("Каталог не найден", f"Нет такого каталога: {candidate}")
        elif choice == MANUAL_PATH:
            raw = console.text("Файл записи", "Путь к медиафайлу или каталогу с записями:",
                               default=f"{directory}/")
            if raw is None:
                return None
            candidate = Path(raw).expanduser()
            if candidate.is_dir():
                directory = candidate
            elif candidate.is_file():
                return candidate
            else:
                console.message("Файл не найден", f"Нет такого файла: {candidate}")
        else:
            return Path(choice)


def _choose_slides(console: Console, input_path: Path, start_dir: Path) -> list[Path] | None:
    """Выбрать презентации: отметки в списке, другой каталог или ручной путь.

    Пустая отметка означает «без слайдов». Один кандидат с именем записи
    (например Lecture.m4a и Lecture.pdf) отмечается заранее; молча ничего не
    подставляется — пользователь видит отметку в диалоге.
    """
    candidates: list[Path] = []
    for directory in (start_dir, input_path.parent):
        for path in sorted(directory.iterdir()):
            # Канонический путь: тот же файл, добавленный вручную, не дублируется.
            resolved = path.resolve()
            if (path.is_file() and path.suffix.lower() in SLIDE_EXTENSIONS
                    and resolved not in candidates):
                candidates.append(resolved)
    preset = [str(p) for p in candidates if p.stem == input_path.stem]
    extra: list[Path] = []
    while True:
        listed = candidates + [path for path in extra if path not in candidates]
        values = [(str(p), str(p)) for p in listed]
        values += [(OTHER_DIRECTORY, "Добавить слайды из другого каталога…"),
                   (MANUAL_PATH, "Указать путь к файлу или каталогу…")]
        chosen = console.select_many(
            "Презентация",
            f"Отметьте слайды к «{input_path.name}».\n"
            "Пробел — отметить, Esc — подтвердить. Ничего не отмечено — конспект без слайдов.",
            values, defaults=preset)
        if chosen is None:
            return None
        if OTHER_DIRECTORY in chosen or MANUAL_PATH in chosen:
            from_directory = OTHER_DIRECTORY in chosen
            raw = console.text(
                "Каталог со слайдами" if from_directory else "Слайды",
                "Путь к каталогу с презентациями:" if from_directory
                else "Путь к файлу презентации или каталогу с ними:",
                default=str(start_dir), only_directories=from_directory)
            if raw is None:
                return None
            try:
                added = resolve_slide_sources([raw])
            except SlidesError as exc:
                console.message("Слайды не найдены", str(exc))
                continue
            for path in added:
                if path not in extra:
                    extra.append(path)
            continue
        return [Path(value) for value in chosen]


def _model_values(table: dict) -> list[tuple[str, str]]:
    """Пункты списка моделей: имя и пометка «по умолчанию» у дефолтной."""
    return [(name, name + ("  (по умолчанию)" if spec.default else ""))
            for name, spec in table.items()]


def _essentials(console: Console, cfg: Config, tokens: list[str]) -> bool:
    """Основные вопросы: слайды, раскладка, язык конспекта, модель синтеза."""
    default_llm = cfg.default_name("llm")
    name = console.select("Модель синтеза", "Модель, которая пишет конспект (--llm-model):",
                          _model_values(cfg.llm), default=default_llm)
    if name is None:
        return False
    # Дефолтную модель в команду не пишем: строка сводки остаётся короткой.
    if name != default_llm:
        tokens += ["--llm-model", name]
    return True


def _advanced(console: Console, cfg: Config, tokens: list[str]) -> bool:
    """Блок «дополнительно»: языки, модели этапов, числа, --force и --skip-*."""
    language = console.select("Язык речи", "Подсказка распознавания речи (--language; "
                             "auto — детекция, для двуязычных лекций):",
                             [("auto", "auto"), ("ru", "ru"), ("en", "en")], default="auto")
    if language is None:
        return False
    tokens += ["--language", language]
    for category, flag in (("stt", "--stt"), ("vision", "--vision-model")):
        name = console.select(f"Модель {category}", f"Модель этапа ({flag}):",
                              _model_values(getattr(cfg, category)),
                              default=cfg.default_name(category))
        if name is None:
            return False
        if name != cfg.default_name(category):
            tokens += [flag, name]
    for flag, text, default, low, high, cast in NUMBER_QUESTIONS:
        value = _ask_number(console, flag, text, default, low, high, cast)
        if value is None:
            return False
        tokens += [flag, value]
    if console.confirm("Пересборка", "Игнорировать кэш и пересобрать все этапы (--force)?",
                       default=False):
        tokens.append("--force")
    skipped = console.select_many("Пропустить этапы", "Отметьте этапы, которые нужно пропустить:",
                                  list(SKIP_CHOICES))
    if skipped is None:
        return False
    tokens.extend(skipped)
    return True


def run_wizard(cfg: Config, config_path: str | None = None, console: Console | None = None,
               start_dir: Path | None = None) -> list[str] | None:
    """Провести пользователя по параметрам прогона; вернуть аргументы командной строки.

    Возвращаются аргументы без имени программы: main переразбирает их тем же
    парсером, поэтому напечатанную сводку можно скопировать в терминал. None —
    отмена: Esc или Ctrl-C в любом диалоге, отказ на последнем подтверждении.
    """
    console = console or TtyConsole()
    start_dir = start_dir or Path.cwd()
    tokens: list[str] = ["--config", config_path] if config_path else []

    try:
        input_path = _choose_input(console, start_dir)
        if input_path is None:
            return None
        slides = _choose_slides(console, input_path, start_dir)
        if slides is None:
            return None
        for path in slides:
            tokens += ["--slides", str(path)]
        flat = console.confirm(
            "Раскладка результата",
            "Плоская: результат рядом с записью (--flat)?\n"
            "«Нет» — отдельный каталог <имя>/ с notes.md.",
            default=True)
        if flat:
            tokens.append("--flat")
        output_language = console.select(
            "Язык конспекта",
            "Язык, на котором модель пишет конспект (--output-language):",
            [("ru", "Русский"), ("en", "English")], default="ru")
        if output_language is None:
            return None
        tokens += ["--output-language", output_language]
        if not _essentials(console, cfg, tokens):
            return None
        if console.confirm("Дополнительные параметры",
                           "Настроить fps, пороги, языки и модели распознавания/описания, "
                           "--force и --skip-*?", default=False) and not _advanced(console, cfg, tokens):
            return None
    except (KeyboardInterrupt, EOFError):
        return None

    # Медиафайл идёт последним: напечатанная команда читается как обычный
    # вызов CLI — сначала флаги, потом вход.
    tokens.append(str(input_path))
    print("Команда: absorbent " + shlex.join(tokens))
    if not console.confirm("Запуск", "Запустить обработку?", default=True):
        return None
    return tokens
