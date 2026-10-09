"""Постпроцессинг готового notes.md: выравнивание таблиц, HTML, PDF.

Таблицы выравнивает встроенный модуль [md_tables]; внешняя программа только
одна — pandoc, он собирает HTML и (через lualatex) PDF.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from absorbent.md_tables import DEFAULT_MAX_WIDTH, process_file

# Пути к ресурсам пакета: CSS-шаблон GitHub для HTML-версии конспекта.
ASSETS_DIR = Path(__file__).parent / "assets"
GITHUB_CSS = ASSETS_DIR / "github-markdown.css"

# Имя внешнего инструмента (ищется в PATH).
PANDOC = "pandoc"

# Шрифты PDF: DejaVu покрывает кириллицу (как в лабораторных отчётах HSE).
PDF_FONTS = {
    "mainfont": "DejaVu Serif",
    "sansfont": "DejaVu Sans",
    "monofont": "DejaVu Sans Mono",
}

# Формат и поля PDF: A4 с полями 1.8 см вмещает более крупные врезки слайдов,
# чем штатный формат pandoc.
PDF_GEOMETRY = "a4paper,margin=1.8cm"

# Предельная ширина полосы контента в HTML-версии: штатная таблица стилей
# pandoc ограничивает её 36em (576 px), из-за чего на широком экране конспект
# остаётся узкой колонкой.
HTML_COLUMN_PX = 1280

# Медиазапросы github-markdown.css: по ним ищутся цвета фона темы.
_DARK_QUERY = "@media (prefers-color-scheme: dark)"
_LIGHT_QUERY = "@media (prefers-color-scheme: light)"
_BG_DEFAULT = re.compile(r"--bgColor-default:\s*([^;]+);")

# Надстройка над преамбулой pandoc для PDF.
_LATEX_HEADER = r"""
% Плавающий figure LaTeX может поставить туда, где рисунок с подписью не
% помещается, и низ страницы при этом обрезается. Встраиваем рисунок в поток
% текста: он занимает место по порядку и переносится на новую страницу целиком.
\makeatletter
\renewenvironment{figure}[1][]{%
  \par\addvspace{\intextsep}%
  \def\@captype{figure}%
  \begin{minipage}{\linewidth}\centering
}{%
  \end{minipage}\par\addvspace{\intextsep}%
}
\makeatother
% Длинные слова и адреса переносятся, а не вылезают за правое поле.
\setlength{\emergencystretch}{6em}
"""


def _tool(name: str) -> str:
    """Путь к внешнему инструменту; ошибка, если его нет в PATH."""
    path = shutil.which(name)
    if not path:
        raise FileNotFoundError(
            f"не найден внешний инструмент '{name}'; установите его или "
            f"запустите конвейер с --skip-render"
        )
    return path


def _extract_title(text: str) -> tuple[str | None, str]:
    """Отделить первый заголовок '# ...' от тела notes.md.

    Возвращает (title, остальной текст): заголовок уходит в метаданные
    pandoc (title-блок HTML / maketitle PDF), чтобы не дублироваться в теле.
    """
    lines = text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("# "):
            title = stripped[2:].strip()
            body = "\n".join(lines[i + 1:]).strip("\n")
            return title, body
    return None, text


def _write_body_md(notes_md: Path, tmp: Path, keep_title: bool = False) -> tuple[str | None, Path]:
    """Тело notes.md во временный файл плюс отделённый первый заголовок.

    Изображения в notes.md ссылаются на images/ относительно исходного
    файла, поэтому pandoc запускается с --resource-path на его каталог.
    В HTML первый заголовок остаётся в теле (нужен видимый заголовок), в
    PDF он служит метаданным title для блока maketitle.
    """
    raw = notes_md.read_text(encoding="utf-8")
    title, body = _extract_title(raw)
    body_md = tmp / "notes.md"
    body_md.write_text(raw if keep_title or title is None else body, encoding="utf-8")
    return title, body_md


def _bg_default(css: str, query: str, fallback: str) -> str:
    """Цвет фона конспекта из блока github-markdown.css для схемы `query`.

    Значения живут внутри .markdown-body и не наследуются до элемента html,
    поэтому их приходится доставать из таблицы стилей текстом; запас
    `fallback` нужен, если шаблон подменили и медиазапроса в нём нет.
    """
    start = css.find(query)
    match = _BG_DEFAULT.search(css, start) if start >= 0 else None
    return match.group(1).strip() if match else fallback


def _page_css(css: str) -> str:
    """CSS-надстройка над github-markdown.css: фон страницы и ширина полосы.

    Штатные стили pandoc задают белый фон html, полосу 36em и тёмные ссылки:
    в тёмной теме это даёт белые поля вокруг конспекта и нечитаемые ссылки.
    Фон html растягивается на всё окно, полоса контента адаптируется к
    ширине экрана и ограничена HTML_COLUMN_PX.
    """
    light = _bg_default(css, _LIGHT_QUERY, "#ffffff")
    dark = _bg_default(css, _DARK_QUERY, "#0d1117")
    return f"""
html, body {{ margin: 0; padding: 0; max-width: none; }}
html {{ color-scheme: light dark; background-color: {light}; }}
@media (prefers-color-scheme: dark) {{ html {{ background-color: {dark}; }} }}
.markdown-body.page {{ box-sizing: border-box; min-height: 100vh;
  padding: 2.5rem clamp(1rem, 4vw, 4rem) 6rem; }}
.page-inner {{ max-width: {HTML_COLUMN_PX}px; margin: 0 auto; }}
@media (max-width: 767px) {{ .markdown-body.page {{ padding: 1.25rem 1rem 4rem; }} }}
"""


def _html_parts(tmp: Path) -> tuple[Path, Path, Path]:
    """Временные файлы: <style> с CSS, открывающий и закрывающий контейнер."""
    header = tmp / "header.html"
    before = tmp / "before.html"
    after = tmp / "after.html"
    css = GITHUB_CSS.read_text(encoding="utf-8")
    header.write_text(f"<style>\n{css}\n{_page_css(css)}\n</style>\n", encoding="utf-8")
    before.write_text(
        '<div class="markdown-body page">\n<div class="page-inner">\n', encoding="utf-8")
    after.write_text("</div>\n</div>\n", encoding="utf-8")
    return header, before, after


def fmt_tables(notes_md: Path) -> None:
    """Выровнять pipe-таблицы notes.md встроенным модулем [md_tables]."""
    process_file(notes_md, DEFAULT_MAX_WIDTH, backup_dir=None, quiet=True)


def render_html(notes_md: Path, out_html: Path, lang: str = "ru") -> None:
    """Собрать самодостаточный notes.html: css и изображения встроены.

    Рецепт pandoc: --standalone + --embed-resources (картинки data-URI),
    CSS инлайн через -H, контейнеры .page/.page-inner через -B/-A
    (CSS-тема github-markdown-css требует контейнер .markdown-body),
    формулы — MathML. Язык `lang` задаёт атрибут html и переносы слов.
    """
    with tempfile.TemporaryDirectory(prefix="absorbent-html-") as tmp:
        work = Path(tmp)
        title, body_md = _write_body_md(notes_md, work, keep_title=True)
        header, before, after = _html_parts(work)
        cmd = [
            _tool(PANDOC), str(body_md), "-o", str(out_html),
            "--standalone", "--embed-resources", "--mathml",
            "--resource-path", str(notes_md.parent),
            "-V", f"lang={lang}",
            "-H", str(header), "-B", str(before), "-A", str(after),
        ]
        if title:
            cmd += ["--metadata", f"pagetitle={title}"]
        subprocess.run(cmd, check=True, capture_output=True, text=True)


def render_pdf(notes_md: Path, out_pdf: Path, lang: str = "ru") -> None:
    """Собрать notes.pdf через pandoc с движком lualatex.

    Русский текст требует шрифты с кириллицей (DejaVu) и включённых правил
    переноса: язык `lang` подключает babel с нужными шаблонами переноса и
    переводит подписи рисунков. Первый заголовок notes.md становится title
    PDF; _LATEX_HEADER убирает плавучесть рисунков и правит вёрстку страниц.
    """
    with tempfile.TemporaryDirectory(prefix="absorbent-pdf-") as tmp:
        work = Path(tmp)
        title, body_md = _write_body_md(notes_md, work)
        header = work / "header.tex"
        header.write_text(_LATEX_HEADER, encoding="utf-8")
        cmd = [
            _tool(PANDOC), str(body_md), "-o", str(out_pdf),
            "--pdf-engine=lualatex", "-V", f"geometry:{PDF_GEOMETRY}",
            "-V", f"lang={lang}",
            "--resource-path", str(notes_md.parent),
            "-H", str(header),
        ]
        for key, value in PDF_FONTS.items():
            cmd += ["-V", f"{key}={value}"]
        if title:
            cmd += ["--metadata", f"title={title}"]
        subprocess.run(cmd, check=True, capture_output=True, text=True)


def outputs_up_to_date(notes_md: Path, out_html: Path, out_pdf: Path) -> bool:
    """Готовы ли HTML и PDF: оба на месте и не старше notes.md.

    Отсутствие любой из версий — повод пересобрать (прогон прервался на
    pandoc или lualatex, либо постпроцессинг отключали флагом --skip-render).
    Правка notes.md сдвигает её mtime, поэтому устаревшие версии тоже
    пересобираются.
    """
    if not (out_html.exists() and out_pdf.exists()):
        return False
    notes_md = Path(notes_md)
    stamp = notes_md.stat().st_mtime
    return min(out_html.stat().st_mtime, out_pdf.stat().st_mtime) >= stamp


def render_all(notes_md: Path, out_html: Path | None = None,
               out_pdf: Path | None = None, lang: str = "ru") -> tuple[Path, Path]:
    """Постпроцессинг: выровнять таблицы, собрать HTML и PDF.

    Таблицы notes.md выравниваются in-place модулем [md_tables].
    Пути HTML и PDF по умолчанию — notes.html и notes.pdf рядом с notes.md,
    как в раскладке <имя>/. Плоский режим передаёт <имя>.html и <имя>.pdf:
    относительные ссылки на картинки разрешает --resource-path по каталогу
    notes.md, поэтому имена файлов результата произвольны. `lang` — язык
    конспекта (он же язык переносов в HTML и PDF). Пропуск уже собранных
    версий — за вызывающей стороной (см. outputs_up_to_date).
    """
    notes_md = Path(notes_md)
    fmt_tables(notes_md)
    out_html = Path(out_html) if out_html is not None else notes_md.with_name("notes.html")
    out_pdf = Path(out_pdf) if out_pdf is not None else notes_md.with_name("notes.pdf")
    render_html(notes_md, out_html, lang=lang)
    render_pdf(notes_md, out_pdf, lang=lang)
    return out_html, out_pdf
