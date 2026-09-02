"""Постпроцессинг готового notes.md: выравнивание таблиц, HTML, PDF.

Вызовы внешних программ (fmt-md-tables, pandoc) через subprocess — скрипты
не интегрируются в проект, а используются как установленные инструменты.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

# Пути к ресурсам пакета: CSS-шаблон GitHub для HTML-версии конспекта.
ASSETS_DIR = Path(__file__).parent / "assets"
GITHUB_CSS = ASSETS_DIR / "github-markdown.css"

# Имена внешних инструментов (ищутся в PATH).
FMT_TABLES = "fmt-md-tables"
PANDOC = "pandoc"

# Шрифты PDF: DejaVu покрывает кириллицу (как в лабораторных отчётах HSE).
PDF_FONTS = {
    "mainfont": "DejaVu Serif",
    "sansfont": "DejaVu Sans",
    "monofont": "DejaVu Sans Mono",
}


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


def _write_body_md(notes_md: Path, tmp: Path) -> tuple[str | None, Path]:
    """Тело notes.md (без первого заголовка) во временный файл.

    Изображения в notes.md ссылаются на images/ относительно исходного
    файла, поэтому pandoc запускается с --resource-path на его каталог.
    """
    raw = notes_md.read_text(encoding="utf-8")
    title, body = _extract_title(raw)
    body_md = tmp / "notes.md"
    body_md.write_text(body, encoding="utf-8")
    return title, body_md


def _html_parts() -> tuple[Path, Path, Path]:
    """Временные файлы: <style> с CSS, открывающий и закрывающий article."""
    temp_dir = Path(tempfile.mkdtemp(prefix="absorbent-html-"))
    header = temp_dir / "header.html"
    before = temp_dir / "before.html"
    after = temp_dir / "after.html"
    css = GITHUB_CSS.read_text(encoding="utf-8")
    container = (
        ".markdown-body { box-sizing: border-box; min-width: 200px; "
        "max-width: 980px; margin: 0 auto; padding: 45px; }\n"
        "@media (max-width: 767px) { .markdown-body { padding: 15px; } }"
    )
    header.write_text(f"<style>\n{css}\n{container}\n</style>\n", encoding="utf-8")
    before.write_text('<article class="markdown-body">\n', encoding="utf-8")
    after.write_text("</article>\n", encoding="utf-8")
    return header, before, after


def fmt_tables(notes_md: Path) -> None:
    """Выровнять pipe-таблицы notes.md внешним скриптом fmt-md-tables."""
    subprocess.run(
        [_tool(FMT_TABLES), "--no-backup", str(notes_md)],
        check=True,
        capture_output=True,
        text=True,
    )


def render_html(notes_md: Path, out_html: Path) -> None:
    """Собрать самодостаточный notes.html: css и изображения встроены.

    Рецепт pandoc: --standalone + --embed-resources (картинки data-URI),
    CSS инлайн через -H, обёртка .markdown-body через -B/-A (CSS-тема
    github-markdown-css требует контейнер с этим классом), формулы — MathML.
    """
    with tempfile.TemporaryDirectory(prefix="absorbent-html-") as tmp:
        work = Path(tmp)
        title, body_md = _write_body_md(notes_md, work)
        header, before, after = _html_parts()
        cmd = [
            _tool(PANDOC), str(body_md), "-o", str(out_html),
            "--standalone", "--embed-resources", "--mathml",
            "--resource-path", str(notes_md.parent),
            "-H", str(header), "-B", str(before), "-A", str(after),
        ]
        if title:
            cmd += ["--metadata", f"pagetitle={title}"]
        subprocess.run(cmd, check=True, capture_output=True, text=True)


def render_pdf(notes_md: Path, out_pdf: Path) -> None:
    """Собрать notes.pdf через pandoc с движком lualatex.

    Русский текст требует шрифты с кириллицей (DejaVu); поля 2 см — как в
    лабораторных отчётах; первый заголовок notes.md становится title PDF.
    """
    with tempfile.TemporaryDirectory(prefix="absorbent-pdf-") as tmp:
        work = Path(tmp)
        title, body_md = _write_body_md(notes_md, work)
        cmd = [
            _tool(PANDOC), str(body_md), "-o", str(out_pdf),
            "--pdf-engine=lualatex", "-V", "geometry:margin=2cm",
            "--resource-path", str(notes_md.parent),
        ]
        for key, value in PDF_FONTS.items():
            cmd += ["-V", f"{key}={value}"]
        if title:
            cmd += ["--metadata", f"title={title}"]
        subprocess.run(cmd, check=True, capture_output=True, text=True)


def render_all(notes_md: Path) -> tuple[Path, Path]:
    """Постпроцессинг: выровнять таблицы, собрать notes.html и notes.pdf.

    Таблицы notes.md выравниваются in-place скриптом fmt-md-tables.
    Возвращает пути (html, pdf) рядом с notes.md.
    """
    notes_md = Path(notes_md)
    fmt_tables(notes_md)
    out_html = notes_md.with_name("notes.html")
    out_pdf = notes_md.with_name("notes.pdf")
    render_html(notes_md, out_html)
    render_pdf(notes_md, out_pdf)
    return out_html, out_pdf
