"""Постпроцессинг Markdown для pandoc: выравнивание pipe-таблиц и границы блоков.

В Markdown строку таблицы открывает `|`. В Python такие же строки живут в
комментариях (`# | ... |`) — например, в jupytext-ноутбуках, — поэтому маркер
комментария выбирается по расширению файла и сохраняется дословно при записи.

Строки таблицы не переносятся никогда: pandoc читает pipe-таблицу, только когда
каждая её логическая строка занимает одну физическую. Перенесённая ячейка
становится лишней строкой таблицы, а перенесённая шапка разваливает таблицу
целиком — заголовок и строка-разделитель перестают быть соседними, и на месте
таблицы остаётся абзац с вертикальными чертами.

Пустая строка перед блоком тоже обязательна: ни таблица, ни список не прерывают
абзац, поэтому «…текст:\n1. пункт» pandoc отдаёт одной строкой текста, а
«…текст:\n| a | b |» — абзацем вместо таблицы. Слипшиеся блоки разделяет
[separate_blocks()].

Модуль перенесён из внешнего скрипта `fmt-md-tables` и работает и как
библиотека для этапа постпроцессинга (`process_file`), и как консольная
команда `absorbent-fmt-tables`.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

# Маркер комментария, открывающий строку таблицы, по расширению файла ('' — чистый Markdown).
COMMENT_BY_SUFFIX: dict[str, str] = {
    ".md": "",
    ".markdown": "",
    ".py": "#",
    ".pyi": "#",
}

# Забор кодового блока: три и более обратных кавычки или тильды.
FENCE = re.compile(r"^\s*(?:`{3,}|~{3,})")

# Выключная формула: строка начинается с двух долларов.
MATH = re.compile(r"^\s*\$\$")

# Строка-пункт списка: маркер `-`, `*`, `+` или номер до девяти цифр.
LIST_ITEM = re.compile(r"^ {0,3}(?:[-*+]|\d{1,9}[.)])(?:\s|$)")

# Начало слипшегося списка: маркер или номер «1». Больший номер стоит в середине
# списка, а перед ним абзац — продолжение предыдущего пункта, не новый список.
LIST_START = re.compile(r"^ {0,3}(?:[-*+]|1[.)])(?:\s|$)")


def comment_marker(path: Path) -> str:
    """Маркер комментария, открывающий строки таблиц в `path`.

    Неизвестные расширения трактуются как чистый Markdown.
    """
    return COMMENT_BY_SUFFIX.get(path.suffix.lower(), "")


def table_row(line: str, marker: str) -> tuple[str, str] | None:
    """Разобрать `line` в (префикс комментария, полезная нагрузка строки таблицы).

    `marker` — маркер комментария файла ('' для Markdown): с маркером
    учитываются только закомментированные строки, а префикс сохраняется
    дословно, чтобы переписанные строки остались комментариями.
    """
    if marker:
        match = re.match(r"[ \t]*" + re.escape(marker) + r"[ \t]?", line)
        if not match:
            return None
        prefix, line = match.group(0), line[match.end():]
    else:
        prefix = ""
    payload = line.strip()
    if payload.startswith("|") and payload.endswith("|"):
        return prefix, payload
    return None


def parse_row(line: str) -> list[str]:
    """Разобрать строку pipe-таблицы на ячейки; внешние `|` и пробелы срезаются."""
    s = line.strip().removeprefix("|").removesuffix("|")
    return [c.strip() for c in s.split("|")]


def is_sep(cells: list[str]) -> bool:
    """Проверить, что строка — разделитель (например, |---|---|)."""
    return all(re.fullmatch(r":?-{3,}:?", c.strip()) for c in cells)


def sep_alignment(cell: str) -> str:
    """Стиль выравнивания из ячейки разделителя: '-', ':-', '-:' или ':-:'."""
    cell = cell.strip()
    if cell.startswith(":") and cell.endswith(":"):
        return ":-:"
    if cell.startswith(":"):
        return ":-"
    if cell.endswith(":"):
        return "-:"
    return "-"


def fmt_cell(text: str, width: int) -> str:
    """Отформатировать ячейку с выравниванием влево по `width`."""
    return f" {text.ljust(width)} "


def fmt_sep(width: int, alignment: str) -> str:
    """Отформатировать ячейку разделителя для ширины содержимого и стиля."""
    inner = width + 2  # Ширина содержимого плюс два пробела отступов.
    if alignment == ":-:":
        return ":" + "-" * (inner - 2) + ":"
    elif alignment == ":-":
        return ":" + "-" * (inner - 1)
    elif alignment == "-:":
        return "-" * (inner - 1) + ":"
    else:
        return "-" * inner


def align_table(rows: list[str]) -> list[str]:
    """Выровнять одну pipe-таблицу по естественным ширинам столбцов.

    Ширина строки не ограничивается: перенос ячейки на вторую физическую строку
    pandoc читает как ещё одну строку таблицы, а перенесённая шапка отодвигает
    разделитель от заголовка и отменяет таблицу целиком.
    """
    parsed = [parse_row(r) for r in rows]
    if not parsed or not parsed[0]:
        return rows

    ncols = len(parsed[0])

    # Строка-разделитель и стили выравнивания столбцов.
    sep_idx = -1
    alignments: list[str] = []
    for i, cells in enumerate(parsed):
        if is_sep(cells):
            sep_idx = i
            alignments = [sep_alignment(c) for c in cells]
            break

    if sep_idx < 0:
        return rows  # Не таблица.

    # Естественные (исходные) ширины столбцов по всем ячейкам, кроме разделителя:
    # его дефисы считались бы самым длинным словом и раздували столбец.
    nat = [0] * ncols
    for i, cells in enumerate(parsed):
        if i == sep_idx:
            continue
        for j, cell in enumerate(cells):
            if j < ncols:
                nat[j] = max(nat[j], len(cell))
    nat = [max(w, 3) for w in nat]  # Минимальная ширина под разделитель.

    out = []
    for i, cells in enumerate(parsed):
        if i == sep_idx:
            out.append("|" + "|".join(
                fmt_sep(nat[j], alignments[j]) for j in range(ncols)
            ) + "|")
        else:
            out.append("|" + "|".join(
                fmt_cell(cells[j] if j < len(cells) else "", nat[j])
                for j in range(ncols)
            ) + "|")
    return out


def glued_block(lines: list[str], i: int) -> bool:
    """Нужна ли пустая строка перед `lines[i]`, чтобы pandoc начал там новый блок.

    Абзац прерывают заголовок, цитата, забор и формула; таблица и список без
    пустой строки остаются его продолжением. Отделять нужно ровно их: шапку
    таблицы (строку `| ... |` с разделителем следом) и первый пункт списка.
    """
    if i == 0 or not lines[i - 1].strip():
        return False
    if table_row(lines[i - 1], "") is not None:
        return False  # Предыдущая строка — строка таблицы, разделять нечего.
    if i + 1 < len(lines) and is_sep(parse_row(lines[i + 1])):
        return table_row(lines[i], "") is not None
    return (LIST_START.match(lines[i]) is not None
            and LIST_ITEM.match(lines[i - 1]) is None)


def separate_blocks(lines: list[str]) -> list[str]:
    """Отделить пустой строкой таблицы и списки, слипшиеся с абзацем.

    Строки внутри кодовых блоков и выключных формул не трогаются: там `|` и
    дефисы — не разметка, и вставленная пустая строка разорвала бы блок.
    """
    out: list[str] = []
    fence = False  # Идёт кодовый блок в заборе.
    math = False   # Идёт выключная формула между двумя `$$`.
    for i, line in enumerate(lines):
        if not fence and not math and glued_block(lines, i):
            out.append("")
        out.append(line)
        if fence:
            if FENCE.match(line):
                fence = False
        elif MATH.match(line) and line.count("$$") % 2:
            # Однострочная `$$x = 1$$` — чётное число `$$` — блок не открывает.
            math = not math
        elif FENCE.match(line):
            fence = True
    return out


def format_lines(lines: list[str], marker: str = "") -> tuple[list[str], list[tuple[int, int]]]:
    """Отделить блоки от абзаца и выровнять таблицы; вернуть строки и отрезки таблиц.

    Отрезки считаются по возвращённым строкам, поэтому вставленные пустые строки
    сдвигают их относительно исходного файла. В исходниках Python блоки не
    отделяются: пустая строка разорвала бы комментарий, в котором живёт таблица.
    """
    if not marker:
        lines = separate_blocks(lines)
    tables = find_tables(lines, marker)
    for start, end in reversed(tables):
        prefix, payloads = read_table(lines, start, end, marker)
        lines[start:end + 1] = [prefix + row for row in align_table(payloads)]
    return lines, tables


def find_tables(lines: list[str], marker: str = "") -> list[tuple[int, int]]:
    """Найти (start, end) включительные номера строк всех pipe-таблиц в `lines`."""
    tables: list[tuple[int, int]] = []
    i = 0
    while i < len(lines):
        if table_row(lines[i], marker) is None:
            i += 1
            continue
        start = i
        while i < len(lines) and table_row(lines[i], marker) is not None:
            i += 1
        end = i - 1
        payloads = read_table(lines, start, end, marker)[1]
        if any(is_sep(parse_row(p)) for p in payloads):
            tables.append((start, end))
    return tables


def read_table(lines: list[str], start: int, end: int, marker: str) -> tuple[str, list[str]]:
    """Вернуть префикс комментария и полезные нагрузки строк таблицы на отрезке."""
    rows = [table_row(line, marker) for line in lines[start:end + 1]]
    prefix = next((row[0] for row in rows if row), "")
    return prefix, [row[1] for row in rows if row]


def process_file(path: Path, backup_dir: Path | None,
                 quiet: bool = False) -> bool:
    """Выровнять все таблицы файла Markdown или Python; True при успехе.

    Кроме выравнивания, в Markdown отделяются пустыми строками таблицы и списки,
    слипшиеся с абзацем (см. [separate_blocks()]).
    `backup_dir` — каталог для копии исходного файла (None — без копии).
    `quiet` подавляет вывод: так модуль зовёт конвейер постпроцессинга.
    """
    try:
        original = path.read_text(encoding="utf-8")
    except OSError as exc:
        if not quiet:
            print(f"Error reading {path}: {exc}", file=sys.stderr)
        return False

    marker = comment_marker(path)
    lines, tables = format_lines(original.split("\n"), marker)
    result = "\n".join(lines)

    if result == original:
        if not quiet:
            print(f"{path}: no tables found" if not tables else f"{path}: already aligned")
        return True

    if backup_dir:
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / path.name
        shutil.copy2(path, backup_path)
        if not quiet:
            print(f"Backup: {backup_path}")

    path.write_text(result, encoding="utf-8")
    if not quiet:
        print(f"{path}: aligned {len(tables)} table(s)")
    return True


def main() -> None:
    """Точка входа `absorbent-fmt-tables`: разбор аргументов и обработка файлов."""
    parser = argparse.ArgumentParser(
        prog="absorbent-fmt-tables",
        description="Выравнивание pipe-таблиц и границ блоков в Markdown и в исходниках Python.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Примеры:\n"
               "  absorbent-fmt-tables README.md\n"
               "  absorbent-fmt-tables notebook.py\n"
               "  absorbent-fmt-tables --dry-run README.md\n"
               "  absorbent-fmt-tables --no-backup README.md",
    )
    parser.add_argument(
        "files", nargs="+", type=Path,
        help="файлы для обработки: .md (таблицы как есть) и .py/.pyi (таблицы в комментариях #)",
    )
    parser.add_argument(
        "-n", "--dry-run", action="store_true",
        help="показать изменения, не трогая файлы",
    )
    parser.add_argument(
        "--no-backup", action="store_true",
        help="не создавать копию перед правкой",
    )
    parser.add_argument(
        "--backup-dir", type=Path, default=Path("/tmp"),
        help="каталог для копий (по умолчанию /tmp)",
    )

    args = parser.parse_args()
    backup_dir = None if args.no_backup else args.backup_dir

    for path in args.files:
        if not path.exists():
            print(f"File not found: {path}", file=sys.stderr)
            continue

        if args.dry_run:
            content = path.read_text(encoding="utf-8")
            marker = comment_marker(path)
            prepared, tables = format_lines(content.split("\n"), marker)
            if tables:
                print(f"\n─── {path} ({len(tables)} table(s)) ───")
                for start, end in tables:
                    print(f"  lines {start + 1}-{end + 1}:")
                    for line in prepared[start:end + 1]:
                        print(f"    {line}")
            else:
                print(f"{path}: no tables")
        else:
            process_file(path, backup_dir)


if __name__ == "__main__":
    main()
