"""Выравнивание pipe-таблиц в Markdown и в исходниках Python.

В Markdown строку таблицы открывает `|`. В Python такие же строки живут в
комментариях (`# | ... |`) — например, в jupytext-ноутбуках, — поэтому маркер
комментария выбирается по расширению файла и сохраняется дословно при записи.

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
from textwrap import wrap

# Маркер комментария, открывающий строку таблицы, по расширению файла ('' — чистый Markdown).
COMMENT_BY_SUFFIX: dict[str, str] = {
    ".md": "",
    ".markdown": "",
    ".py": "#",
    ".pyi": "#",
}

# Ширина таблицы по умолчанию: её превышение включает перенос ячеек.
DEFAULT_MAX_WIDTH = 120


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


def align_table(rows: list[str], max_width: int) -> list[str]:
    """Выровнять одну pipe-таблицу; при превышении `max_width` включить перенос."""
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

    # Полная ширина: | + сумма (ширина + 2 пробела на столбец) + разделители + |.
    total = sum(nat) + 3 * ncols + 1

    if total <= max_width or ncols <= 1:
        # Перенос не нужен — простое дополнение пробелами.
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

    # --- Режим переноса ---
    # Минимальная ширина столбца = самое длинное неразрывное слово, снова
    # без строки-разделителя.
    min_w = [3] * ncols
    for i, cells in enumerate(parsed):
        if i == sep_idx:
            continue
        for j, cell in enumerate(cells):
            if j < ncols and cell.strip():
                words = cell.split()
                min_w[j] = max(min_w[j], max((len(w) for w in words), default=3))

    available = max_width - 3 * ncols - 1  # Место только под содержимое.
    available = max(available, sum(min_w))

    # Пропорциональное распределение доступного места.
    total_nat = max(sum(nat), 1)
    alloc = [max(min_w[j], int(available * nat[j] / total_nat)) for j in range(ncols)]

    # Подгонка под точный размер: излишек места урезается, недостаток отдаётся
    # последнему столбцу.
    diff = sum(alloc) - available
    if diff < 0:
        alloc[-1] -= diff
    else:
        # Минимальные ширины всегда помещаются в доступное место, поэтому
        # счётчик излишка гарантированно доходит до нуля.
        order = sorted(range(ncols), key=lambda j: alloc[j], reverse=True)
        index = 0
        while diff > 0:
            j = order[index]
            cut = min(diff, alloc[j] - min_w[j])
            alloc[j] -= cut
            diff -= cut
            index += 1

    out = []
    for i, cells in enumerate(parsed):
        if i == sep_idx:
            out.append("|" + "|".join(
                fmt_sep(alloc[j], alignments[j]) for j in range(ncols)
            ) + "|")
            continue

        # Перенос каждой ячейки по её ширине.
        wrapped = []
        max_lines = 0
        for j in range(ncols):
            cell = cells[j] if j < len(cells) else ""
            lines = wrap(cell, width=alloc[j]) or [""]
            wrapped.append(lines)
            max_lines = max(max_lines, len(lines))

        # Короткие ячейки дополняются пустыми строками.
        for j in range(ncols):
            while len(wrapped[j]) < max_lines:
                wrapped[j].append("")

        for line_no in range(max_lines):
            out.append("|" + "|".join(
                fmt_cell(wrapped[j][line_no], alloc[j]) for j in range(ncols)
            ) + "|")

    return out


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


def process_file(path: Path, max_width: int, backup_dir: Path | None,
                 quiet: bool = False) -> bool:
    """Выровнять все таблицы файла Markdown или Python; True при успехе.

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
    lines = original.split("\n")
    tables = find_tables(lines, marker)

    if not tables:
        if not quiet:
            print(f"{path}: no tables found")
        return True

    # Обработка снизу вверх сохраняет номера строк.
    for start, end in reversed(tables):
        prefix, payloads = read_table(lines, start, end, marker)
        lines[start:end + 1] = [prefix + row for row in align_table(payloads, max_width)]

    result = "\n".join(lines)

    if result == original:
        if not quiet:
            print(f"{path}: already aligned")
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
        description="Выравнивание pipe-таблиц в Markdown и в исходниках Python.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Примеры:\n"
               "  absorbent-fmt-tables README.md\n"
               "  absorbent-fmt-tables notebook.py\n"
               "  absorbent-fmt-tables -w 100 docs/*.md\n"
               "  absorbent-fmt-tables --dry-run README.md\n"
               "  absorbent-fmt-tables --no-backup README.md",
    )
    parser.add_argument(
        "files", nargs="+", type=Path,
        help="файлы для обработки: .md (таблицы как есть) и .py/.pyi (таблицы в комментариях #)",
    )
    parser.add_argument(
        "-w", "--max-width", type=int, default=DEFAULT_MAX_WIDTH,
        help="предельная полная ширина таблицы до включения переноса ячеек (по умолчанию 120)",
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
            file_lines = content.split("\n")
            marker = comment_marker(path)
            file_tables = find_tables(file_lines, marker)
            if file_tables:
                print(f"\n─── {path} ({len(file_tables)} table(s)) ───")
                for start, end in file_tables:
                    prefix, payloads = read_table(file_lines, start, end, marker)
                    aligned = align_table(payloads, args.max_width)
                    print(f"  lines {start + 1}-{end + 1}:")
                    for line in aligned:
                        print(f"    {prefix}{line}")
            else:
                print(f"{path}: no tables")
        else:
            process_file(path, args.max_width, backup_dir)


if __name__ == "__main__":
    main()
