"""Страницы презентаций лектора: конвертация, растеризация, кэш страниц.

Внешние программы (LibreOffice для не-PDF и poppler для растеризации)
вызываются через subprocess; сетевых вызовов этап не делает.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

# Расширения файлов презентаций, которые ищутся при передаче каталога.
SLIDE_EXTENSIONS: frozenset[str] = frozenset(
    {".pdf", ".pptx", ".ppt", ".odp", ".key", ".docx", ".doc", ".odt"}
)

# Имя файла-подписи источника в его каталоге кэша.
SIGNATURE_NAME = "signature.json"

# Шаблон имён страниц, которые пишет pdftoppm: page-01.png, page-001.png.
PAGE_GLOB = "page-*.png"

# Ширина номера в именах итоговых картинок: slide_001.png сортируется как число.
SLIDE_INDEX_WIDTH = 3


class SlidesError(Exception):
    """Ошибка этапа презентаций: путь, конвертация или растеризация."""


def _tool(name: str) -> str:
    """Путь к внешнему инструменту; ошибка, если его нет в PATH."""
    path = shutil.which(name)
    if not path:
        raise FileNotFoundError(
            f"не найден внешний инструмент '{name}'; установите его или "
            f"запустите без --slides"
        )
    return path


def resolve_slide_sources(raw: list[str]) -> list[Path]:
    """Развернуть значения --slides в список файлов презентаций.

    Файл берётся как есть, каталог разворачивается в файлы SLIDE_EXTENSIONS
    (отсортированные), порядок аргументов сохраняется. Возвращаются
    абсолютные пути; повторы по каноническому пути отбрасываются.
    """
    resolved: list[Path] = []
    for value in raw:
        path = Path(value).expanduser().resolve()
        if path.is_file():
            candidates = [path]
        elif path.is_dir():
            candidates = sorted(p.resolve() for p in path.iterdir()
                                if p.is_file() and p.suffix.lower() in SLIDE_EXTENSIONS)
            if not candidates:
                raise SlidesError(f"в каталоге {value} не найдено презентаций с расширениями "
                                  + ", ".join(sorted(SLIDE_EXTENSIONS)))
        else:
            raise SlidesError(f"путь презентации не существует: {value}")
        for candidate in candidates:
            if candidate not in resolved:
                resolved.append(candidate)
    return resolved


def _page_number(path: Path) -> int:
    """Номер страницы из имени файла pdftoppm (page-NNN.png)."""
    return int(path.stem.rsplit("-", 1)[1])


def _sorted_pages(directory: Path) -> list[Path]:
    """Страницы из каталога по возрастанию номера (ширина нумерации не важна)."""
    return sorted(directory.glob(PAGE_GLOB), key=_page_number)


def _cached_pages(source_dir: Path, signature: dict) -> list[Path] | None:
    """Страницы из кэша источника, если подпись совпала и страницы на месте."""
    signature_path = source_dir / SIGNATURE_NAME
    if not signature_path.exists():
        return None
    try:
        cached = json.loads(signature_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if cached != signature:
        return None
    pages = _sorted_pages(source_dir)
    return pages or None


def _pdf_of(source: Path, source_dir: Path) -> Path:
    """PDF источника: сам файл для PDF, иначе результат конвертации LibreOffice."""
    if source.suffix.lower() == ".pdf":
        return source
    profile = source_dir / "profile"
    proc = subprocess.run(
        [_tool("soffice"), "--headless", f"-env:UserInstallation=file://{profile}",
         "--convert-to", "pdf", "--outdir", str(source_dir), str(source)],
        check=False, capture_output=True, text=True,
    )
    # Профиль LibreOffice — служебный мусор: удаляется независимо от результата.
    shutil.rmtree(profile, ignore_errors=True)
    if proc.returncode != 0:
        raise SlidesError(f"LibreOffice не сконвертировал {source.name}: {proc.stderr.strip()}")
    converted = source_dir / f"{source.stem}.pdf"
    if not converted.exists():
        raise SlidesError(f"LibreOffice не оставил PDF для {source.name}")
    return converted


def _rasterize(pdf: Path, source_dir: Path, dpi: int) -> list[Path]:
    """Растеризовать PDF в страницы page-NNN.png; страницы по возрастанию номера."""
    proc = subprocess.run(
        [_tool("pdftoppm"), "-png", "-r", str(dpi), str(pdf), str(source_dir / "page")],
        check=False, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise SlidesError(f"pdftoppm не развернул {pdf.name}: {proc.stderr.strip()}")
    pages = _sorted_pages(source_dir)
    if not pages:
        raise SlidesError(f"в файле {pdf.name} не нашлось страниц")
    return pages


def prepare_slides(sources: list[Path], images_dir: Path, work_dir: Path, dpi: int,
                   force: bool = False) -> list[dict]:
    """Разложить презентации в страницы картинок; вернуть записи страниц.

    Каждый источник получает подкаталог кэша work_dir/<номер>_<имя_без_расширения>:
    не-PDF конвертируется туда LibreOffice (--convert-to pdf), затем страницы
    растрируются pdftoppm с разрешением dpi в page-NNN.png. Кэш принимается,
    если совпала подпись signature.json (канонический путь источника, его
    размер и mtime, а также dpi) и в подкаталоге есть страницы; иначе подкаталог
    стирается и источник обрабатывается заново. Страницы копируются в images_dir
    как slide_NNN.png сквозной нумерацией по всем источникам, чужие slide_*.png
    из прошлого прогона удаляются (кадры frame_*.jpg не трогаются).

    Возвращаются записи {"image", "page", "source", "source_page"} по порядку:
    источник за источником, страница за страницей.
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []
    for index, source in enumerate(sources, start=1):
        source_dir = work_dir / f"{index:02d}_{source.stem}"
        # Подпись — по состояния источника на диске: правка файла или смена dpi
        # делают кэш недействительным.
        signature = {"source": str(source), "size": source.stat().st_size,
                     "mtime": source.stat().st_mtime, "dpi": dpi}
        pages = None if force else _cached_pages(source_dir, signature)
        if pages is None:
            shutil.rmtree(source_dir, ignore_errors=True)
            source_dir.mkdir(parents=True, exist_ok=True)
            pages = _rasterize(_pdf_of(source, source_dir), source_dir, dpi)
            (source_dir / SIGNATURE_NAME).write_text(
                json.dumps(signature, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  {source.name}: {len(pages)} страниц")
        for page in pages:
            page_number = _page_number(page)
            out_index = len(entries) + 1
            name = f"slide_{out_index:0{SLIDE_INDEX_WIDTH}d}.png"
            shutil.copy2(page, images_dir / name)
            entries.append({"image": name, "page": out_index, "source": str(source),
                            "source_page": page_number})

    names = {entry["image"] for entry in entries}
    for stale in images_dir.glob("slide_*.png"):
        if stale.name not in names:
            stale.unlink()
    return entries
