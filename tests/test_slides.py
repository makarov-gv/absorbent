"""Тесты этапа презентаций: разворачивание путей, растеризация, кэш страниц."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from absorbent.slides import (
    PAGE_GLOB,
    SIGNATURE_NAME,
    SLIDE_EXTENSIONS,
    SlidesError,
    _tool,
    prepare_slides,
    resolve_slide_sources,
)


class _Completed:
    """Результат subprocess.run с полями, которые читает этап."""

    def __init__(self, returncode: int = 0, stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = ""
        self.stderr = stderr


class FakeRunner:
    """Подделка subprocess.run: пишет страницы и сконвертированный PDF вместо программ."""

    def __init__(self, pages: int | list[int] = 2, soffice_code: int = 0, leave_pdf: bool = True,
                 pdftoppm_code: int = 0, write_pages: bool = True) -> None:
        self.fixed_pages = pages if isinstance(pages, int) else None
        self.per_source = list(pages) if isinstance(pages, list) else []
        self.soffice_code = soffice_code
        self.leave_pdf = leave_pdf
        self.pdftoppm_code = pdftoppm_code
        self.write_pages = write_pages
        self.calls: list[list[str]] = []

    def __call__(self, cmd, check=False, capture_output=True, text=True) -> object:
        command = list(cmd)
        self.calls.append(command)
        if Path(command[0]).name == "soffice":
            # Профиль LibreOffice создаётся по аргументу — этап обязан его удалить.
            env_arg = next(arg for arg in command if arg.startswith("-env:UserInstallation="))
            Path(env_arg.removeprefix("-env:UserInstallation=file://")).mkdir(parents=True)
            if self.soffice_code == 0 and self.leave_pdf:
                outdir = Path(command[command.index("--outdir") + 1])
                (outdir / f"{Path(command[-1]).stem}.pdf").write_bytes(b"%PDF-1.4")
            return _Completed(self.soffice_code, stderr="нет фильтра")
        prefix = Path(command[-1])
        count = self.fixed_pages if self.fixed_pages is not None else self.per_source.pop(0)
        if self.pdftoppm_code == 0 and self.write_pages:
            width = max(2, len(str(count)))
            for number in range(1, count + 1):
                (prefix.parent / f"{prefix.name}-{number:0{width}d}.png").write_bytes(b"png")
        return _Completed(self.pdftoppm_code, stderr="битый PDF")


@pytest.fixture
def tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """Внешние инструменты «находятся» в PATH, но не запускаются."""
    import absorbent.slides as slides_module

    monkeypatch.setattr(slides_module.shutil, "which", lambda name: f"/usr/bin/{name}")


@pytest.fixture
def slide_env(tmp_path: Path, tools: None, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Окружение этапа: подставной subprocess.run, каталоги и одна PDF-колода."""
    import absorbent.slides as slides_module

    deck = tmp_path / "deck.pdf"
    deck.write_bytes(b"%PDF-1.4 fake")
    runner = FakeRunner(pages=2)
    monkeypatch.setattr(slides_module.subprocess, "run", runner)
    return SimpleNamespace(module=slides_module, deck=deck, runner=runner,
                           images=tmp_path / "images", work=tmp_path / "work")


class TestTool:
    def test_returns_path_from_which(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import absorbent.slides as slides_module

        monkeypatch.setattr(slides_module.shutil, "which", lambda name: f"/opt/{name}")
        assert _tool("pdftoppm") == "/opt/pdftoppm"

    def test_missing_tool_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import absorbent.slides as slides_module

        monkeypatch.setattr(slides_module.shutil, "which", lambda name: None)
        with pytest.raises(FileNotFoundError, match="pdftoppm"):
            _tool("pdftoppm")


class TestResolveSlideSources:
    def test_pdf_file_kept_as_is(self, tmp_path: Path) -> None:
        deck = tmp_path / "deck.pdf"
        deck.write_bytes(b"%PDF")
        assert resolve_slide_sources([str(deck)]) == [deck]

    def test_pptx_file_accepted_as_is(self, tmp_path: Path) -> None:
        deck = tmp_path / "deck.pptx"
        deck.write_bytes(b"zip")
        assert resolve_slide_sources([str(deck)]) == [deck]

    def test_directory_expands_known_extensions_alphabetically(self, tmp_path: Path) -> None:
        (tmp_path / "b.odp").write_bytes(b"x")
        (tmp_path / "a.pdf").write_bytes(b"x")
        (tmp_path / "notes.txt").write_bytes(b"x")
        assert [p.name for p in resolve_slide_sources([str(tmp_path)])] == ["a.pdf", "b.odp"]

    def test_directory_without_slides_lists_extensions(self, tmp_path: Path) -> None:
        (tmp_path / "notes.txt").write_bytes(b"x")
        with pytest.raises(SlidesError) as err:
            resolve_slide_sources([str(tmp_path)])
        message = str(err.value)
        assert "не найдено презентаций" in message
        assert ".pdf" in message and ".pptx" in message

    def test_missing_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(SlidesError, match="не существует"):
            resolve_slide_sources([str(tmp_path / "nope.pdf")])

    def test_duplicates_collapse_by_canonical_path(self, tmp_path: Path) -> None:
        deck = tmp_path / "deck.pdf"
        deck.write_bytes(b"%PDF")
        # Вторая запись — тот же файл через «.», канонический путь совпадает.
        spelled_differently = f"{tmp_path / '.' / 'deck.pdf'}"
        assert resolve_slide_sources([str(deck), spelled_differently]) == [deck]

    def test_argument_order_preserved(self, tmp_path: Path) -> None:
        first = tmp_path / "a.pdf"
        second = tmp_path / "b.pdf"
        first.write_bytes(b"%PDF")
        second.write_bytes(b"%PDF")
        assert resolve_slide_sources([str(second), str(first)]) == [second, first]

    def test_extensions_set_documented(self) -> None:
        assert {".pdf", ".pptx", ".odp"} <= SLIDE_EXTENSIONS
        assert ".txt" not in SLIDE_EXTENSIONS


class TestPrepareSlides:
    def test_pdftoppm_command_and_copied_pages(self, slide_env: SimpleNamespace) -> None:
        entries = slide_env.module.prepare_slides([slide_env.deck], slide_env.images,
                                                  slide_env.work, 150)
        source_dir = slide_env.work / "01_deck"
        assert slide_env.runner.calls == [
            ["/usr/bin/pdftoppm", "-png", "-r", "150", str(slide_env.deck),
             str(source_dir / "page")],
        ]
        assert entries == [
            {"image": "slide_001.png", "page": 1, "source": str(slide_env.deck),
             "source_page": 1},
            {"image": "slide_002.png", "page": 2, "source": str(slide_env.deck),
             "source_page": 2},
        ]
        assert sorted(p.name for p in slide_env.images.iterdir()) == [
            "slide_001.png", "slide_002.png"]
        # Подпись источника записана вместе со страницами.
        assert (source_dir / SIGNATURE_NAME).exists()

    def test_two_sources_numbered_continuously(self, tmp_path: Path, tools: None,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
        import absorbent.slides as slides_module

        runner = FakeRunner(pages=[3, 2])
        monkeypatch.setattr(slides_module.subprocess, "run", runner)
        first = tmp_path / "deck_a.pdf"
        second = tmp_path / "deck_b.pdf"
        first.write_bytes(b"%PDF")
        second.write_bytes(b"%PDF")
        entries = prepare_slides([first, second], tmp_path / "images", tmp_path / "work", 150)
        assert [entry["image"] for entry in entries] == [
            "slide_001.png", "slide_002.png", "slide_003.png", "slide_004.png", "slide_005.png"]
        assert [entry["page"] for entry in entries] == [1, 2, 3, 4, 5]
        assert [entry["source_page"] for entry in entries] == [1, 2, 3, 1, 2]
        assert [entry["source"] for entry in entries] == [str(first)] * 3 + [str(second)] * 2
        assert (tmp_path / "work" / "02_deck_b" / SIGNATURE_NAME).exists()

    def test_second_run_restores_image_from_cache(self, slide_env: SimpleNamespace) -> None:
        entries = slide_env.module.prepare_slides([slide_env.deck], slide_env.images,
                                                  slide_env.work, 150)
        (slide_env.images / "slide_002.png").unlink()
        slide_env.runner.calls.clear()
        again = slide_env.module.prepare_slides([slide_env.deck], slide_env.images,
                                                slide_env.work, 150)
        assert slide_env.runner.calls == []
        assert again == entries
        assert (slide_env.images / "slide_002.png").exists()

    def test_changed_mtime_rerasterizes(self, slide_env: SimpleNamespace) -> None:
        slide_env.module.prepare_slides([slide_env.deck], slide_env.images, slide_env.work, 150)
        slide_env.runner.calls.clear()
        os.utime(slide_env.deck, (1_000_000.0, 1_000_000.0))
        slide_env.module.prepare_slides([slide_env.deck], slide_env.images, slide_env.work, 150)
        assert len(slide_env.runner.calls) == 1

    def test_changed_dpi_rerasterizes(self, slide_env: SimpleNamespace) -> None:
        slide_env.module.prepare_slides([slide_env.deck], slide_env.images, slide_env.work, 150)
        slide_env.runner.calls.clear()
        slide_env.module.prepare_slides([slide_env.deck], slide_env.images, slide_env.work, 300)
        assert slide_env.runner.calls[0][3] == "300"

    def test_corrupt_signature_rerasterizes(self, slide_env: SimpleNamespace) -> None:
        slide_env.module.prepare_slides([slide_env.deck], slide_env.images, slide_env.work, 150)
        slide_env.runner.calls.clear()
        (slide_env.work / "01_deck" / SIGNATURE_NAME).write_text("не json", encoding="utf-8")
        slide_env.module.prepare_slides([slide_env.deck], slide_env.images, slide_env.work, 150)
        assert len(slide_env.runner.calls) == 1

    def test_signature_without_pages_rerasterizes(self, slide_env: SimpleNamespace) -> None:
        """Подпись совпала, но страницы из кэша пропали: колода рисуется заново."""
        slide_env.module.prepare_slides([slide_env.deck], slide_env.images, slide_env.work, 150)
        slide_env.runner.calls.clear()
        for page in (slide_env.work / "01_deck").glob(PAGE_GLOB):
            page.unlink()
        entries = slide_env.module.prepare_slides([slide_env.deck], slide_env.images,
                                                  slide_env.work, 150)
        assert len(slide_env.runner.calls) == 1
        assert [entry["image"] for entry in entries] == ["slide_001.png", "slide_002.png"]

    def test_force_rerasterizes(self, slide_env: SimpleNamespace) -> None:
        slide_env.module.prepare_slides([slide_env.deck], slide_env.images, slide_env.work, 150)
        slide_env.runner.calls.clear()
        slide_env.module.prepare_slides([slide_env.deck], slide_env.images, slide_env.work, 150,
                                        force=True)
        assert len(slide_env.runner.calls) == 1

    def test_stale_slides_removed_frames_kept(self, slide_env: SimpleNamespace) -> None:
        slide_env.images.mkdir(parents=True)
        (slide_env.images / "slide_009.png").write_bytes(b"old slide")
        (slide_env.images / "frame_00_00_00.jpg").write_bytes(b"frame")
        slide_env.module.prepare_slides([slide_env.deck], slide_env.images, slide_env.work, 150)
        assert not (slide_env.images / "slide_009.png").exists()
        assert (slide_env.images / "frame_00_00_00.jpg").exists()

    def test_pptx_converted_by_libreoffice(self, tmp_path: Path, tools: None,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
        import absorbent.slides as slides_module

        runner = FakeRunner(pages=2)
        monkeypatch.setattr(slides_module.subprocess, "run", runner)
        deck = tmp_path / "deck.pptx"
        deck.write_bytes(b"zip")
        source_dir = tmp_path / "work" / "01_deck"
        entries = prepare_slides([deck], tmp_path / "images", tmp_path / "work", 150)
        assert runner.calls[0] == [
            "/usr/bin/soffice", "--headless",
            f"-env:UserInstallation=file://{source_dir / 'profile'}",
            "--convert-to", "pdf", "--outdir", str(source_dir), str(deck),
        ]
        assert runner.calls[1][-2:] == [str(source_dir / "deck.pdf"), str(source_dir / "page")]
        assert not (source_dir / "profile").exists()
        assert [entry["source_page"] for entry in entries] == [1, 2]

    def test_soffice_failure_raises(self, tmp_path: Path, tools: None,
                                    monkeypatch: pytest.MonkeyPatch) -> None:
        import absorbent.slides as slides_module

        monkeypatch.setattr(slides_module.subprocess, "run", FakeRunner(soffice_code=1))
        deck = tmp_path / "deck.pptx"
        deck.write_bytes(b"zip")
        with pytest.raises(SlidesError, match="не сконвертировал"):
            prepare_slides([deck], tmp_path / "images", tmp_path / "work", 150)

    def test_soffice_without_pdf_raises(self, tmp_path: Path, tools: None,
                                        monkeypatch: pytest.MonkeyPatch) -> None:
        import absorbent.slides as slides_module

        monkeypatch.setattr(slides_module.subprocess, "run", FakeRunner(leave_pdf=False))
        deck = tmp_path / "deck.odp"
        deck.write_bytes(b"zip")
        with pytest.raises(SlidesError, match="не оставил PDF"):
            prepare_slides([deck], tmp_path / "images", tmp_path / "work", 150)

    def test_pdftoppm_failure_raises(self, tmp_path: Path, tools: None,
                                     monkeypatch: pytest.MonkeyPatch) -> None:
        import absorbent.slides as slides_module

        monkeypatch.setattr(slides_module.subprocess, "run", FakeRunner(pdftoppm_code=1))
        deck = tmp_path / "deck.pdf"
        deck.write_bytes(b"%PDF")
        with pytest.raises(SlidesError, match="не развернул"):
            prepare_slides([deck], tmp_path / "images", tmp_path / "work", 150)

    def test_no_pages_produced_raises(self, tmp_path: Path, tools: None,
                                      monkeypatch: pytest.MonkeyPatch) -> None:
        import absorbent.slides as slides_module

        monkeypatch.setattr(slides_module.subprocess, "run", FakeRunner(write_pages=False))
        deck = tmp_path / "deck.pdf"
        deck.write_bytes(b"%PDF")
        with pytest.raises(SlidesError, match="не нашлось страниц"):
            prepare_slides([deck], tmp_path / "images", tmp_path / "work", 150)

    def test_missing_pdftoppm_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import absorbent.slides as slides_module

        monkeypatch.setattr(slides_module.shutil, "which", lambda name: None)
        deck = tmp_path / "deck.pdf"
        deck.write_bytes(b"%PDF")
        with pytest.raises(FileNotFoundError, match="pdftoppm"):
            prepare_slides([deck], tmp_path / "images", tmp_path / "work", 150)

    def test_missing_soffice_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import absorbent.slides as slides_module

        monkeypatch.setattr(slides_module.shutil, "which", lambda name: None)
        deck = tmp_path / "deck.pptx"
        deck.write_bytes(b"zip")
        with pytest.raises(FileNotFoundError, match="soffice"):
            prepare_slides([deck], tmp_path / "images", tmp_path / "work", 150)


@pytest.mark.skipif(shutil.which("pdftoppm") is None, reason="pdftoppm не установлен")
class TestRealPdftoppm:
    def test_pages_of_pillow_pdf(self, tmp_path: Path) -> None:
        """Настоящий pdftoppm на многостраничном PDF из Pillow."""
        pages = [Image.new("RGB", (640, 360), (255, 255, 255)) for _ in range(2)]
        deck = tmp_path / "deck.pdf"
        pages[0].save(deck, "PDF", save_all=True, append_images=pages[1:], resolution=150)
        entries = prepare_slides([deck], tmp_path / "images", tmp_path / "work", 150)
        assert [entry["image"] for entry in entries] == ["slide_001.png", "slide_002.png"]
        assert (tmp_path / "work" / "01_deck" / SIGNATURE_NAME).exists()
        with Image.open(tmp_path / "images" / "slide_002.png") as image:
            assert image.size == (640, 360)
        # Повторный прогон берёт страницы из кэша: страницы те же.
        assert prepare_slides([deck], tmp_path / "images", tmp_path / "work", 150) == entries
