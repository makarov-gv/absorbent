"""Границы JSONL-журнала: жизненный цикл, контексты, пропуски и потеря записи."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from uuid import UUID

import pytest

from absorbent.events import (
    Journal,
    JournalError,
    api_call,
    current_journal,
    emit,
    report_progress,
    request_fields,
    skip_stage,
    stage,
)


def records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class TestStageProgress:
    """События прогресса этапа: поля, троттлинг, границы контекста."""

    def test_reports_fields_and_throttles_between_forced_events(self, tmp_path: Path) -> None:
        path = tmp_path / "events.jsonl"
        with Journal(path):
            with stage("stt") as status:
                report_progress("seconds", 0.0, 30.0, force=True)
                report_progress("seconds", 12.0, 30.0)  # подавлен троттлингом.
                report_progress("seconds", 30.0, 30.0, force=True)
        rows = [row for row in records(path) if row["event"] == "stage_progress"]
        assert [(row["completed"], row["total"]) for row in rows] == [(0.0, 30.0), (30.0, 30.0)]
        assert all(row["stage"] == "stt" and row["unit"] == "seconds" for row in rows)
        assert all(row["schema_version"] == 1 for row in rows)

    def test_interval_allows_intermediate_events(self, tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("absorbent.events._PROGRESS_INTERVAL_SECONDS", 0.0)
        path = tmp_path / "events.jsonl"
        with Journal(path):
            with stage("describe_keyframes"):
                for done in range(1, 4):
                    report_progress("images", done, 3)
        completed = [row["completed"] for row in records(path)
                     if row["event"] == "stage_progress"]
        assert completed == [1, 2, 3]

    def test_no_journal_writes_nothing(self, capsys: pytest.CaptureFixture) -> None:
        with stage("stt"):
            report_progress("seconds", 1.0, 2.0, force=True)
        assert capsys.readouterr() == ("", "")

    def test_outside_stage_is_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "events.jsonl"
        with Journal(path):
            report_progress("images", 1, 2, force=True)
        assert [row for row in records(path) if row["event"] == "stage_progress"] == []

    def test_progress_stage_resets_between_stages(self, tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
        # Троттлинг не переносится на следующий этап: первый нефорсированный
        # вызов нового этапа не должен подавляться временем предыдущего.
        path = tmp_path / "events.jsonl"
        with Journal(path):
            with stage("describe_keyframes"):
                report_progress("images", 1, 1, force=True)
            with stage("describe_slides"):
                report_progress("images", 1, 1)
        stages = [row["stage"] for row in records(path) if row["event"] == "stage_progress"]
        assert stages == ["describe_keyframes", "describe_slides"]


class FailingFile:
    """Настоящий файл с управляемым отказом записи, flush или закрытия."""

    def __init__(self, handle) -> None:
        self.handle = handle
        self.event = None
        self.fail_flush = False
        self.fail_close = False

    def write(self, text):
        if json.loads(text)["event"] == self.event:
            raise OSError("disk full")
        return self.handle.write(text)

    def flush(self):
        if self.fail_flush:
            raise OSError("flush failed")
        self.handle.flush()

    def close(self):
        self.handle.close()
        if self.fail_close:
            raise OSError("close failed")


@pytest.fixture
def failing_journal_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "events.jsonl"
    wrapper = FailingFile(path.open("a", encoding="utf-8"))
    original = Path.open

    def open_file(self, *args, **kwargs):
        if self == path and args and args[0] == "a":
            return wrapper
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_file)
    return path, wrapper


def test_append_flush_lifecycle_and_stage_statuses(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "events.jsonl"
    with Journal(path):
        with stage("stt"):
            emit("measurement", text="русский")
            assert records(path)[-1]["text"] == "русский"
        with stage("render") as status:
            status.skip()
        with stage("describe_slides"):
            skip_stage()
    with Journal(path) as journal:
        journal.exit_code = 1
    rows = records(path)
    runs = [row for row in rows if row["event"] == "run_start"]
    assert len(runs) == 2
    assert runs[0]["run_id"] != runs[1]["run_id"]
    for run in runs:
        UUID(run["run_id"])
        members = [row for row in rows if row["run_id"] == run["run_id"]]
        assert [row["seq"] for row in members] == list(range(1, len(members) + 1))
    for row in rows:
        assert row["schema_version"] == 1
        assert datetime.fromisoformat(row["timestamp_utc"]).utcoffset().total_seconds() == 0
    assert [(row["stage"], row["status"]) for row in rows if row["event"] == "stage_end"] == [
        ("stt", "ok"), ("render", "skipped"), ("describe_slides", "skipped")]
    assert all(row["elapsed_seconds"] >= 0 for row in rows if row["event"] == "stage_end")
    ends = [row for row in rows if row["event"] == "run_end"]
    assert [(row["status"], row["exit_code"]) for row in ends] == [("ok", 0), ("error", 1)]
    assert {key: runs[0][key] for key in ("version", "commit", "dirty")} == {
        key: ends[0][key] for key in ("version", "commit", "dirty")}


def test_exception_closes_stage_and_run_and_restores_context(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    with pytest.raises(RuntimeError, match="stage failed"):
        with Journal(path):
            with stage("synthesize"):
                raise RuntimeError("stage failed")
    rows = records(path)
    assert rows[-2]["event"] == "stage_end" and rows[-2]["status"] == "error"
    assert rows[-1]["event"] == "run_end" and rows[-1]["exit_code"] == 1
    assert current_journal() is None
    assert request_fields() == {"stage": None, "provider": None, "model": None}


def test_no_journal_has_no_output_and_nested_contexts_restore(tmp_path: Path, capsys) -> None:
    emit("disabled")
    skip_stage()
    with stage("outer"):
        assert request_fields()["stage"] == "outer"
        with api_call("stt", "local", "voxtral"):
            assert request_fields() == {"stage": "stt", "provider": "local", "model": "voxtral"}
            with api_call("synthesize", "cloud", "model"):
                assert request_fields()["provider"] == "cloud"
            assert request_fields()["provider"] == "local"
        assert request_fields()["provider"] is None
    with Journal(tmp_path / "outer.jsonl") as outer:
        with Journal(tmp_path / "inner.jsonl"):
            emit("inner")
        assert current_journal() is outer
        emit("outer")
    assert records(tmp_path / "outer.jsonl")[-2]["event"] == "outer"
    assert records(tmp_path / "inner.jsonl")[-2]["event"] == "inner"
    assert capsys.readouterr() == ("", "")


def test_open_failure_is_distinct_journal_error(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    with pytest.raises(JournalError, match="открыть"):
        with Journal(blocker / "events.jsonl"):
            pass
    assert current_journal() is None


@pytest.mark.parametrize("failure", ["run_start", "stage_start", "measurement", "flush", "close"])
def test_io_failures_are_fatal_and_restore_sink(failing_journal_file, failure: str) -> None:
    path, wrapper = failing_journal_file
    if failure == "run_start":
        wrapper.event = "run_start"
    with pytest.raises(JournalError):
        with Journal(path):
            if failure == "stage_start":
                wrapper.event = "stage_start"
            elif failure == "measurement":
                wrapper.event = "measurement"
            elif failure == "flush":
                wrapper.fail_flush = True
            else:
                wrapper.fail_close = True
            with stage("stt"):
                emit("measurement")
    assert wrapper.handle.closed
    assert current_journal() is None


@pytest.mark.parametrize("raw", ["0", "-1", "1.5", "nan", "invalid"])
def test_cli_invalid_material_limits_are_rejected(raw: str) -> None:
    from absorbent.cli import build_parser

    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["lecture.wav", "--max-material-pages", raw])
    assert exc.value.code == 2


def test_cli_journal_open_failure_exits_two_before_processing(tmp_path: Path, minimal_config, monkeypatch) -> None:
    from absorbent.cli import main

    source = tmp_path / "lecture.wav"
    source.write_bytes(b"wav")
    blocker = tmp_path / "blocker"
    blocker.write_bytes(b"x")
    calls = []
    monkeypatch.setattr("absorbent.pipeline.process_file", lambda *args: calls.append(args))
    assert main([str(source), "--config", str(minimal_config), "--journal", str(blocker / "events")]) == 2
    assert calls == []


@pytest.mark.parametrize("failure", ["run_start", "measurement", "run_end"])
def test_cli_journal_write_failure_exits_one_and_stops(failing_journal_file, minimal_config, monkeypatch, failure) -> None:
    from absorbent.cli import main

    path, wrapper = failing_journal_file
    source = path.parent / "lecture.wav"
    source.write_bytes(b"wav")
    calls = []

    def process(input_path, cfg, args):
        calls.append(input_path)
        emit("measurement")

    wrapper.event = failure
    monkeypatch.setattr("absorbent.pipeline.process_file", process)
    assert main([str(source), "--config", str(minimal_config), "--journal", str(path)]) == 1
    assert len(calls) == (0 if failure == "run_start" else 1)
    assert current_journal() is None


def test_cli_directory_processing_failure_closes_run(tmp_path: Path, minimal_config, monkeypatch) -> None:
    from absorbent.cli import main

    for name in ("a.wav", "b.wav"):
        (tmp_path / name).write_bytes(b"wav")
    calls = []

    def process(input_path, cfg, args):
        calls.append(input_path.name)
        assert args.max_material_pages == 100
        if input_path.name == "a.wav":
            raise RuntimeError("failed")

    path = tmp_path / "journal" / "events.jsonl"
    monkeypatch.setattr("absorbent.pipeline.process_file", process)
    assert main([str(tmp_path), "--config", str(minimal_config), "--journal", str(path),
                 "--max-material-pages", "100"]) == 1
    assert calls == ["a.wav", "b.wav"]
    assert records(path)[-1]["exit_code"] == 1


@pytest.mark.parametrize("wizard_limit", [None, 200])
def test_cli_wizard_preserves_explicit_journal_and_material_limit(tmp_path: Path, minimal_config, monkeypatch, wizard_limit) -> None:
    import sys

    from absorbent.cli import main

    source = tmp_path / "lecture.wav"
    source.write_bytes(b"wav")
    path = tmp_path / "events.jsonl"
    seen = []
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    wizard_args = [str(source)]
    if wizard_limit is not None:
        wizard_args.extend(["--max-material-pages", str(wizard_limit)])
    monkeypatch.setattr("absorbent.wizard.run_wizard", lambda cfg, config: wizard_args)
    monkeypatch.setattr("absorbent.pipeline.process_file", lambda input_path, cfg, args: seen.append(args))
    assert main(["--config", str(minimal_config), "--journal", str(path), "--max-material-pages", "100"]) == 0
    assert seen[0].max_material_pages == (100 if wizard_limit is None else wizard_limit)
    assert records(path)[-1]["exit_code"] == 0


def test_material_stage_skips_only_entirely_cached_sources(tmp_path: Path) -> None:
    from absorbent.slides import prepare_slides

    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    first.write_bytes(b"photo")
    second.write_bytes(b"another photo")
    path = tmp_path / "events.jsonl"
    with Journal(path):
        with stage("prepare_slides"):
            prepare_slides([first], tmp_path / "images", tmp_path / "cache", 150)
        with stage("prepare_slides"):
            prepare_slides([first], tmp_path / "images", tmp_path / "cache", 150)
        with stage("prepare_slides"):
            prepare_slides([first, second], tmp_path / "images", tmp_path / "cache", 150)
    assert [row["status"] for row in records(path) if row["event"] == "stage_end"] == ["ok", "skipped", "ok"]
