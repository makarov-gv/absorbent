"""Тесты параллельного запуска задач: окна, порядок, исключения, контекст."""

from __future__ import annotations

import contextvars
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from absorbent.events import Journal, current_journal, stage
from absorbent.parallel import run_batches, start_in_thread, submit


class TestRunBatches:
    def test_empty_items_returns_empty_list(self) -> None:
        seen: list = []
        assert run_batches([], lambda item: item, 2, lambda item, value: seen.append(value)) == []
        assert seen == []

    @pytest.mark.parametrize("batch_size", [0, -1])
    def test_non_positive_batch_size_rejected(self, batch_size: int) -> None:
        with pytest.raises(ValueError, match="размер батча должен быть положительным"):
            run_batches([1], lambda item: item, batch_size, lambda item, value: None)

    def test_window_runs_in_parallel(self) -> None:
        # Барьер из трёх потоков срабатывает только при настоящем параллелизме:
        # при последовательном исполнении он падает по таймауту.
        barrier = threading.Barrier(3, timeout=5.0)

        def worker(item: int) -> int:
            barrier.wait()
            return item * 2

        assert run_batches([1, 2, 3], worker, 3, lambda item, value: None) == [2, 4, 6]

    def test_results_follow_input_order_despite_timing(self) -> None:
        items = [
            {"delay": 0.20, "value": "a"},
            {"delay": 0.00, "value": "b"},
            {"delay": 0.10, "value": "c"},
        ]
        seen: list[tuple[str, str]] = []

        def worker(item: dict) -> str:
            time.sleep(item["delay"])
            return item["value"]

        results = run_batches(items, worker, 3, lambda item, value: seen.append((item["value"], value)))
        assert results == ["a", "b", "c"]
        assert seen == [("a", "a"), ("b", "b"), ("c", "c")]

    def test_on_result_called_for_every_item(self) -> None:
        seen: list[tuple[int, int]] = []
        run_batches([1, 2, 3, 4, 5], lambda item: item + 1, 2,
                    lambda item, value: seen.append((item, value)))
        assert seen == [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]

    def test_worker_exception_propagates(self) -> None:
        def worker(item: int) -> int:
            if item == 2:
                raise RuntimeError("boom")
            return item

        with pytest.raises(RuntimeError, match="boom"):
            run_batches([1, 2, 3], worker, 2, lambda item, value: None)

    def test_caller_context_variable_visible_in_worker(self) -> None:
        var = contextvars.ContextVar("probe_batches", default="unset")
        var.set("caller")
        assert run_batches([1], lambda item: var.get(), 1, lambda item, value: None) == ["caller"]


class TestSubmit:
    def test_returns_value(self) -> None:
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            assert submit(pool, lambda a, b: a + b, 2, 3).result(timeout=5) == 5
        finally:
            pool.shutdown()

    def test_propagates_exception(self) -> None:
        def boom() -> None:
            raise RuntimeError("bad")

        pool = ThreadPoolExecutor(max_workers=1)
        try:
            future = submit(pool, boom)
            with pytest.raises(RuntimeError, match="bad"):
                future.result(timeout=5)
        finally:
            pool.shutdown()

    def test_context_variable_visible_in_task(self) -> None:
        var = contextvars.ContextVar("probe_submit", default="unset")
        var.set("caller")
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            assert submit(pool, var.get).result(timeout=5) == "caller"
        finally:
            pool.shutdown()

    def test_journal_context_visible_in_task(self, tmp_path) -> None:
        # Журнал живёт в contextvars: задача в потоке должна видеть тот же sink.
        with Journal(tmp_path / "events.jsonl"):
            journal = current_journal()
            pool = ThreadPoolExecutor(max_workers=1)
            try:
                assert submit(pool, current_journal).result(timeout=5) is journal
            finally:
                pool.shutdown()

    def test_stage_context_visible_in_task(self) -> None:
        with stage("probe"):
            pool = ThreadPoolExecutor(max_workers=1)
            try:
                # Прогресс активного этапа доступен из задачи: проверяем через
                # функцию, возвращающую имя этапа из contextvars.
                assert submit(pool, _active_stage_name).result(timeout=5) == "probe"
            finally:
                pool.shutdown()


class TestStartInThread:
    def test_returns_value(self) -> None:
        assert start_in_thread(lambda x: x + 1, 41).result(timeout=5) == 42

    def test_propagates_exception(self) -> None:
        def boom() -> None:
            raise RuntimeError("bad")

        with pytest.raises(RuntimeError, match="bad"):
            start_in_thread(boom).result(timeout=5)

    def test_runs_as_daemon(self) -> None:
        # Демон не должен держать завершение процесса: проверяем через флаг потока.
        assert start_in_thread(_is_daemon_thread).result(timeout=5) is True

    def test_context_variable_visible_in_thread(self) -> None:
        var = contextvars.ContextVar("probe_thread", default="unset")
        var.set("caller")
        assert start_in_thread(var.get).result(timeout=5) == "caller"


def _active_stage_name() -> str:
    """Имя активного этапа из contextvars; пустая строка, если этапа нет."""
    from absorbent.events import _active_stage

    active = _active_stage.get()
    return active.name if active is not None else ""


def _is_daemon_thread() -> bool:
    """Запущена ли текущая функция в потоке-демоне."""
    return threading.current_thread().daemon
