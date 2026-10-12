"""Параллельный запуск независимых задач с сохранением контекста журнала.

Задачи этапов (вызовы модели по кадрам, страницам, фрагментам) не зависят
друг от друга, поэтому выполняются окнами в пуле потоков. Журнал, активный
этап и привязка HTTP-вызовов живут в contextvars: в новый поток они сами не
наследуются, поэтому каждая задача получает собственную копию контекста.
"""

from __future__ import annotations

import contextvars
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def submit(pool: ThreadPoolExecutor, function: Callable[..., R],
           *args: object) -> Future[R]:
    """Отправить задачу в пул с копией текущего контекста.

    Копия нужна на каждую задачу: в новом потоке contextvars не наследуются,
    а от них зависят журнал (events._sink), текущий этап
    (events._active_stage) и привязка HTTP-вызовов (events._call); один и
    тот же объект контекста нельзя входить из двух потоков одновременно.
    """
    return pool.submit(contextvars.copy_context().run, function, *args)


def run_batches(items: Sequence[T], worker: Callable[[T], R], batch_size: int,
                on_result: Callable[[T, R], None]) -> list[R]:
    """Выполнить items окнами не длиннее batch_size и собрать результаты по порядку.

    Окно отправляется целиком, результаты ожидаются по очереди в порядке
    отправки: on_result(item, value) вызывается сразу после того, как
    очередной результат получен, и потому видит входной порядок независимо от
    времени ответа. Исключение из worker прекращает работу и всплывает
    вызывающему: новые окна не запускаются, а ещё не начатые задачи окна
    отменяются. Возвращает список значений в порядке items.
    """
    if batch_size <= 0:
        raise ValueError("размер батча должен быть положительным")
    results: list[R] = []
    pool = ThreadPoolExecutor(max_workers=batch_size)
    try:
        for start in range(0, len(items), batch_size):
            window = list(items[start:start + batch_size])
            futures = [submit(pool, worker, item) for item in window]
            for item, future in zip(window, futures):
                value = future.result()
                results.append(value)
                on_result(item, value)
    except BaseException:
        pool.shutdown(cancel_futures=True)
        raise
    pool.shutdown()
    return results


def start_in_thread(function: Callable[..., R], *args: object) -> Future[R]:
    """Запустить функцию в потоке-демоне с копией контекста; вернуть Future.

    Демон нужен, чтобы прерванный прогон (Ctrl-C или ошибка соседнего этапа)
    не ждал окончания распознавания речи; и результат, и исключение задачи
    доезжают через Future.
    """
    future: Future[R] = Future()
    context = contextvars.copy_context()

    def run() -> None:
        try:
            future.set_result(context.run(function, *args))
        except BaseException as exc:  # исключение задачи доезжает через Future.
            future.set_exception(exc)

    threading.Thread(target=run, daemon=True).start()
    return future
