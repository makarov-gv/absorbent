"""Краткая выжимка транскрипта лекции отдельным вызовом текстовой LLM.

Выжимка — это общий контекст для моделей, которые описывают отдельные кадры
записи: кадр видит лишь своё окно транскрипта, а тема лекции, её ход и
терминология известны только по всему транскрипту. Поэтому выжимка считается
один раз и кэшируется отдельным файлом, а не растворяется в описаниях кадров.
"""

from __future__ import annotations

from pathlib import Path

from absorbent.api import create_client
from absorbent.config import ModelSpec, Provider
from absorbent.events import api_call, skip_stage
from absorbent.synthesize import empty_answer_details, render_transcript

SYSTEM_PROMPTS: dict[str, str] = {
    "ru": (
        "Ты составляешь краткую выжимку лекции по её транскрипту. На вход: полный транскрипт "
        "с таймкодами. Выжимка нужна как общий контекст для других моделей, которые описывают "
        "отдельные кадры записи, поэтому в неё входит: тема лекции; ход изложения по порядку; "
        "ключевые термины, определения, обозначения и формулы ($...$) в том виде, как их "
        "называет лектор; имена и названия, если они прозвучали. Пиши на русском языке, но "
        "дословные термины и фразы, произнесённые на другом языке, сохраняй как есть. Опирайся "
        "только на транскрипт и не додумывай того, чего в нём нет. Кратко: список, не длиннее "
        "600 слов, без вступления, без воды и без цитат из транскрипта. Верни только выжимку."
    ),
    "en": (
        "You compile a concise summary of a lecture from its transcript. Input: the full "
        "transcript with timestamps. The summary serves as shared context for other models that "
        "describe individual frames of the recording, so it contains: the topic of the lecture; "
        "the order of exposition; key terms, definitions, notations and formulas ($...$) as the "
        "lecturer names them; names and titles if they were spoken. Write in English, but keep "
        "verbatim terms and phrases spoken in another language as they are. Rely only on the "
        "transcript and do not invent what is not in it. Be concise: a list, no longer than 600 "
        "words, without an introduction, without filler and without quotes from the transcript. "
        "Return only the summary."
    ),
}


# Число попыток выжимки: провайдер изредка отдаёт пустой ответ, повтор лечит разовый сбой.
SUMMARY_ATTEMPTS = 2


class SummaryError(Exception):
    """Ошибка этапа выжимки: модель не вернула текст выжимки."""


def _prompt_language(output_language: str) -> str:
    """Язык промпта выжимки: всегда явный (ru/en), с запасом на 'ru'."""
    return output_language if output_language in SYSTEM_PROMPTS else "ru"


def _cached_summary(out_md: Path, transcript_path: Path | None) -> str | None:
    """Готовая выжимка из кэша или None, если её нужно получить заново.

    Файл кэша годен, если существует; при заданном transcript_path он вдобавок
    должен быть не старше самого транскрипта. Отсутствующий транскрипт свежесть
    не подтверждает: сравнить время не с чем, поэтому выжимка считается заново.
    """
    if not out_md.exists():
        return None
    if transcript_path is not None:
        if not transcript_path.exists():
            return None
        if out_md.stat().st_mtime < transcript_path.stat().st_mtime:
            return None
    return out_md.read_text(encoding="utf-8")


def run_summarize(llm_spec: ModelSpec, provider: Provider, transcript: list[dict],
                  output_language: str, out_md: Path,
                  transcript_path: Path | None = None, force: bool = False) -> str:
    """Сделать выжимку транскрипта текстовой LLM; вернуть её текст.

    Выжимка нужна как общий контекст для описания отдельных кадров записи,
    поэтому её артефакт — отдельный файл кэша (out_md), а не часть notes.md.
    Язык выжимки задаётся явно (output_language: ru/en): язык выбирает наш
    промпт, а дословные фразы, произнесённые на другом языке, модель сохраняет
    как есть.

    Кэш переиспользуется, если out_md существует, force не задан и — при
    заданном transcript_path — файл не старше самого транскрипта: правка
    транскрипта делает выжимку негодной, а неизменный транскрипт позволяет
    обойтись без повторного вызова модели. transcript_path=None означает, что
    свежесть не проверяется и кэш годен по одному лишь наличию файла. При
    промахе по кэшу модель вызывается заново, а результат пишется в out_md
    (родительский каталог создаётся).

    Пустой (или состоящий из пробелов) ответ модели — ошибка этапа: запрос
    повторяется один раз, а при повторной пустоте поднимается SummaryError с
    диагностикой (finish_reason, расход токенов, длина reasoning), чтобы
    выжимка не оказалась пустой и не испортила описания кадров.
    """
    if not force:
        cached = _cached_summary(out_md, transcript_path)
        if cached is not None:
            skip_stage()
            return cached

    system_prompt = SYSTEM_PROMPTS[_prompt_language(output_language)]
    user_message = "Транскрипт лекции:\n" + render_transcript(transcript)

    client = create_client(provider)
    text = ""
    for attempt in range(SUMMARY_ATTEMPTS):
        with api_call("summarize", provider.name, llm_spec.api_model):
            response = client.chat.completions.create(
                model=llm_spec.api_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
        text = (response.choices[0].message.content or "").strip()
        if text:
            break
        if attempt + 1 < SUMMARY_ATTEMPTS:
            print("внимание: модель вернула пустой ответ, повтор запроса")
    if not text:
        raise SummaryError(
            f"модель не вернула выжимку ({empty_answer_details(response)}); "
            "повторите прогон: выжимка пойдёт заново, остальные этапы возьмутся из кэша"
        )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(text, encoding="utf-8")
    return text
