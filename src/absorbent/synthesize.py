"""Синтез итогового Markdown-конспекта текстовой LLM."""

from __future__ import annotations

from pathlib import Path

from absorbent.config import ModelSpec, Provider, api_key_from

SYSTEM_PROMPTS: dict[str, str] = {
    "ru": (
        "Ты составляешь качественный структурированный конспект лекции в Markdown. На вход: "
        "полный транскрипт с таймкодами и список ключевых кадров (имя файла, таймкод, описание). "
        "Пиши весь конспект на русском языке, но дословные фразы и термины из транскрипта, "
        "произнесённые на другом языке (например, английском), сохраняй как есть, не переводя. "
        "Составь полный Markdown: заголовок, разделы, таблицы где уместно, формулы в LaTeX "
        "($...$), хронологический порядок. Вставляй изображения только ссылкой вида "
        "![подпись]({images_prefix}<имя_файла>) ровно с теми именами файлов, что даны в списке "
        "кадров, где кадр иллюстрирует мысль. Файлы изображений уже лежат в {images_prefix}, "
        "а не в каком-либо другом каталоге. В том же входе могут быть слайды презентации — "
        "они помечены «слайд N»: это страницы презентации лектора в порядке изложения, без "
        "таймкодов. Опирайся на их содержание там, где тема совпадает с речью, и вставляй их "
        "ссылкой ![Слайд N]({images_prefix}<имя_файла>). Не выдумывай имена файлов."
    ),
    "en": (
        "You write a high-quality structured Markdown summary of a lecture. Input: the full "
        "transcript with timestamps, and a list of keyframes (filename, timestamp, description). "
        "Write the whole summary in English, but keep verbatim phrases and terms from the "
        "transcript spoken in another language (e.g. Russian) as they are, without translating. "
        "Produce complete Markdown: title, sections, tables where useful, LaTeX formulas "
        "($...$), chronological order. Embed images only as ![caption]({images_prefix}<filename>) "
        "using exactly the filenames given in the keyframe list, where a frame illustrates the "
        "point. Image files already exist under {images_prefix}, not in any other directory. "
        "The same input may contain presentation slides labelled 'slide N': these are the "
        "lecturer's pages in presentation order, with no timestamps. Use their content where "
        "the topic matches the speech, and embed them as "
        "![Slide N]({images_prefix}<filename>). Do not invent filenames."
    ),
}


# Число попыток синтеза: провайдер изредка отдаёт пустой ответ, повтор лечит разовый сбой.
SYNTHESIS_ATTEMPTS = 2


class SynthesisError(Exception):
    """Ошибка этапа синтеза: модель не вернула текст конспекта."""


def _prompt_language(output_language: str) -> str:
    """Язык промпта синтеза: всегда явный (ru/en), с запасом на 'ru'."""
    return output_language if output_language in SYSTEM_PROMPTS else "ru"


def _render_transcript(transcript: list[dict]) -> str:
    """Весь транскрипт строками [MM:SS] text (или маркер недоступности)."""
    if not transcript:
        return "(transcript unavailable)"
    lines = []
    for seg in transcript:
        mm = int(float(seg["start"]) // 60)
        ss = int(float(seg["start"]) % 60)
        text = seg["text"]
        speaker = seg.get("speaker")
        prefix = f"[{speaker}] " if speaker else ""
        lines.append(f"[{mm:02d}:{ss:02d}] {prefix}{text}")
    return "\n".join(lines)


def _render_keyframes(descriptions: dict, images_prefix: str) -> str:
    """Список ключевых кадров для модели: путь, таймкод, описание."""
    lines = []
    # Стабильный порядок по имени файла = хронологический.
    for name in sorted(descriptions):
        entry = descriptions[name]
        lines.append(f"- {images_prefix}{name} [{entry['timestamp']}]: {entry['description']}")
    return "\n".join(lines)


def _render_slides(slides: dict, images_prefix: str) -> str:
    """Список слайдов для модели: путь, номер страницы, описание."""
    lines = []
    # Стабильный порядок по имени файла = порядок страниц (имена slide_NNN.png).
    for name in sorted(slides):
        entry = slides[name]
        lines.append(f"- {images_prefix}{name} [слайд {entry['page']}]: {entry['description']}")
    return "\n".join(lines)


def _answer_text(response: object) -> str:
    """Текст ответа модели; пустая строка, если провайдер вернул пустой content."""
    return (response.choices[0].message.content or "").strip()


def _empty_answer_details(response: object) -> str:
    """Диагностика пустого ответа: причина остановки, расход токенов и reasoning.

    Ключ reasoning_content есть только у reasoning-моделей и приходит
    необъявленным полем ответа, поэтому читается из model_extra.
    """
    choice = response.choices[0]
    usage = getattr(response, "usage", None)
    reasoning = (getattr(choice.message, "model_extra", None) or {}).get("reasoning_content") or ""
    reasoning_tokens = getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", None)
    return (f"finish_reason={choice.finish_reason}, "
            f"completion_tokens={getattr(usage, 'completion_tokens', '?')}, "
            f"reasoning_tokens={reasoning_tokens}, reasoning_content={len(reasoning)} симв.")


def run_synthesize(llm_spec: ModelSpec, provider: Provider, transcript: list[dict],
                   descriptions: dict, output_language: str, out_md: Path,
                   images_prefix: str = "images/", slides: dict | None = None) -> Path:
    """Синтезировать notes.md из транскрипта, описаний кадров и слайдов; вернуть путь.

    Язык конспекта задаётся явно (output_language: ru/en). Дословные RU/EN-
    фразы из транскрипта модель сохраняет в исходном языке; язык выбирает
    только наш промпт, а не каприз модели. images_prefix — каталог картинок
    относительно итогового Markdown: в обычной раскладке 'images/', в плоском
    режиме '<имя>.images/'.

    slides — описания страниц презентации лектора ({"slide_003.png": {"page",
    "source", "description"}}); они идут в тот же вход, что и кадры, отдельным
    разделом «Слайды презентации» с меткой «слайд N», потому что таймкодов у
    страниц нет и сопоставление с речью делает сама модель. None и пустой
    словарь означают «слайдов нет».

    Пустой (или состоящий из пробелов) ответ модели — ошибка этапа: запрос
    повторяется один раз, а при повторной пустоте поднимается SynthesisError
    с диагностикой (finish_reason, расход токенов, длина reasoning), чтобы
    конвейер не записал пустой notes.md и не собрал по нему HTML и PDF.
    """
    from openai import OpenAI

    system_prompt = SYSTEM_PROMPTS[_prompt_language(output_language)].format(
        images_prefix=images_prefix)

    user_message = (
        "Транскрипт лекции:\n"
        f"{_render_transcript(transcript)}\n\n"
        "Ключевые кадры:\n"
        f"{_render_keyframes(descriptions, images_prefix)}"
    )
    if slides:
        user_message += "\n\nСлайды презентации:\n" + _render_slides(slides, images_prefix)

    api_key = api_key_from(provider)
    client = OpenAI(base_url=provider.base_url, api_key=api_key)
    text = ""
    for attempt in range(SYNTHESIS_ATTEMPTS):
        response = client.chat.completions.create(
            model=llm_spec.api_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        text = _answer_text(response)
        if text:
            break
        if attempt + 1 < SYNTHESIS_ATTEMPTS:
            print("внимание: модель вернула пустой ответ, повтор запроса")
    if not text:
        raise SynthesisError(
            f"модель не вернула текст конспекта ({_empty_answer_details(response)}); "
            "повторите прогон по кэшу: --skip-stt --skip-frames --skip-describe"
        )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(text, encoding="utf-8")
    return out_md
