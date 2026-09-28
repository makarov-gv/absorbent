"""Синтез итогового Markdown-конспекта текстовой LLM."""

from __future__ import annotations

from pathlib import Path

from absorbent.config import PLACEHOLDER_API_KEY, ModelSpec, Provider, api_key_from
from absorbent.slides import KIND_PHOTO, KIND_SLIDE

SYSTEM_PROMPTS: dict[str, str] = {
    "ru": (
        "Ты составляешь качественный структурированный конспект лекции в Markdown. На вход: "
        "полный транскрипт с таймкодами — это речь лектора и единственный источник того, что "
        "на лекции действительно было; список ключевых кадров (имя файла, таймкод, описание); "
        "дополнительные материалы — страницы презентации лектора (метка «слайд N») и "
        "фотографии доски, снятые на лекции. "
        "Пиши весь конспект на русском языке, но дословные фразы и термины из транскрипта, "
        "произнесённые на другом языке (например, английском), сохраняй как есть, не переводя. "
        "Составь полный Markdown: заголовок, разделы, таблицы где уместно, формулы в LaTeX "
        "($...$), хронологический порядок — по ходу речи лектора. "
        "Конспект описывает ровно то, о чём лектор успел рассказать. Презентация обычно "
        "длиннее лекции: лектор мог остановиться в её середине, поэтому слайд, содержание "
        "которого не прозвучало в транскрипте, в конспект не попадает — ни отдельным "
        "разделом, ни пересказом, ни иллюстрацией; перечислять неразобранные слайды тоже не "
        "нужно. Не додумывай за лектора то, чего в транскрипте нет. Слайды и фотографии "
        "доски — справочный материал: там, где они уточняют или поправляют сказанное, их "
        "формулировки, формулы и подписи можно перенести в конспект, но своей темой они "
        "конспект не расширяют. "
        "Вставляй изображения только ссылкой вида ![подпись]({images_prefix}<имя_файла>) "
        "ровно с теми именами файлов, что даны в списках кадров, слайдов и фотографий. "
        "Файлы изображений уже лежат в {images_prefix}, а не в каком-либо другом каталоге. "
        "Слайд вставляй как ![Слайд N]({images_prefix}<имя_файла>), фотографию доски — как "
        "![Фотография доски]({images_prefix}<имя_файла>). Не выдумывай имена файлов."
    ),
    "en": (
        "You write a high-quality structured Markdown summary of a lecture. Input: the full "
        "transcript with timestamps — that is the lecturer's speech and the only source of what "
        "actually happened in the lecture; a list of keyframes (filename, timestamp, "
        "description); and supplementary material — the lecturer's presentation pages "
        "(labelled 'slide N') and photos of the board taken during the lecture. "
        "Write the whole summary in English, but keep verbatim phrases and terms from the "
        "transcript spoken in another language (e.g. Russian) as they are, without translating. "
        "Produce complete Markdown: title, sections, tables where useful, LaTeX formulas "
        "($...$), chronological order — following the lecturer's speech. "
        "The summary covers exactly what the lecturer had time to say. A deck is usually longer "
        "than the lecture: the lecturer may have stopped in the middle of it, so a slide whose "
        "content was never spoken about does not appear in the summary at all — neither as a "
        "section, nor as a retelling, nor as an illustration; there is no need to list the "
        "untouched slides either. Do not invent what the lecturer did not say. Slides and board "
        "photos are reference material: where they clarify or correct the speech, their wording, "
        "formulas and captions may go into the summary, but they do not extend it to topics of "
        "their own. "
        "Embed images only as ![caption]({images_prefix}<filename>) using exactly the filenames "
        "given in the keyframe, slide and photo lists. Image files already exist under "
        "{images_prefix}, not in any other directory. Embed a slide as "
        "![Slide N]({images_prefix}<filename>) and a board photo as "
        "![Board photo]({images_prefix}<filename>). Do not invent filenames."
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
    """Список страниц презентации для модели: путь, номер страницы, описание."""
    lines = []
    # Стабильный порядок по имени файла = порядок страниц (имена slide_NNN.png).
    for name in sorted(slides):
        entry = slides[name]
        if entry.get("kind", KIND_SLIDE) == KIND_PHOTO:
            continue
        lines.append(f"- {images_prefix}{name} [слайд {entry['page']}]: {entry['description']}")
    return "\n".join(lines)


def _render_photos(slides: dict, images_prefix: str) -> str:
    """Список фотографий доски для модели: путь и описание (номер страницы не нужен)."""
    lines = []
    for name in sorted(slides):
        entry = slides[name]
        if entry.get("kind", KIND_SLIDE) != KIND_PHOTO:
            continue
        lines.append(f"- {images_prefix}{name} [фотография доски]: {entry['description']}")
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

    slides — описания дополнительных материалов к лекции ({"slide_003.png":
    {"page", "source", "kind", "description"}}): страницы презентации (kind
    "slide", метка «слайд N») и фотографии доски (kind "photo"). Они идут в тот
    же вход, что и кадры, отдельными разделами, потому что таймкодов у
    страниц и снимков нет и сопоставление с речью делает сама модель. None и
    пустой словарь означают «дополнительных материалов нет»; отсутствие поля
    kind читается как страница презентации.

    Пустой (или состоящий из пробелов) ответ модели — ошибка этапа: запрос
    повторяется один раз, а при повторной пустоте поднимается SynthesisError
    с диагностикой (finish_reason, расход токенов, длина reasoning), чтобы
    конвейер не записал пустой notes.md и не собрал по нему HTML и PDF.
    """
    from openai import OpenAI

    system_prompt = SYSTEM_PROMPTS[_prompt_language(output_language)].format(
        images_prefix=images_prefix)

    user_message = (
        "Транскрипт лекции (основной источник; конспект строится по нему):\n"
        f"{_render_transcript(transcript)}\n\n"
        "Ключевые кадры записи:\n"
        f"{_render_keyframes(descriptions, images_prefix)}"
    )
    materials = slides or {}
    pages = _render_slides(materials, images_prefix)
    if pages:
        user_message += ("\n\nСлайды презентации (справочно: часть колоды могла остаться "
                         "за рамками лекции):\n" + pages)
    photos = _render_photos(materials, images_prefix)
    if photos:
        user_message += "\n\nФотографии доски (снимки, сделанные на лекции):\n" + photos

    api_key = api_key_from(provider)
    client = OpenAI(base_url=provider.base_url, api_key=api_key or PLACEHOLDER_API_KEY)
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
            "повторите прогон: синтез пойдёт заново, остальные этапы возьмутся из кэша"
        )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(text, encoding="utf-8")
    return out_md
