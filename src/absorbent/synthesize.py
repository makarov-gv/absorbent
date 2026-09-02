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
        "![подпись](images/<имя_файла>) ровно с теми именами файлов, что даны в списке кадров, "
        "где кадр иллюстрирует мысль. Файлы изображений уже лежат в images/. Не выдумывай имена "
        "файлов."
    ),
    "en": (
        "You write a high-quality structured Markdown summary of a lecture. Input: the full "
        "transcript with timestamps, and a list of keyframes (filename, timestamp, description). "
        "Write the whole summary in English, but keep verbatim phrases and terms from the "
        "transcript spoken in another language (e.g. Russian) as they are, without translating. "
        "Produce complete Markdown: title, sections, tables where useful, LaTeX formulas "
        "($...$), chronological order. Embed images only as ![caption](images/<filename>) using "
        "exactly the filenames given in the keyframe list, where a frame illustrates the point. "
        "Image files already exist under images/. Do not invent filenames."
    ),
}


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


def _render_keyframes(descriptions: dict) -> str:
    """Список ключевых кадров для модели: путь, таймкод, описание."""
    lines = []
    # Стабильный порядок по имени файла = хронологический.
    for name in sorted(descriptions):
        entry = descriptions[name]
        lines.append(f"- images/{name} [{entry['timestamp']}]: {entry['description']}")
    return "\n".join(lines)


def run_synthesize(llm_spec: ModelSpec, provider: Provider, transcript: list[dict],
                   descriptions: dict, output_language: str, out_md: Path) -> Path:
    """Синтезировать notes.md из транскрипта и описаний кадров; вернуть путь файла.

    Язык конспекта задаётся явно (output_language: ru/en). Дословные RU/EN-
    фразы из транскрипта модель сохраняет в исходном языке; язык выбирает
    только наш промпт, а не каприз модели.
    """
    from openai import OpenAI

    system_prompt = SYSTEM_PROMPTS[_prompt_language(output_language)]

    user_message = (
        "Транскрипт лекции:\n"
        f"{_render_transcript(transcript)}\n\n"
        "Ключевые кадры:\n"
        f"{_render_keyframes(descriptions)}"
    )

    api_key = api_key_from(provider)
    client = OpenAI(base_url=provider.base_url, api_key=api_key)
    response = client.chat.completions.create(
        model=llm_spec.api_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    text = (response.choices[0].message.content or "").strip()
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(text, encoding="utf-8")
    return out_md
