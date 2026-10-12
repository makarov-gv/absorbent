"""Описание ключевых кадров vision-моделью с привязкой к транскрипту."""

from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path

from absorbent.api import create_client
from absorbent.config import ModelSpec, Provider
from absorbent.events import JournalError, api_call, report_progress, skip_stage
from absorbent.slides import KIND_PHOTO, KIND_SLIDE

# Окно транскрипта вокруг кадра, подаваемое модели (в обе стороны, секунды).
TRANSCRIPT_WINDOW_SECONDS = 60

SYSTEM_PROMPTS: dict[str, str] = {
    "ru": (
        "Ты извлекаешь содержимое слайда из кадра скринкаста лекции. На вход: изображение кадра, "
        "его таймкод и фрагмент транскрипта вокруг этого момента. Опиши, что показано на "
        "слайде/экране: суть, ключевые формулы в LaTeX ($...$), схемы и графики, важные термины. "
        "Отвечай на русском языке, но дословные фразы из транскрипта и подписи со слайда "
        "сохраняй в исходном языке. Кратко (2–4 предложения или список), без воды. Верни только "
        "описание."
    ),
    "en": (
        "You extract slide content from a lecture screencast frame. Input: the frame image, its "
        "timestamp, and the transcript excerpt around that moment. Describe what is shown: the "
        "point, key formulas in LaTeX ($...$), any diagrams/charts, and important terms. Answer "
        "in English, but keep verbatim transcript phrases and slide captions in their original "
        "language. Be concise (2–4 sentences or a bullet list), no filler. Return only the "
        "description."
    ),
}


SLIDE_SYSTEM_PROMPTS: dict[str, str] = {
    "ru": (
        "Ты извлекаешь содержимое страницы презентации к лекции. На вход: изображение "
        "одной страницы. Перечисли заголовок слайда, его тезисы, формулы в LaTeX ($...$), "
        "подписи к схемам и графикам; текст со слайда приводи по возможности дословно, на "
        "языке оригинала. Отвечай на русском языке (кроме дословного текста слайда). "
        "Кратко (2–5 предложений или список), без воды. Верни только описание."
    ),
    "en": (
        "You extract the content of a lecture presentation page. Input: an image of one "
        "page. List the title, the bullet points, formulas in LaTeX ($...$), diagram and "
        "chart captions; transcribe the slide text as verbatim as possible, in its original "
        "language. Answer in English (except verbatim slide text). Be concise (2–5 sentences "
        "or a bullet list), no filler. Return only the description."
    ),
}


PHOTO_SYSTEM_PROMPTS: dict[str, str] = {
    "ru": (
        "Ты извлекаешь содержимое фотографии доски с лекции. На вход: снимок, сделанный "
        "в аудитории. Перечисли, что на нём написано и нарисовано: формулы в LaTeX "
        "($...$), определения, схемы со стрелками и подписями, графики; надписи приводи "
        "по возможности дословно, на языке оригинала, а неразборчивые места помечай как "
        "неразборчивые. Отвечай на русском языке (кроме дословного текста). Кратко "
        "(2–5 предложений или список), без воды. Верни только описание."
    ),
    "en": (
        "You extract the content of a photo of a lecture board. Input: a snapshot taken in "
        "the classroom. List what is written and drawn on it: formulas in LaTeX ($...$), "
        "definitions, diagrams with arrows and captions, plots; transcribe the notes as "
        "verbatim as possible, in their original language, and mark illegible places as "
        "illegible. Answer in English (except verbatim text). Be concise (2–5 sentences or a "
        "bullet list), no filler. Return only the description."
    ),
}


def _prompt_language(output_language: str) -> str:
    """Язык промпта описания: всегда явный (ru/en), без автоопределения."""
    return output_language if output_language in SYSTEM_PROMPTS else "ru"


def _transcript_window(transcript: list[dict], seconds: float) -> str:
    """Фрагмент транскрипта в окне ±60 секунд вокруг кадра, строки [MM:SS] text."""
    lines = []
    for seg in transcript:
        if abs(float(seg["start"]) - seconds) <= TRANSCRIPT_WINDOW_SECONDS:
            mm = int(float(seg["start"]) // 60)
            ss = int(float(seg["start"]) % 60)
            lines.append(f"[{mm:02d}:{ss:02d}] {seg['text']}")
    if not lines:
        return "(no transcript near this frame)"
    return "\n".join(lines)


def _data_url(image_path: Path) -> str:
    """Картинка как data-URL для OpenAI-совместимого vision-запроса."""
    mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    payload = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def _vision_answer(client, model_name: str, system_prompt: str, prompt: str,
                   image_path: Path) -> str:
    """Один вызов vision-модели: системный промпт, текст и картинка; ответ модели."""
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": _data_url(image_path)}},
                ],
            },
        ],
    )
    return (response.choices[0].message.content or "").strip()


def _describe_one(client, model_name: str, system_prompt: str, image_path: Path, seconds: float,
                  timestamp: str, transcript: list[dict]) -> str:
    """Один вызов vision-модели для кадра; текст ответа."""
    window = _transcript_window(transcript, seconds)
    prompt = (
        f"Кадр на таймкоде {timestamp}.\n"
        f"Фрагмент транскрипта вокруг этого момента:\n{window}\n\n"
        "Опиши содержимое кадра."
    )
    return _vision_answer(client, model_name, system_prompt, prompt, image_path)


def describe_keyframes(vision_spec: ModelSpec, provider: Provider, keyframes: list[dict],
                       transcript: list[dict], images_dir: Path, output_language: str,
                       out_json: Path, force: bool = False) -> dict:
    """Описать каждый ключевой кадр; вернуть и записать descriptions.json.

    Формат: {"frame_HH_MM_SS.jpg": {"timestamp": "HH:MM:SS", "description": str}}.
    Язык описания задаётся явно (output_language: ru/en); дословные фразы из
    транскрипта и подписи со слайда модель сохраняет в исходном языке. Уже
    описанные кадры (без force) пропускаются. Ошибка API на кадре не прерывает
    прогон: в описание пишется текст ошибки.
    Ошибка журнала прерывает прогон и не сохраняется как описание кадра.
    """
    descriptions: dict = {}
    if out_json.exists() and not force:
        descriptions = json.loads(out_json.read_text(encoding="utf-8"))

    prompt_lang = _prompt_language(output_language)
    system_prompt = SYSTEM_PROMPTS[prompt_lang]

    client = create_client(provider)

    pending = [kf for kf in keyframes if force or kf["image"] not in descriptions]
    if not pending:
        skip_stage()
    # Прогресс считает готовые описания среди текущего набора кадров: старт
    # учитывает уже описанные (кэш без force), каждый кадр прибавляет один.
    # Кадр с ошибкой API тоже обработан (не будет повторён) и входит в счёт.
    total = len(keyframes)
    done = total - len(pending)
    report_progress("images", done, total, force=True)
    for kf in pending:
        name = kf["image"]
        image_path = images_dir / name
        try:
            with api_call("describe_keyframes", provider.name, vision_spec.api_model):
                text = _describe_one(client, vision_spec.api_model, system_prompt, image_path,
                                     kf["seconds"], kf["timestamp"], transcript)
        except JournalError:
            raise
        except Exception as exc:  # ошибка одного кадра не должна ронять весь прогон.
            text = f"(ошибка описания кадра: {exc})"
        descriptions[name] = {"timestamp": kf["timestamp"], "description": text}
        out_json.write_text(json.dumps(descriptions, ensure_ascii=False, indent=2), encoding="utf-8")
        done += 1
        report_progress("images", done, total, force=done == total)
    return descriptions


def describe_slides(vision_spec: ModelSpec, provider: Provider, slides: list[dict],
                    images_dir: Path, output_language: str, out_json: Path,
                    force: bool = False) -> dict:
    """Описать каждую страницу презентации и фотографию доски; записать JSON.

    Формат: {"slide_003.png": {"page": 3, "source": "/путь/колода.pdf",
    "kind": "slide", "description": str}} и {"photo_001.jpg": {"page": 1,
    "source": "/путь/доска.jpg", "kind": "photo", "description": str}}. Язык
    описания — output_language (ru/en), промпт выбирается по нему с запасом на
    'ru'. Вид материала (kind, по умолчанию страница презентации) выбирает
    промпт: у страницы колоды и у снимка доски разное содержимое — печатный
    слайд против рукописной записи. Уже описанные страницы (без force)
    пропускаются, ошибка API на странице не прерывает прогон: в описание
    пишется текст ошибки. Транскрипт лекции не передаётся: ни у страницы, ни у
    снимка нет таймкода, сопоставление делает синтез по содержанию.
    Ошибка журнала прерывает прогон и не сохраняется как описание материала.
    """
    descriptions: dict = {}
    if out_json.exists() and not force:
        descriptions = json.loads(out_json.read_text(encoding="utf-8"))

    language = _prompt_language(output_language)
    slide_prompt = SLIDE_SYSTEM_PROMPTS[language]
    photo_prompt = PHOTO_SYSTEM_PROMPTS[language]

    client = create_client(provider)

    pending = [entry for entry in slides if force or entry["image"] not in descriptions]
    if not pending:
        skip_stage()
    # Как и у кадров: счёт ведётся по текущему набору материалов и учитывает кэш.
    total = len(slides)
    done = total - len(pending)
    report_progress("images", done, total, force=True)
    for entry in pending:
        name = entry["image"]
        image_path = images_dir / name
        source_name = Path(entry["source"]).name
        if entry.get("kind", KIND_SLIDE) == KIND_PHOTO:
            system_prompt = photo_prompt
            prompt = (f"Фотография доски (файл {source_name}), снятая на лекции.\n"
                      "Опиши содержимое снимка.")
        else:
            system_prompt = slide_prompt
            prompt = (f"Слайд {entry['page']} из {len(slides)} "
                      f"(файл {source_name}).\n"
                      "Опиши содержимое слайда.")
        try:
            with api_call("describe_slides", provider.name, vision_spec.api_model):
                text = _vision_answer(client, vision_spec.api_model, system_prompt, prompt, image_path)
        except JournalError:
            raise
        except Exception as exc:  # ошибка одной страницы не должна ронять весь прогон.
            text = f"(ошибка описания: {exc})"
        descriptions[name] = {"page": entry["page"], "source": entry["source"],
                              "kind": entry.get("kind", KIND_SLIDE), "description": text}
        out_json.write_text(json.dumps(descriptions, ensure_ascii=False, indent=2), encoding="utf-8")
        done += 1
        report_progress("images", done, total, force=done == total)
    return descriptions
