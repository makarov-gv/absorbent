"""Описание ключевых кадров vision-моделью с привязкой к транскрипту."""

from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path

from absorbent.config import ModelSpec, Provider, api_key_from

# Окно транскрипта вокруг кадра, подаваемое модели (в обе стороны, секунды).
TRANSCRIPT_WINDOW_SECONDS = 60

SYSTEM_PROMPTS: dict[str, str] = {
    "ru": (
        "Ты извлекаешь содержимое слайда из кадра скринкаста лекции. На вход: изображение кадра, "
        "его таймкод и фрагмент транскрипта вокруг этого момента. Опиши, что показано на "
        "слайде/экране: суть, ключевые формулы в LaTeX ($...$), схемы и графики, важные термины. "
        "Кратко (2–4 предложения или список), без воды. Верни только описание."
    ),
    "en": (
        "You extract slide content from a lecture screencast frame. Input: the frame image, its "
        "timestamp, and the transcript excerpt around that moment. Describe what is shown: the "
        "point, key formulas in LaTeX ($...$), any diagrams/charts, and important terms. Be "
        "concise (2–4 sentences or a bullet list), no filler. Return only the description."
    ),
}


def _prompt_language(language: str, detected: str | None) -> str:
    """Язык промпта: явный язык, иначе детектированный, иначе 'ru'."""
    if language != "auto":
        return language
    if detected and detected.startswith("en"):
        return "en"
    return "ru"


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


def _describe_one(client, model_name: str, system_prompt: str, image_path: Path, seconds: float,
                  timestamp: str, transcript: list[dict]) -> str:
    """Один вызов vision-модели для кадра; текст ответа."""
    window = _transcript_window(transcript, seconds)
    prompt = (
        f"Кадр на таймкоде {timestamp}.\n"
        f"Фрагмент транскрипта вокруг этого момента:\n{window}\n\n"
        "Опиши содержимое кадра."
    )
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


def describe_keyframes(vision_spec: ModelSpec, provider: Provider, keyframes: list[dict],
                       transcript: list[dict], images_dir: Path, language: str, out_json: Path,
                       detected_language: str | None = None, force: bool = False) -> dict:
    """Описать каждый ключевой кадр; вернуть и записать descriptions.json.

    Формат: {"frame_HH_MM_SS.jpg": {"timestamp": "HH:MM:SS", "description": str}}.
    Уже описанные кадры (без force) пропускаются. Ошибка API на кадре не
    прерывает прогон: в описание пишется текст ошибки.
    """
    from openai import OpenAI

    descriptions: dict = {}
    if out_json.exists() and not force:
        descriptions = json.loads(out_json.read_text(encoding="utf-8"))

    prompt_lang = _prompt_language(language, detected_language)
    system_prompt = SYSTEM_PROMPTS[prompt_lang]

    api_key = api_key_from(provider)
    client = OpenAI(base_url=provider.base_url, api_key=api_key)

    pending = [kf for kf in keyframes if force or kf["image"] not in descriptions]
    for kf in pending:
        name = kf["image"]
        image_path = images_dir / name
        try:
            text = _describe_one(client, vision_spec.api_model, system_prompt, image_path,
                                 kf["seconds"], kf["timestamp"], transcript)
        except Exception as exc:  # ошибка одного кадра не должна ронять весь прогон.
            text = f"(ошибка описания кадра: {exc})"
        descriptions[name] = {"timestamp": kf["timestamp"], "description": text}
        out_json.write_text(json.dumps(descriptions, ensure_ascii=False, indent=2), encoding="utf-8")
    return descriptions
