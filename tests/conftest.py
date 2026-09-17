"""Общие фикстуры и подделки для тестов Absorbent."""

from __future__ import annotations

from pathlib import Path

import pytest

# Репозиторий: корень проекта (там лежит config.yaml).
REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sample_config(tmp_path: Path) -> Path:
    """Копия штатного config.yaml во временной директории."""
    target = tmp_path / "config.yaml"
    target.write_bytes((REPO_ROOT / "config.yaml").read_bytes())
    return target


@pytest.fixture
def minimal_config(tmp_path: Path) -> Path:
    """Минимальная конфигурация: по одной модели в каждой категории."""
    target = tmp_path / "config.yaml"
    target.write_text(
        """
providers:
  prov-a:
    base_url: https://api.example.test/v1
    api_key_env: TEST_KEY_A
stt:
  whisper-local:
    backend: whisper
    whisper_model: tiny
    default: true
vision:
  vis-a:
    provider: prov-a
    default: true
llm:
  llm-a:
    provider: prov-a
    default: true
""",
        encoding="utf-8",
    )
    return target


class FakeChatCompletions:
    """Подделка chat.completions: возвращает заданные тексты ответов.

    finish_reason, usage и reasoning соответствуют реальному ответу: их
    читает диагностика пустого ответа в synthesize._empty_answer_details().
    """

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[dict] = []
        self.finish_reason = "stop"
        self.reasoning: str | None = None

    def create(self, **kwargs) -> object:
        self.calls.append(kwargs)
        content = self.replies.pop(0) if self.replies else "ответ модели"
        message = type("Msg", (), {"content": content,
                                   "model_extra": {"reasoning_content": self.reasoning}})()
        choice = type("Choice", (), {"message": message, "finish_reason": self.finish_reason})()
        usage = type("Usage", (), {
            "completion_tokens": 123,
            "completion_tokens_details": type("Details", (), {"reasoning_tokens": 99})(),
        })()
        return type("Resp", (), {"choices": [choice], "usage": usage})()


class FakeAudioTranscriptions:
    """Подделка audio.transcriptions: отдаёт заранее заданный JSON."""

    def __init__(self, payload: dict | None = None) -> None:
        self.payload = payload or {"text": "", "language": None, "segments": [], "usage": {}}
        self.calls: list[dict] = []
        self._raw = _RawResponse(self, self.calls)

    @property
    def with_raw_response(self) -> "_RawResponse":
        return self._raw


class _RawResponse:
    """Результат with_raw_response.create(...): сохраняет kwargs и отдаёт JSON.

    Payload читается из родительского объекта на момент вызова json(),
    чтобы тест мог назначить payload уже после создания экземпляра.
    """

    def __init__(self, owner: FakeAudioTranscriptions, calls: list[dict]) -> None:
        self.owner = owner
        self.calls = calls

    def create(self, **kwargs) -> "_RawResponse":
        self.calls.append(kwargs)
        return self

    def json(self) -> dict:
        return self.owner.payload


class _AudioProxy:
    """Прокси client.audio: держит один экземпляр audio.transcriptions."""

    def __init__(self, transcriptions: FakeAudioTranscriptions) -> None:
        self.transcriptions = transcriptions


class _ChatProxy:
    """Прокси client.chat: держит один экземпляр chat.completions."""

    def __init__(self, completions: FakeChatCompletions) -> None:
        self.completions = completions


class FakeOpenAI:
    """Минимальный двойник openai.OpenAI для chat- и audio-вызовов.

    Все экземпляры разделяют одно состояние (audio/chat): код модулей
    создаёт собственные экземпляры OpenAI(), а тест настраивает ответы через
    фикстуру — состояние должно быть общим, иначе ответы не увидятся.
    """

    instances: list["FakeOpenAI"] = []
    _shared_audio: FakeAudioTranscriptions | None = None
    _shared_chat: FakeChatCompletions | None = None

    def __init__(self, base_url: str = "", api_key: str | None = None, **kwargs) -> None:
        self.base_url = base_url
        self.api_key = api_key
        if FakeOpenAI._shared_audio is None:
            FakeOpenAI._shared_audio = FakeAudioTranscriptions()
        if FakeOpenAI._shared_chat is None:
            FakeOpenAI._shared_chat = FakeChatCompletions([])
        FakeOpenAI.instances.append(self)

    # Путь доступа кода: client.audio.transcriptions.with_raw_response.create(...).
    @property
    def audio(self) -> _AudioProxy:
        return _AudioProxy(FakeOpenAI._shared_audio)

    @property
    def chat(self) -> _ChatProxy:
        return _ChatProxy(FakeOpenAI._shared_chat)

    def set_replies(self, replies: list[str]) -> None:
        FakeOpenAI._shared_chat.replies = list(replies)

    @classmethod
    def reset(cls) -> None:
        cls.instances = []
        cls._shared_audio = None
        cls._shared_chat = None


@pytest.fixture
def fake_openai(monkeypatch: pytest.MonkeyPatch) -> FakeOpenAI:
    """Подменяет openai.OpenAI на FakeOpenAI во всех модулях absorbent."""
    import sys

    FakeOpenAI.reset()
    fake_module = type("fake_openai_module", (), {"OpenAI": FakeOpenAI})()
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    return FakeOpenAI()


@pytest.fixture
def fake_whisper(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Подменяет torch/whisper: load_model возвращает записывающую подделку."""
    import sys
    import types

    state: dict = {"load_calls": [], "transcribe_calls": [], "segments": [], "language": None}

    class FakeModel:
        def transcribe(self, path, **kwargs):
            state["transcribe_calls"].append({"path": path, **kwargs})
            return {"segments": state["segments"], "language": state["language"]}

    def _load_model(name, device=None):
        state["load_calls"].append((name, device))
        return FakeModel()

    def _cuda_available() -> bool:
        return state.get("cuda", True)

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.ModuleType("torch.cuda")
    fake_torch.cuda.is_available = _cuda_available
    fake_whisper_mod = types.ModuleType("whisper")
    fake_whisper_mod.load_model = _load_model
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper_mod)
    return state


@pytest.fixture
def args_namespace() -> object:
    """Namespace с дефолтами флагов командной строки для pipeline.process_file."""
    from types import SimpleNamespace

    return SimpleNamespace(
        output_dir=None,
        flat=False,
        fps=1.0,
        hash_threshold=10,
        min_scene_seconds=5.0,
        force=False,
        skip_stt=False,
        skip_frames=False,
        skip_describe=False,
        skip_synthesize=False,
        skip_render=False,
        language="auto",
        output_language="ru",
        slides=[],
        slides_dpi=150,
        skip_slides=False,
        stt=None,
        vision_model=None,
        llm_model=None,
    )
