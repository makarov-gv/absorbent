"""Наблюдаемые границы SDK: реальные попытки HTTP, расход, редактирование и ошибки."""

from __future__ import annotations

import json
from pathlib import Path

import httpx2
import pytest

from absorbent.api import JournalHTTPClient, create_client
from absorbent.config import ConfigError, ModelSpec, Provider
from absorbent.describe import describe_keyframes, describe_slides
from absorbent.events import Journal, JournalError, api_call, stage
from absorbent.synthesize import run_synthesize
from absorbent.transcribe import run_stt


PROVIDER = Provider(name="local", base_url="https://model.example.test/v1", api_key_env="")
SPEC = ModelSpec(name="selected", provider="local", model="effective-model")


def _rows(path: Path, event: str) -> list[dict]:
    return [row for line in path.read_text(encoding="utf-8").splitlines()
            if (row := json.loads(line))["event"] == event]


def _completion(text: str = "# Конспект", usage=None) -> dict:
    data = {"id": "response", "object": "chat.completion", "created": 0, "model": "effective-model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}]}
    if usage is not None:
        data["usage"] = usage
    return data


@pytest.fixture
def transport_factory(monkeypatch: pytest.MonkeyPatch):
    """Подставить настоящий HTTP-клиент с управляемым транспортом без внешней сети."""
    monkeypatch.setattr("openai._base_client.time.sleep", lambda seconds: None)
    clients = []

    def install(handler):
        def factory():
            client = JournalHTTPClient(transport=httpx2.MockTransport(handler), trust_env=False)
            clients.append(client)
            return client
        monkeypatch.setattr("absorbent.api.JournalHTTPClient", factory)

    yield install
    for client in clients:
        client.close()


@pytest.mark.parametrize("first_failure", ["http", "transport", "read", "decoding"])
def test_sdk_retries_are_separate_closed_attempts(tmp_path: Path, transport_factory, first_failure: str) -> None:
    calls = []

    class BrokenBody(httpx2.SyncByteStream):
        def __iter__(self):
            raise httpx2.ReadError("lost body")
            yield b""

    def handle(request):
        calls.append(request)
        if len(calls) == 1:
            if first_failure == "http":
                return httpx2.Response(500, json={"error": {"message": "failed"}})
            if first_failure == "decoding":
                return httpx2.Response(200, headers={"content-encoding": "gzip"},
                                       stream=httpx2.ByteStream(b"not-gzip"))
            if first_failure == "read":
                return httpx2.Response(200, stream=BrokenBody())
            raise httpx2.ConnectError("offline", request=request)
        return httpx2.Response(200, json=_completion(usage={"prompt_tokens": 5, "completion_tokens": 2}))

    transport_factory(handle)
    path = tmp_path / "events.jsonl"
    with Journal(path), stage("synthesize"), api_call("synthesize", PROVIDER.name, SPEC.api_model):
        with create_client(PROVIDER) as client:
            result = client.chat.completions.create(model=SPEC.api_model, messages=[{"role": "user", "content": "private"}])
    assert result.choices[0].message.content == "# Конспект"
    starts = _rows(path, "api_request_start")
    ends = _rows(path, "api_request_end")
    assert len(calls) == len(starts) == len(ends) == 2
    assert len({row["request_id"] for row in starts}) == 2
    assert [row["request_id"] for row in starts] == [row["request_id"] for row in ends]
    assert ends[0]["usage"] is None and ends[0]["billing"] == "unknown"
    assert ends[0]["error_type"] == {
        "http": None, "transport": "ConnectError", "read": "ReadError", "decoding": "DecodingError",
    }[first_failure]
    assert ends[0]["status_code"] == (500 if first_failure == "http" else None)
    assert ends[1]["status_code"] == 200
    assert ends[1]["usage"]["prompt_tokens"] == 5
    assert ends[1]["usage"]["completion_tokens"] == 2
    assert all(row["provider"] == "local" and row["model"] == SPEC.api_model for row in ends)


def test_redirects_are_individual_attempts(tmp_path: Path) -> None:
    calls = []

    def handle(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx2.Response(307, headers={"location": "https://model.example.test/redirected"})
        return httpx2.Response(200, json={"usage": {"prompt_tokens": 0}})

    path = tmp_path / "events.jsonl"
    with Journal(path), JournalHTTPClient(transport=httpx2.MockTransport(handle), trust_env=False) as client:
        assert client.get("https://model.example.test/original").status_code == 200
    assert len(calls) == len(_rows(path, "api_request_end")) == 2
    assert _rows(path, "api_request_end")[-1]["usage"]["prompt_tokens"] == 0


@pytest.mark.parametrize("payload", [None, [], {"usage": None}, {"usage": "private"}, {"usage": {}}])
def test_missing_usage_is_unknown_not_zero(tmp_path: Path, payload) -> None:
    response = httpx2.Response(200, text="not JSON") if payload is None else httpx2.Response(200, json=payload)
    path = tmp_path / "events.jsonl"
    with Journal(path), JournalHTTPClient(transport=httpx2.MockTransport(lambda request: response), trust_env=False) as client:
        client.get("https://model.example.test/usage")
    usage = _rows(path, "api_request_end")[0]["usage"]
    if payload == {"usage": {}}:
        assert all(value is None for value in usage.values())
    else:
        assert usage is None


def test_usage_redaction_cache_reasoning_and_audio(tmp_path: Path) -> None:
    secret = "SECRET-PROMPT-KEY-IMAGE-RESPONSE"
    usage = {"prompt_tokens": 40, "completion_tokens": 9, "prompt_cache_hit_tokens": 30,
             "prompt_cache_miss_tokens": 10, "prompt_tokens_details": {"cached_tokens": 99, "audio_tokens": 3},
             "completion_tokens_details": {"reasoning_tokens": 4, "audio_tokens": 2}, "seconds": 1.5,
             "prompt_audio_seconds": 2.5, "completion_audio_seconds": 0,
             "input_tokens": 3, "output_tokens": 2, "total_tokens": 5, "private": secret}
    path = tmp_path / "events.jsonl"
    with Journal(path), api_call("stt", "provider", "model"), JournalHTTPClient(
            transport=httpx2.MockTransport(lambda request: httpx2.Response(
                200, json={"usage": usage, "text": secret}, headers={"x-private": secret})), trust_env=False) as client:
        response = client.post("https://example.test/" + secret, headers={"Authorization": secret},
                               json={"prompt": secret, "image": secret})
        assert response.json()["text"] == secret
    row = _rows(path, "api_request_end")[0]
    assert secret not in path.read_text(encoding="utf-8")
    assert row["stage"] == "stt" and row["provider"] == "provider" and row["model"] == "model"
    assert row["usage"] == {"prompt_tokens": 40, "completion_tokens": 9, "cache_hit": 30, "cache_miss": 10,
                            "reasoning_tokens": 4, "audio_tokens": 3, "completion_audio_tokens": 2,
                            "audio_seconds": 1.5, "prompt_audio_seconds": 2.5, "completion_audio_seconds": 0,
                            "input_tokens": 3, "output_tokens": 2, "total_tokens": 5}


@pytest.mark.parametrize("value", ["private", True, -1, -1.5, float("inf"), float("nan")])
def test_invalid_usage_values_do_not_leak_or_break_http(tmp_path: Path, value) -> None:
    path = tmp_path / "events.jsonl"
    payload = {"usage": {"prompt_tokens": value, "prompt_tokens_details": "private",
                         "completion_tokens_details": []}}
    with Journal(path), JournalHTTPClient(transport=httpx2.MockTransport(
            lambda request: httpx2.Response(200, content=json.dumps(payload))), trust_env=False) as client:
        assert client.get("https://example.test").status_code == 200
    assert _rows(path, "api_request_end")[0]["usage"]["prompt_tokens"] is None


def test_standard_cached_tokens_without_inventing_miss(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    with Journal(path), JournalHTTPClient(transport=httpx2.MockTransport(lambda request: httpx2.Response(
            200, json={"usage": {"prompt_tokens_details": {"cached_tokens": 0}}})), trust_env=False) as client:
        client.get("https://example.test")
    usage = _rows(path, "api_request_end")[0]["usage"]
    assert usage["cache_hit"] == 0 and usage["cache_miss"] is None


def test_response_journal_failure_never_repeats_paid_request(tmp_path: Path, monkeypatch, transport_factory) -> None:
    calls = []

    def handle(request):
        calls.append(request)
        return httpx2.Response(200, json=_completion(usage={"prompt_tokens": 1, "completion_tokens": 1}))

    original = Journal.write

    def write(self, event, **fields):
        if event == "api_request_end":
            raise JournalError("disk full")
        original(self, event, **fields)

    transport_factory(handle)
    monkeypatch.setattr(Journal, "write", write)
    path = tmp_path / "events.jsonl"
    with pytest.raises(JournalError, match="disk full"):
        with Journal(path), stage("synthesize"):
            run_synthesize(SPEC, PROVIDER, [], {}, "ru", tmp_path / "notes.md")
    assert len(calls) == 1
    assert not (tmp_path / "notes.md").exists()
    assert _rows(path, "stage_end")[0]["status"] == "error"


def test_empty_synthesis_accounts_for_both_paid_responses(tmp_path: Path, transport_factory) -> None:
    calls = []

    def handle(request):
        calls.append(request)
        return httpx2.Response(200, json=_completion("" if len(calls) == 1 else "# Готово",
                                                    {"prompt_tokens": 3, "completion_tokens": 7}))

    transport_factory(handle)
    path = tmp_path / "events.jsonl"
    with Journal(path), stage("synthesize"):
        run_synthesize(SPEC, PROVIDER, [], {}, "ru", tmp_path / "notes.md")
    ends = _rows(path, "api_request_end")
    assert len(ends) == len(calls) == 2
    assert sum(row["usage"]["completion_tokens"] for row in ends) == 14
    assert (tmp_path / "notes.md").read_text(encoding="utf-8") == "# Готово"


@pytest.mark.parametrize("kind", ["keyframes", "slides"])
def test_fatal_vision_journal_failure_is_not_cached(tmp_path: Path, transport_factory, monkeypatch, kind: str) -> None:
    calls = []

    def handle(request):
        calls.append(request)
        return httpx2.Response(200, json=_completion("описание"))

    original = Journal.write

    def write(self, event, **fields):
        if event == "api_request_end":
            raise JournalError("journal lost")
        original(self, event, **fields)

    transport_factory(handle)
    monkeypatch.setattr(Journal, "write", write)
    image = tmp_path / "image.jpg"
    image.write_bytes(b"jpeg")
    out = tmp_path / "descriptions.json"
    path = tmp_path / "events.jsonl"
    with pytest.raises(JournalError, match="journal lost"):
        with Journal(path), stage("describe_" + kind):
            if kind == "keyframes":
                describe_keyframes(SPEC, PROVIDER, [{"image": image.name, "seconds": 0, "timestamp": "00:00:00"}],
                                   [], tmp_path, "ru", out)
            else:
                describe_slides(SPEC, PROVIDER, [{"image": image.name, "page": 1, "source": "material.pdf"}],
                                tmp_path, "ru", out)
    assert len(calls) == 1 and not out.exists()
    assert _rows(path, "stage_end")[0]["status"] == "error"


def test_voxtral_audio_usage_and_context(tmp_path: Path, transport_factory, monkeypatch) -> None:
    def handle(request):
        return httpx2.Response(200, json={"text": "речь", "segments": [], "language": "ru",
                                        "usage": {"prompt_audio_seconds": 2.5, "total_tokens": 12}})

    transport_factory(handle)
    monkeypatch.setattr("absorbent.transcribe._extract_audio", lambda source, target: target.write_bytes(b"wav"))
    monkeypatch.setattr("absorbent.transcribe._audio_duration_seconds", lambda source: 2.5)
    path = tmp_path / "events.jsonl"
    with Journal(path), stage("stt"):
        assert run_stt(ModelSpec(name="audio", backend="voxtral", model="voxtral"), PROVIDER,
                       tmp_path / "lecture.wav", "auto", tmp_path / "transcript.json", tmp_path) == "ru"
    end = _rows(path, "api_request_end")[0]
    assert end["stage"] == "stt" and end["model"] == "voxtral"
    assert end["usage"]["prompt_audio_seconds"] == 2.5


def test_client_without_journal_never_uses_ambient_openai_key(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "foreign-secret")
    with create_client(PROVIDER) as client:
        assert client.api_key != "foreign-secret"
    monkeypatch.setenv("OWN_KEY", "own-secret")
    with create_client(Provider(name="cloud", base_url=PROVIDER.base_url, api_key_env="OWN_KEY")) as client:
        assert client.api_key == "own-secret"
    monkeypatch.delenv("OWN_KEY")
    with pytest.raises(ConfigError, match="OWN_KEY"):
        create_client(Provider(name="cloud", base_url=PROVIDER.base_url, api_key_env="OWN_KEY"))


def test_instrumented_client_without_active_sink_preserves_response() -> None:
    with JournalHTTPClient(transport=httpx2.MockTransport(lambda request: httpx2.Response(
            200, json={"usage": {"prompt_tokens": 1}, "text": "response"})), trust_env=False) as client:
        assert client.get("https://example.test").json()["text"] == "response"
    with JournalHTTPClient(transport=httpx2.MockTransport(lambda request: (_ for _ in ()).throw(
            httpx2.ConnectError("offline", request=request))), trust_env=False) as client:
        with pytest.raises(httpx2.ConnectError):
            client.get("https://example.test")


@pytest.mark.parametrize("kind", ["keyframes", "slides", "stt"])
def test_cached_model_stages_skip_without_http(tmp_path: Path, transport_factory, kind: str) -> None:
    calls = []
    transport_factory(lambda request: calls.append(request))
    out = tmp_path / "cached.json"
    out.write_text(json.dumps([] if kind == "stt" else {
        "image.jpg": {"timestamp": "00:00:00", "description": "cached", "page": 1, "source": "material.pdf"}}),
        encoding="utf-8")
    path = tmp_path / "events.jsonl"
    stage_name = "stt" if kind == "stt" else "describe_" + kind
    with Journal(path), stage(stage_name):
        if kind == "keyframes":
            result = describe_keyframes(SPEC, PROVIDER, [{"image": "image.jpg", "seconds": 0, "timestamp": "00:00:00"}],
                                       [], tmp_path, "ru", out)
            assert result["image.jpg"]["description"] == "cached"
        elif kind == "slides":
            result = describe_slides(SPEC, PROVIDER, [{"image": "image.jpg", "page": 1, "source": "material.pdf"}],
                                    tmp_path, "ru", out)
            assert result["image.jpg"]["description"] == "cached"
        else:
            assert run_stt(ModelSpec(name="whisper", backend="whisper"), PROVIDER, tmp_path / "lecture.wav",
                           "auto", out, tmp_path) is None
    assert calls == []
    assert _rows(path, "stage_end")[0]["status"] == "skipped"


def test_transcription_token_usage_reports_audio_details(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    with Journal(path), JournalHTTPClient(transport=httpx2.MockTransport(lambda request: httpx2.Response(
            200, json={"usage": {"type": "tokens", "input_tokens": 10, "output_tokens": 2,
                                 "total_tokens": 12, "input_token_details": {"audio_tokens": 8}}})),
            trust_env=False) as client:
        client.get("https://example.test/audio")
    usage = _rows(path, "api_request_end")[0]["usage"]
    assert usage["audio_tokens"] == 8
    assert usage["input_tokens"] == 10 and usage["output_tokens"] == 2
    assert usage["prompt_tokens"] is None
