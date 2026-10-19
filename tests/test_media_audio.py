"""Whisper engine: /v1/audio/transcriptions (the OpenAI surface).

Run with the media config (tiny llama + whisper + diffusion):

    pytest tests/test_media_audio.py --serve-config tests/configs/tiny-media.json

The success-path tests synthesise a short speech fixture with espeak-ng
at test time and skip when it is unavailable.
"""

import json
import shutil
import subprocess

import pytest

PHRASE = "The quick brown fox jumps over the lazy dog."


def media_model(server, engine):
    for mid, e in server.cfg.get("models", {}).items():
        if e.get("engine") == engine:
            return mid
    return None


def need_model(server, mid):
    if not mid:
        pytest.skip("no whisper model in the config "
                    "(tests/configs/tiny-media.json)")
    return mid


@pytest.fixture(scope="module")
def speech_wav():
    if not shutil.which("espeak-ng"):
        pytest.skip("espeak-ng not available; cannot synthesise a fixture")
    out = "/tmp/opencode/chimera-pytest-speech.wav"
    subprocess.run(["espeak-ng", "-w", out, PHRASE], check=True)
    with open(out, "rb") as f:
        return f.read()


def test_transcription_returns_text(server, speech_wav):
    mid = need_model(server, media_model(server, "whisper"))
    s, data = server.client.multipart(
        "/v1/audio/transcriptions",
        {"model": mid, "response_format": "json"},
        "file", "speech.wav", speech_wav)
    assert s == 200, f"status={s} body={str(data)[:300]}"
    text = (json.loads(data).get("text") or "").strip().lower()
    # espeak is a robotic voice; require a couple of the phrase's words.
    hits = sum(w in text for w in ("quick", "brown", "fox", "lazy", "dog"))
    assert hits >= 2, f"expected words from {PHRASE!r} in {text!r}"


def test_transcription_verbose_json(server, speech_wav):
    mid = need_model(server, media_model(server, "whisper"))
    s, data = server.client.multipart(
        "/v1/audio/transcriptions",
        {"model": mid, "response_format": "verbose_json"},
        "file", "speech.wav", speech_wav)
    assert s == 200, f"status={s} body={str(data)[:300]}"
    r = json.loads(data)
    assert r.get("text"), f"missing text: {r}"
    segs = r.get("segments")
    assert isinstance(segs, list) and segs, f"missing segments: {r}"
    for seg in segs:
        assert set(seg) >= {"start", "end", "text"}, f"segment: {seg}"


def test_transcription_text_format(server, speech_wav):
    mid = need_model(server, media_model(server, "whisper"))
    s, data = server.client.multipart(
        "/v1/audio/transcriptions",
        {"model": mid, "response_format": "text"},
        "file", "speech.wav", speech_wav)
    assert s == 200, f"status={s} body={str(data)[:300]}"
    assert isinstance(data, bytes)
    assert data.strip(), "empty text response"


def test_transcription_error_matrix(server):
    c = server.client
    # Plain JSON, not multipart.
    s, _ = c.raw("/v1/audio/transcriptions", b'{"model": "x"}')
    assert s == 400, f"expected 400 for non-multipart, got {s}"
    # Multipart without a model field.
    s, _ = c.multipart("/v1/audio/transcriptions", {}, "file",
                       "speech.wav", b"RIFFxxxx")
    assert s == 400, f"expected 400 for missing model, got {s}"
    # Unknown model id.
    s, _ = c.multipart("/v1/audio/transcriptions", {"model": "nope"},
                       "file", "speech.wav", b"RIFFxxxx")
    assert s == 404, f"expected 404 for unknown model, got {s}"
    # A llama model: unsupported engine.
    s, _ = c.multipart("/v1/audio/transcriptions", {"model": server.model},
                       "file", "speech.wav", b"RIFFxxxx")
    assert s == 400, f"expected 400 for wrong engine, got {s}"
    # Multipart with fields but no file part.
    body = (b"----t\r\n"
            b'Content-Disposition: form-data; name="model"\r\n\r\n'
            + server.model.encode() + b"\r\n----t--\r\n")
    s, _ = c._request("/v1/audio/transcriptions", body=body,
                      headers={"Content-Type":
                               "multipart/form-data; boundary=t"})
    assert s == 400, f"expected 400 for missing file, got {s}"
