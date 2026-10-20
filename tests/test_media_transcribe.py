"""Transcribe engine: /v1/audio/transcriptions with a transcribe model.

The endpoint is engine-agnostic: the same OpenAI surface serves whisper
and transcribe models alike, selected by the request's model id --
swapping the engine is a config entry, not a route change.  Run with the
sdcpp media config (it carries the canary-180m transcribe model):

    pytest tests/test_media_transcribe.py \
        --serve-config tests/configs/tiny-media-sdcpp.json

The success-path tests synthesise a short speech fixture with espeak-ng
at test time and skip when it is unavailable.
"""

import json
import shutil
import subprocess

import pytest

PHRASE = "The quick brown fox jumps over the lazy dog."


def transcribe_model(server):
    for mid, e in server.cfg.get("models", {}).items():
        if e.get("engine") == "transcribe":
            return mid
    return None


def need_model(server, mid):
    if not mid:
        pytest.skip("no transcribe model in the config "
                    "(tests/configs/tiny-media-sdcpp.json)")
    return mid


@pytest.fixture(scope="module")
def speech_wav():
    if not shutil.which("espeak-ng"):
        pytest.skip("espeak-ng not available; cannot synthesise a fixture")
    out = "/tmp/opencode/chimera-pytest-speech-transcribe.wav"
    # The canary-180m model transcribes espeak's default (en-gb, 175 wpm)
    # voice to empty; en-us at 120 wpm is reliable on every backend tested.
    subprocess.run(["espeak-ng", "-v", "en-us", "-s", "120",
                    "-w", out, PHRASE], check=True)
    with open(out, "rb") as f:
        return f.read()


def test_transcribe_engine_returns_text(server, speech_wav):
    mid = need_model(server, transcribe_model(server))
    s, data = server.client.multipart(
        "/v1/audio/transcriptions",
        {"model": mid, "response_format": "json"},
        "file", "speech.wav", speech_wav)
    assert s == 200, f"status={s} body={str(data)[:300]}"
    text = (json.loads(data).get("text") or "").strip().lower()
    # espeak is a robotic voice; require a couple of the phrase's words.
    hits = sum(w in text for w in ("quick", "brown", "fox", "lazy", "dog"))
    assert hits >= 2, f"expected words from {PHRASE!r} in {text!r}"


def test_transcribe_engine_verbose_json(server, speech_wav):
    mid = need_model(server, transcribe_model(server))
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


def test_transcribe_rejects_non_speech_model(server):
    need_model(server, transcribe_model(server))
    mid = None
    for m, e in server.cfg.get("models", {}).items():
        if e.get("engine") not in ("whisper", "transcribe"):
            mid = m
            break
    if not mid:
        pytest.skip("no non-speech model in the config")
    s, data = server.client.multipart(
        "/v1/audio/transcriptions",
        {"model": mid, "response_format": "json"},
        "file", "speech.wav", b"RIFF")
    assert s == 400, f"status={s} body={str(data)[:300]}"
    assert "speech" in str(data).lower(), data
