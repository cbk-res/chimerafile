"""Diffusion engine: /v1/images/generations + /v1/images/edits.

Run with the media config (tiny llama + whisper + diffusion):

    pytest tests/test_media_images.py --serve-config tests/configs/tiny-media.json

Generations are kept small (256x256, 8 steps) so the tests stay fast on
the CPU; point a GPU config with a bigger model at this module for the
full-size path.
"""

import base64
import json
import struct
import zlib

import pytest

W = 256
STEPS = 8
PROMPT = "a red square on a white background"


def media_model(server, engine):
    for mid, e in server.cfg.get("models", {}).items():
        if e.get("engine") == engine:
            return mid
    return None


def need_model(server, mid):
    if not mid:
        pytest.skip("no diffusion model in the config "
                    "(tests/configs/tiny-media.json)")
    return mid


def png(width=64, height=64, rgb=(200, 30, 30)):
    """A solid-colour PNG with no image library available."""
    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def decode_image(r):
    """Extract and decode the first image of an OpenAI images response.

    Returns (png-bytes, width, height)."""
    data = r.get("data")
    assert isinstance(data, list) and data, f"no data: {str(r)[:300]}"
    item = data[0]
    b64 = item.get("b64_json")
    assert b64, f"no b64_json in {str(item)[:200]}"
    img = base64.b64decode(b64)
    assert img[:8] == b"\x89PNG\r\n\x1a\n", \
        f"not a PNG: {img[:8]!r} ({len(img)} bytes)"
    w, h = struct.unpack(">II", img[16:24])
    return img, w, h


def test_image_generation_returns_image(server):
    mid = need_model(server, media_model(server, "diffusion"))
    s, data = server.client.post(
        "/v1/images/generations",
        {"model": mid, "prompt": PROMPT, "size": f"{W}x{W}",
         "steps": STEPS})
    assert s == 200, f"status={s} body={str(data)[:300]}"
    _, w, h = decode_image(json.loads(data))
    assert (w, h) == (W, W), f"requested {W}x{W}, got {w}x{h}"


def test_image_edits(server):
    mid = need_model(server, media_model(server, "diffusion"))
    s, data = server.client.multipart(
        "/v1/images/edits",
        {"model": mid, "prompt": "make it blue", "steps": str(STEPS)},
        "image", "input.png", png(), file_type="image/png")
    assert s == 200, f"status={s} body={str(data)[:300]}"
    decode_image(json.loads(data))


def test_image_generation_error_matrix(server):
    c = server.client
    # Missing prompt.
    s, _ = c.post("/v1/images/generations", {"model": server.model})
    assert s == 400, f"expected 400 for missing prompt, got {s}"
    # Unknown model.
    s, _ = c.post("/v1/images/generations", {"model": "nope",
                                             "prompt": PROMPT})
    assert s == 404, f"expected 404 for unknown model, got {s}"
    # A llama model: unsupported engine.
    s, _ = c.post("/v1/images/generations",
                  {"model": server.model, "prompt": PROMPT})
    assert s == 400, f"expected 400 for wrong engine, got {s}"
    # Malformed JSON.
    s, _ = c.raw("/v1/images/generations", b'{"model": ')
    assert s == 400, f"expected 400 for bad JSON, got {s}"
