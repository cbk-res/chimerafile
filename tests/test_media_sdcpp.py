"""sdcpp surface: /sdcpp/v1/* (mirrors stable-diffusion.cpp's own server).

Run with the sdcpp media config (tiny llama + whisper + diffusion +
transcribe, "sdcpp" enabled on the diffusion model):

    pytest tests/test_media_sdcpp.py \
        --serve-config tests/configs/tiny-media-sdcpp.json

Generations are kept small (128x128, 8 steps) so the tests stay fast on
the CPU.
"""

import base64
import json
import struct
import time
import zlib

import pytest

W = 128
STEPS = 8
PROMPT = "a red square on a white background"


def need_sdcpp(server):
    if not server.cfg.get("sdcpp"):
        pytest.skip("sdcpp is not enabled in the config "
                    "(tests/configs/tiny-media-sdcpp.json)")


def png(width=64, height=64, rgb=(200, 30, 30)):
    """A solid-colour PNG with no image library available."""
    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def b64png(width=64, height=64, rgb=(200, 30, 30)):
    return base64.b64encode(png(width, height, rgb)).decode()


def submit(server, **params):
    payload = {"prompt": PROMPT, "width": W, "height": W,
               "sample_params": {"sample_steps": STEPS}}
    payload.update(params)
    s, data = server.client.post("/sdcpp/v1/img_gen", payload)
    assert s == 202, f"status={s} body={str(data)[:300]}"
    j = json.loads(data)
    assert j.get("kind") == "img_gen" and j.get("status") == "queued", \
        f"bad 202 body: {j}"
    assert j.get("poll_url", "").startswith("/sdcpp/v1/jobs/"), j
    return j


def poll(server, poll_url, timeout=600):
    deadline = time.time() + timeout
    job = {}
    while time.time() < deadline:
        s, job = server.client.get(poll_url, timeout=30)
        assert s == 200, f"status={s} body={str(job)[:300]}"
        if job["status"] in ("completed", "failed", "cancelled"):
            return job
        time.sleep(1)
    pytest.fail(f"job did not finish in {timeout}s: {job}")


def expect_image(job, w, h):
    assert job["status"] == "completed", f"job: {str(job)[:400]}"
    assert job.get("error") is None, f"job: {str(job)[:400]}"
    result = job.get("result") or {}
    assert result.get("output_format") == "png", f"result: {result}"
    images = result.get("images") or []
    assert images, f"no images: {str(job)[:400]}"
    img = base64.b64decode(images[0]["b64_json"])
    assert img[:8] == b"\x89PNG\r\n\x1a\n", f"not a PNG: {img[:8]!r}"
    pw, ph = struct.unpack(">II", img[16:24])
    assert (pw, ph) == (w, h), f"requested {w}x{h}, got {pw}x{ph}"


def test_capabilities(server):
    need_sdcpp(server)
    s, caps = server.client.get("/sdcpp/v1/capabilities")
    assert s == 200, f"status={s} body={str(caps)[:300]}"
    assert caps["model"]["stem"].startswith("stable-diffusion-v1-5"), caps
    assert "img_gen" in caps["supported_modes"], caps
    assert caps["defaults"]["width"] == 512
    assert isinstance(caps["samplers"], list) and caps["samplers"], caps
    assert isinstance(caps["schedulers"], list) and caps["schedulers"], caps
    names = [u["name"] for u in caps["upscalers"]]
    for expected in ("None", "Latent", "esrgan-4x"):
        assert expected in names, f"missing {expected!r} in {names}"


def test_txt2img_job(server):
    need_sdcpp(server)
    job = submit(server)
    expect_image(poll(server, job["poll_url"]), W, W)


@pytest.mark.xfail(reason="the pinned engine's init-image (VAE encode) path "
                          "overflows the APE's 80 KB guest-thread stack and "
                          "crashes the child (core 26483; pass-through-plan "
                          "14.10): the serve layer answers 502 "
                          "media_unavailable after its respawn. Non-strict: "
                          "an xpass means the engine now copes and the limit "
                          "can be lifted.", strict=False)
def test_img2img(server):
    need_sdcpp(server)
    job = submit(server,
                 init_image=b64png(W, W, (200, 30, 30)),
                 strength=0.5,
                 prompt="make it blue")
    expect_image(poll(server, job["poll_url"]), W, W)


def test_hires_upscale(server):
    need_sdcpp(server)
    job = submit(server, width=64, height=64,
                 hires={"enabled": True, "upscaler": "Latent", "scale": 2})
    expect_image(poll(server, job["poll_url"]), 128, 128)


def test_cancel_queued_job(server):
    need_sdcpp(server)
    c = server.client
    # Occupy the pump with a running job so the second one stays queued
    # and the cancel deterministically reaches the queued branch.
    runner = submit(server)
    victim = submit(server)
    s, body = c.post(victim["poll_url"] + "/cancel", {})
    body = json.loads(body) if isinstance(body, bytes) else body
    if s == 200:
        # still queued: the cancel sticks and the run no-ops
        assert body["status"] == "cancelled", body
        assert body["error"]["code"] == "cancelled", body
        final = poll(server, victim["poll_url"], timeout=60)
        assert final["status"] == "cancelled", final
    else:
        # already generating: the cancel is refused, the job completes
        assert s == 409, f"status={s} body={str(body)[:300]}"
        final = poll(server, victim["poll_url"])
        assert final["status"] == "completed", final
    # the queue keeps working after a cancel
    job2 = submit(server)
    expect_image(poll(server, job2["poll_url"]), W, W)


def test_error_matrix(server):
    need_sdcpp(server)
    c = server.client
    # Empty body.
    s, body = c.raw("/sdcpp/v1/img_gen", b"")
    assert s == 400, f"expected 400 for empty body, got {s}"
    assert b"empty body" in body, body
    # Malformed JSON.
    s, _ = c.raw("/sdcpp/v1/img_gen", b'{"prompt": ')
    assert s == 400, f"expected 400 for bad JSON, got {s}"
    # Missing prompt.
    s, _ = c.post("/sdcpp/v1/img_gen", {})
    assert s == 400, f"expected 400 for missing prompt, got {s}"
    # Out-of-range width.
    s, _ = c.post("/sdcpp/v1/img_gen", {"prompt": PROMPT, "width": 5000})
    assert s == 400, f"expected 400 for huge width, got {s}"
    # Unsupported output format.
    s, _ = c.post("/sdcpp/v1/img_gen",
                  {"prompt": PROMPT, "output_format": "jpeg"})
    assert s == 400, f"expected 400 for jpeg, got {s}"
    # mask_image is rejected (the engine's mask path is broken in this
    # APE; see the note in serve_media.cpp).
    s, body = c.post("/sdcpp/v1/img_gen",
                     {"prompt": PROMPT,
                      "mask_image": b64png(64, 64, (255, 255, 255))})
    assert s == 400, f"expected 400 for mask_image, got {s}"
    assert "not supported" in str(body), body
    # Unknown job.
    s, body = c.get("/sdcpp/v1/jobs/job_00000000_deadbeef")
    assert s == 404, f"expected 404 for unknown job, got {s}"
    assert "not found" in str(body), body
    # Cancelling an unknown job.
    s, _ = c.post("/sdcpp/v1/jobs/job_00000000_deadbeef/cancel", {})
    assert s == 404, f"expected 404 for unknown job, got {s}"
