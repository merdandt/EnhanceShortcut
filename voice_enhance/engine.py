"""Model lifecycle and audio processing pipeline for resemble-enhance.

The model is loaded once at service startup (see __main__.py lifespan) and
cached by resemble_enhance's own functools.cache on load_enhancer(run_dir,
device) — the (run_dir, device) pair must therefore be identical between the
startup load and per-request calls.
"""

import asyncio
import logging
import os
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# One enhancement at a time on the GPU. Shared by the HTTP /enhance endpoint
# and the Telegram bot (the service also deploys with --concurrency=1, but
# the bot processes in background tasks, so this is the real serializer).
GPU_SEMAPHORE = asyncio.Semaphore(1)

# Where the enhancer_stage2 checkpoint lives. The Docker image bakes it at
# /models/enhancer_stage2 (MODEL_RUN_DIR set in Dockerfile); locally it is
# downloaded on first startup (~700 MB) into <repo>/models/enhancer_stage2.
_DEFAULT_RUN_DIR = Path(__file__).resolve().parent.parent / "models" / "enhancer_stage2"
MODEL_RUN_DIR = os.environ.get("MODEL_RUN_DIR", "").strip() or str(_DEFAULT_RUN_DIR)

DEVICE: str = "cpu"
MODEL_LOADED: bool = False
LOAD_ERROR: str | None = None

FFMPEG_TIMEOUT_S = 120


class AudioDecodeError(Exception):
    """Input could not be decoded as audio."""


def cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def load_model() -> bool:
    """Load resemble-enhance onto GPU (or CPU fallback). Returns success."""
    global DEVICE, MODEL_LOADED, LOAD_ERROR

    start = time.monotonic()
    try:
        import torch

        if torch.cuda.is_available():
            DEVICE = "cuda"
            logger.info("CUDA available, using GPU: %s", torch.cuda.get_device_name(0))
        else:
            DEVICE = "cpu"
            logger.warning("CUDA not available — falling back to CPU (expected for local dev; slow)")

        from resemble_enhance.enhancer.inference import load_enhancer

        logger.info("Loading resemble-enhance model from %s onto %s ...", MODEL_RUN_DIR, DEVICE)
        load_enhancer(MODEL_RUN_DIR, DEVICE)

        MODEL_LOADED = True
        LOAD_ERROR = None
        logger.info("Model loaded in %.1fs (device=%s)", time.monotonic() - start, DEVICE)
    except Exception as exc:  # noqa: BLE001 — surfaced via /health and REQUIRE_MODEL
        MODEL_LOADED = False
        LOAD_ERROR = f"{type(exc).__name__}: {exc}"
        logger.exception("Failed to load enhancement model")

    return MODEL_LOADED


def _decode_to_wav(src: Path, dst: Path) -> None:
    """Decode any supported input (wav/mp3/m4a/ogg) to mono 16-bit PCM WAV.

    ffmpeg gives one deterministic decode path for every container/codec;
    sample rate is preserved (resemble-enhance resamples internally).
    """
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", str(src),
        "-vn",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        str(dst),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=FFMPEG_TIMEOUT_S)
    except FileNotFoundError as exc:
        raise AudioDecodeError("ffmpeg is not installed on this host") from exc
    except subprocess.TimeoutExpired as exc:
        raise AudioDecodeError(f"audio decode timed out after {FFMPEG_TIMEOUT_S}s") from exc

    if proc.returncode != 0:
        detail = proc.stderr.decode(errors="replace").strip()[-500:]
        raise AudioDecodeError(f"ffmpeg could not decode input: {detail}")


def process_file(
    src: Path,
    dst: Path,
    *,
    denoise_only: bool,
    nfe: int,
    solver: str,
    lambd: float,
    tau: float,
) -> dict:
    """Run enhancement on src audio, write 16-bit WAV to dst, return timings."""
    import soundfile as sf
    import torch

    from resemble_enhance.enhancer.inference import denoise, enhance

    t0 = time.monotonic()
    decoded = src.with_name("decoded.wav")
    _decode_to_wav(src, decoded)

    data, sr = sf.read(decoded, dtype="float32", always_2d=True)
    if data.shape[0] == 0:
        raise AudioDecodeError("decoded audio is empty")
    dwav = torch.from_numpy(data).mean(dim=1)
    input_seconds = dwav.shape[0] / sr
    decode_ms = int((time.monotonic() - t0) * 1000)

    t1 = time.monotonic()
    if denoise_only:
        out, out_sr = denoise(dwav, sr, DEVICE, run_dir=MODEL_RUN_DIR)
    else:
        out, out_sr = enhance(
            dwav, sr, DEVICE,
            nfe=nfe, solver=solver, lambd=lambd, tau=tau,
            run_dir=MODEL_RUN_DIR,
        )
    enhance_ms = int((time.monotonic() - t1) * 1000)

    out_np = out.detach().cpu().numpy().clip(-1.0, 1.0)
    sf.write(dst, out_np, out_sr, subtype="PCM_16")

    return {
        "input_seconds": round(input_seconds, 2),
        "output_sr": out_sr,
        "decode_ms": decode_ms,
        "enhance_ms": enhance_ms,
        "device": DEVICE,
    }
