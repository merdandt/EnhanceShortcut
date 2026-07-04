#!/usr/bin/env python3
"""Voice Enhance Service - FastAPI Application

Wraps resemble-enhance (denoise + enhance) behind a small API:

  POST /enhance   multipart file upload OR {"r2_key": ...} JSON body
                  -> uploads result to R2 and returns {"url", "key"}
                  -> ?inline=true streams the enhanced WAV back instead
  GET  /health    Cloud Run health check + GPU/model status
"""

import asyncio
import hmac
import logging
import os
import re
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response

from voice_enhance import engine, r2

# Import shared config if available
try:
    import common.config as config
except ImportError:
    config = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration (root .env + voice_enhance/.env are loaded in __init__.py)
# =============================================================================

API_KEY = os.environ.get("API_KEY", "").strip()

MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", "100"))
MAX_UPLOAD_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)

ALLOWED_EXTENSIONS = (
    tuple(config.ALLOWED_AUDIO_EXTENSIONS) if config else (".wav", ".mp3", ".m4a", ".ogg")
)

DEFAULT_NFE = int(os.environ.get("DEFAULT_NFE", "64"))
DEFAULT_SOLVER = os.environ.get("DEFAULT_SOLVER", "midpoint")
DEFAULT_LAMBD = float(os.environ.get("DEFAULT_LAMBD", "0.9"))
DEFAULT_TAU = float(os.environ.get("DEFAULT_TAU", "0.5"))

# Fail hard at startup if the model cannot load (set in the Docker image so a
# broken GPU deploy is visible immediately; leave unset for local dev).
REQUIRE_MODEL = os.environ.get("REQUIRE_MODEL", "").strip().lower() in ("1", "true", "yes")

# One enhancement at a time: the service is deployed with --concurrency=1,
# this semaphore keeps local dev honest too.
_gpu_semaphore = asyncio.Semaphore(1)


# =============================================================================
# FastAPI Application
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not API_KEY:
        logger.error("API_KEY is not set — all /enhance requests will be rejected (503)")
    loaded = await asyncio.to_thread(engine.load_model)
    if not loaded and REQUIRE_MODEL:
        raise RuntimeError(f"REQUIRE_MODEL is set and model load failed: {engine.LOAD_ERROR}")
    yield


app = FastAPI(
    title="Voice Enhance",
    description="Denoise and enhance phone-recorded voiceover audio to studio quality using resemble-enhance",
    version="1.0.0",
    lifespan=lifespan,
)


# =============================================================================
# Auth (application-layer; Cloud Run itself is --allow-unauthenticated)
# =============================================================================

async def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    """Constant-time API key check. Missing server key never allows access."""
    if not API_KEY:
        raise HTTPException(status_code=503, detail="Service misconfigured: API key not set")
    if x_api_key is None or not hmac.compare_digest(x_api_key.encode(), API_KEY.encode()):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# =============================================================================
# Health Check Endpoint (required for Cloud Run)
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint for Cloud Run; also reports GPU/model status."""
    return JSONResponse({
        "status": "healthy",
        "service": "voice_enhance",
        "version": "1.0.0",
        "cuda_available": engine.cuda_available(),
        "device": engine.DEVICE,
        "model_loaded": engine.MODEL_LOADED,
        "model_error": engine.LOAD_ERROR,
    })


# =============================================================================
# API Endpoints
# =============================================================================

def _validate_extension(name: str | None) -> str:
    ext = Path(name or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension {ext or '(none)'}; allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )
    return ext


def _safe_stem(name: str | None) -> str:
    stem = Path(name or "audio").stem
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem)[:80] or "audio"


async def _receive_upload(upload, dest: Path) -> int:
    """Stream a multipart upload to disk, enforcing MAX_UPLOAD_BYTES."""
    size = 0
    with dest.open("wb") as f:
        while chunk := await upload.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds the {MAX_UPLOAD_MB:g} MB limit",
                )
            f.write(chunk)
    if size == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    return size


@app.post("/enhance", dependencies=[Depends(require_api_key)])
async def enhance_audio(
    request: Request,
    inline: bool = Query(default=False, description="Stream enhanced WAV back instead of uploading to R2"),
    denoise_only: bool = Query(default=False, description="Run only the denoiser (skip CFM enhancement)"),
    nfe: int = Query(default=DEFAULT_NFE, ge=1, le=128, description="CFM number of function evaluations"),
    solver: str = Query(default=DEFAULT_SOLVER, pattern="^(midpoint|rk4|euler)$"),
    lambd: float = Query(default=DEFAULT_LAMBD, ge=0.0, le=1.0, description="Denoise strength before enhancement"),
    tau: float = Query(default=DEFAULT_TAU, ge=0.0, le=1.0, description="CFM prior temperature"),
):
    if not engine.MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail=f"Enhancement model is not loaded: {engine.LOAD_ERROR or 'still starting'}",
        )

    # Cheap early rejection on declared size (multipart adds a little overhead).
    content_length = request.headers.get("content-length")
    if content_length and content_length.isdigit():
        if int(content_length) > MAX_UPLOAD_BYTES + 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"Request exceeds the {MAX_UPLOAD_MB:g} MB limit")

    started = time.monotonic()
    content_type = request.headers.get("content-type", "")

    with tempfile.TemporaryDirectory(prefix="voice_enhance_") as tmp:
        tmpdir = Path(tmp)
        source_label = ""
        source_name = ""

        if content_type.startswith("multipart/form-data"):
            form = await request.form()
            upload = form.get("file")
            if upload is None or isinstance(upload, str):
                raise HTTPException(status_code=400, detail="multipart/form-data must include a 'file' field")
            ext = _validate_extension(upload.filename)
            src = tmpdir / f"input{ext}"
            size = await _receive_upload(upload, src)
            source_label, source_name = "upload", upload.filename or ""
        elif content_type.split(";")[0].strip() in ("application/json", ""):
            try:
                body = await request.json()
            except Exception:
                raise HTTPException(status_code=400, detail="Body must be valid JSON with an 'r2_key' field")
            r2_key = body.get("r2_key") if isinstance(body, dict) else None
            if not isinstance(r2_key, str) or not r2_key.strip():
                raise HTTPException(status_code=400, detail="JSON body must include a non-empty 'r2_key' string")
            r2_key = r2_key.strip()
            ext = _validate_extension(r2_key)
            if not r2.is_configured():
                raise HTTPException(status_code=503, detail="R2 storage is not configured on this server")
            try:
                size = await asyncio.to_thread(r2.head_size, r2_key)
            except r2.R2ObjectNotFound:
                raise HTTPException(status_code=404, detail=f"Object not found in R2: {r2_key}")
            if size > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail=f"R2 object exceeds the {MAX_UPLOAD_MB:g} MB limit")
            src = tmpdir / f"input{ext}"
            await asyncio.to_thread(r2.download, r2_key, src)
            source_label, source_name = "r2", r2_key
        else:
            raise HTTPException(
                status_code=415,
                detail="Send multipart/form-data with a 'file' field, or application/json with 'r2_key'",
            )

        out_path = tmpdir / "enhanced.wav"
        async with _gpu_semaphore:
            try:
                stats = await asyncio.to_thread(
                    engine.process_file,
                    src,
                    out_path,
                    denoise_only=denoise_only,
                    nfe=nfe,
                    solver=solver,
                    lambd=lambd,
                    tau=tau,
                )
            except engine.AudioDecodeError as exc:
                raise HTTPException(status_code=422, detail=f"Could not decode audio: {exc}")

        total_ms = int((time.monotonic() - started) * 1000)
        logger.info(
            "enhanced source=%s name=%s bytes=%d in_s=%.1f denoise_only=%s nfe=%d solver=%s "
            "lambd=%.2f tau=%.2f device=%s decode_ms=%d enhance_ms=%d total_ms=%d",
            source_label, source_name, size, stats["input_seconds"], denoise_only, nfe, solver,
            lambd, tau, stats["device"], stats["decode_ms"], stats["enhance_ms"], total_ms,
        )

        if inline:
            wav_bytes = out_path.read_bytes()
            return Response(
                content=wav_bytes,
                media_type="audio/wav",
                headers={
                    "Content-Disposition": f'attachment; filename="enhanced_{_safe_stem(source_name)}.wav"',
                    "X-Processing-Ms": str(total_ms),
                },
            )

        if not r2.is_configured():
            raise HTTPException(
                status_code=503,
                detail="R2 storage is not configured on this server; use ?inline=true or set R2_* env vars",
            )
        try:
            key = await asyncio.to_thread(r2.upload_enhanced, out_path)
        except Exception:
            logger.exception("R2 upload failed")
            raise HTTPException(status_code=502, detail="Failed to upload enhanced audio to R2")

        return JSONResponse(
            {"url": r2.public_url(key), "key": key},
            headers={"X-Processing-Ms": str(total_ms)},
        )


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Voice Enhance is running"}


# =============================================================================
# Local Development Runner
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    default_port = config.VOICE_ENHANCE_PORT if config else 8081
    port = int(os.environ.get("PORT", default_port))
    logger.info(f"Starting Voice Enhance on port {port}")
    uvicorn.run(
        "voice_enhance.__main__:app",
        host="127.0.0.1",
        port=port,
        # reload disabled (deviation from the service template): every reload
        # would re-load the ~700 MB enhancement model.
        reload=False,
    )
