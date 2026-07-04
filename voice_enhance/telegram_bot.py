"""Telegram bot integration (@enhance_shortcut_bot).

Runs inside the same service/process as the /enhance API: a webhook route on
this FastAPI app, verified via Telegram's secret_token header. Updates are
ACKed immediately and the enhancement runs as a background task (the service
deploys with --no-cpu-throttling, so background work keeps the CPU/GPU), then
the cleaned WAV is sent back as a document.

Only user IDs listed in TELEGRAM_USER_IDS may use the bot; everyone else gets
a reply containing their numeric ID so the owner can whitelist them.
"""

import asyncio
import hmac
import logging
import os
import tempfile
import time
from collections import deque
from pathlib import Path

import httpx
from fastapi import APIRouter, Header, HTTPException, Request

from voice_enhance import engine

logger = logging.getLogger(__name__)

# httpx logs full request URLs at INFO — which would put the bot token into
# service logs. Keep it at WARNING.
logging.getLogger("httpx").setLevel(logging.WARNING)

router = APIRouter()

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
WEBHOOK_SECRET = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "").strip()

# Telegram Bot API caps file downloads for bots at 20 MB.
TELEGRAM_MAX_DOWNLOAD_MB = 20

# Enhancement defaults (same env knobs as the HTTP API)
DEFAULT_NFE = int(os.environ.get("DEFAULT_NFE", "64"))
DEFAULT_SOLVER = os.environ.get("DEFAULT_SOLVER", "midpoint")
DEFAULT_LAMBD = float(os.environ.get("DEFAULT_LAMBD", "0.9"))
DEFAULT_TAU = float(os.environ.get("DEFAULT_TAU", "0.5"))

_API = f"https://api.telegram.org/bot{BOT_TOKEN}"
_FILE_API = f"https://api.telegram.org/file/bot{BOT_TOKEN}"

# Telegram retries undelivered updates (e.g. after a cold start slow-ack);
# remember recent update_ids so a retry never enhances twice.
_seen_update_ids: deque[int] = deque(maxlen=1000)

HELP_TEXT = (
    "Send me a voice message or an audio file (wav/mp3/m4a/ogg, up to "
    f"{TELEGRAM_MAX_DOWNLOAD_MB} MB — Telegram's limit for bots) and I'll "
    "reply with a studio-cleaned WAV.\n\n"
    "Powered by resemble-enhance on an NVIDIA L4."
)


def allowed_user_ids() -> set[int]:
    raw = os.environ.get("TELEGRAM_USER_IDS", "")
    ids = set()
    for part in raw.replace('"', "").split(","):
        part = part.strip()
        if part.isdigit():
            ids.add(int(part))
    return ids


def is_configured() -> bool:
    return bool(BOT_TOKEN and WEBHOOK_SECRET)


async def _tg(method: str, **params) -> dict | None:
    """Call a Bot API method; log and swallow errors (bot must never crash)."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(f"{_API}/{method}", json=params)
        data = resp.json()
        if not data.get("ok"):
            logger.warning("Telegram %s failed: %s", method, data.get("description"))
            return None
        return data.get("result")
    except Exception:
        logger.exception("Telegram %s call errored", method)
        return None


async def _send_text(chat_id: int, text: str, reply_to: int | None = None) -> None:
    params: dict = {"chat_id": chat_id, "text": text}
    if reply_to:
        params["reply_to_message_id"] = reply_to
        params["allow_sending_without_reply"] = True
    await _tg("sendMessage", **params)


async def _send_document(chat_id: int, path: Path, caption: str, reply_to: int | None) -> bool:
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            with path.open("rb") as f:
                resp = await client.post(
                    f"{_API}/sendDocument",
                    data={
                        "chat_id": str(chat_id),
                        "caption": caption,
                        "reply_to_message_id": str(reply_to or ""),
                        "allow_sending_without_reply": "true",
                    },
                    files={"document": (path.name, f, "audio/wav")},
                )
        data = resp.json()
        if not data.get("ok"):
            logger.warning("sendDocument failed: %s", data.get("description"))
            return False
        return True
    except Exception:
        logger.exception("sendDocument errored")
        return False


async def _download_file(file_id: str, dest_dir: Path) -> Path:
    """Resolve file_id via getFile and download it. Raises on failure."""
    info = await _tg("getFile", file_id=file_id)
    if not info or "file_path" not in info:
        raise RuntimeError("getFile returned no file_path")
    file_path = info["file_path"]
    suffix = Path(file_path).suffix or ".bin"
    dest = dest_dir / f"input{suffix}"
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.get(f"{_FILE_API}/{file_path}")
        resp.raise_for_status()
        dest.write_bytes(resp.content)
    return dest


def _extract_audio(msg: dict) -> dict | None:
    """Return the Telegram file descriptor for voice/audio content, if any."""
    if "voice" in msg:
        return msg["voice"]
    if "audio" in msg:
        return msg["audio"]
    doc = msg.get("document")
    if doc and str(doc.get("mime_type", "")).startswith("audio/"):
        return doc
    return None


async def _keep_typing(chat_id: int, stop: asyncio.Event) -> None:
    """Show 'typing…' in the chat for the whole processing window.

    Telegram chat actions expire after ~5s, so they must be re-sent until the
    job finishes.
    """
    while not stop.is_set():
        await _tg("sendChatAction", chat_id=chat_id, action="typing")
        try:
            await asyncio.wait_for(stop.wait(), timeout=4.5)
        except asyncio.TimeoutError:
            pass


async def _process_audio_job(chat_id: int, reply_to: int, audio: dict) -> None:
    """Background task: download -> enhance -> reply with the WAV."""
    started = time.monotonic()
    stop_typing = asyncio.Event()
    typing_task = asyncio.create_task(_keep_typing(chat_id, stop_typing))
    try:
        with tempfile.TemporaryDirectory(prefix="tg_enhance_") as tmp:
            tmpdir = Path(tmp)
            src = await _download_file(audio["file_id"], tmpdir)
            out = tmpdir / "enhanced.wav"
            async with engine.GPU_SEMAPHORE:
                stats = await asyncio.to_thread(
                    engine.process_file,
                    src,
                    out,
                    denoise_only=False,
                    nfe=DEFAULT_NFE,
                    solver=DEFAULT_SOLVER,
                    lambd=DEFAULT_LAMBD,
                    tau=DEFAULT_TAU,
                )
            caption = (
                f"✅ {stats['input_seconds']:.0f}s of audio enhanced in "
                f"{stats['enhance_ms'] / 1000:.1f}s ({stats['device']})"
            )
            stop_typing.set()  # let the upload replace the typing indicator
            sent = await _send_document(chat_id, out, caption, reply_to)
            if not sent:
                await _send_text(chat_id, "Enhanced OK but sending the file failed — try again.", reply_to)
        logger.info(
            "telegram enhanced chat=%s in_s=%.1f enhance_ms=%d total_s=%.1f",
            chat_id, stats["input_seconds"], stats["enhance_ms"], time.monotonic() - started,
        )
    except engine.AudioDecodeError as exc:
        await _send_text(chat_id, f"Couldn't decode that audio ({exc}). Try wav/mp3/m4a/ogg.", reply_to)
    except Exception:
        logger.exception("telegram enhancement job failed")
        await _send_text(chat_id, "Something went wrong enhancing that file. Check the service logs.", reply_to)
    finally:
        stop_typing.set()
        await typing_task


@router.post("/telegram/webhook")
async def telegram_webhook(
    request: Request,
    secret_header: str | None = Header(default=None, alias="X-Telegram-Bot-Api-Secret-Token"),
):
    if not is_configured():
        raise HTTPException(status_code=503, detail="Telegram bot not configured")
    if secret_header is None or not hmac.compare_digest(secret_header.encode(), WEBHOOK_SECRET.encode()):
        raise HTTPException(status_code=401, detail="Bad webhook secret")

    try:
        update = await request.json()
    except Exception:
        return {"ok": True}

    update_id = update.get("update_id")
    if isinstance(update_id, int):
        if update_id in _seen_update_ids:
            return {"ok": True}
        _seen_update_ids.append(update_id)

    msg = update.get("message")
    if not isinstance(msg, dict):
        return {"ok": True}

    from_user = msg.get("from") or {}
    user_id = from_user.get("id")
    chat_id = (msg.get("chat") or {}).get("id")
    message_id = msg.get("message_id")
    if user_id is None or chat_id is None:
        return {"ok": True}

    if user_id not in allowed_user_ids():
        logger.warning("telegram unauthorized user id=%s username=%s", user_id, from_user.get("username"))
        await _send_text(
            chat_id,
            f"⛔ Not authorized.\nYour Telegram ID: {user_id}\n"
            "Ask the owner to add it to TELEGRAM_USER_IDS.",
        )
        return {"ok": True}

    text = (msg.get("text") or "").strip()
    if text.startswith("/start") or text.startswith("/help"):
        await _send_text(chat_id, HELP_TEXT)
        return {"ok": True}

    audio = _extract_audio(msg)
    if audio is None:
        await _send_text(chat_id, "Only audio is supported for now.\n\n" + HELP_TEXT, message_id)
        return {"ok": True}

    if not engine.MODEL_LOADED:
        await _send_text(chat_id, "The enhancement model isn't loaded — try again in a minute.", message_id)
        return {"ok": True}

    file_size = int(audio.get("file_size") or 0)
    if file_size > TELEGRAM_MAX_DOWNLOAD_MB * 1024 * 1024:
        await _send_text(
            chat_id,
            f"That file is {file_size / (1024 * 1024):.0f} MB — Telegram only lets bots "
            f"download up to {TELEGRAM_MAX_DOWNLOAD_MB} MB. Use the HTTP API for big files.",
            message_id,
        )
        return {"ok": True}

    # ACK the webhook now; enhance + reply in the background.
    asyncio.create_task(_process_audio_job(chat_id, message_id, audio))
    return {"ok": True}
