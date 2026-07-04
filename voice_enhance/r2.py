"""Cloudflare R2 storage access (S3-compatible API via boto3)."""

import logging
import os
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

_client = None


class R2NotConfigured(Exception):
    """R2 credentials/endpoint are missing from the environment."""


class R2ObjectNotFound(Exception):
    """Requested key does not exist in the bucket."""


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def is_configured() -> bool:
    return all(
        _env(name)
        for name in ("R2_ENDPOINT", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")
    )


def get_client():
    global _client
    if _client is not None:
        return _client
    if not is_configured():
        raise R2NotConfigured(
            "R2 is not configured (need R2_ENDPOINT, R2_ACCESS_KEY_ID, "
            "R2_SECRET_ACCESS_KEY, R2_BUCKET in the environment)"
        )

    import boto3
    from botocore.config import Config

    _client = boto3.client(
        "s3",
        endpoint_url=_env("R2_ENDPOINT"),
        aws_access_key_id=_env("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=_env("R2_SECRET_ACCESS_KEY"),
        region_name="auto",
        config=Config(
            signature_version="s3v4",
            retries={"max_attempts": 3, "mode": "standard"},
            # R2 compatibility: only send the newer flexible checksums when
            # the operation strictly requires them.
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    )
    return _client


def head_size(key: str) -> int:
    """Return object size in bytes; raise R2ObjectNotFound if missing."""
    from botocore.exceptions import ClientError

    try:
        resp = get_client().head_object(Bucket=_env("R2_BUCKET"), Key=key)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey", "NotFound"):
            raise R2ObjectNotFound(key) from exc
        raise
    return int(resp["ContentLength"])


def download(key: str, dest: Path) -> None:
    get_client().download_file(_env("R2_BUCKET"), key, str(dest))


def upload_enhanced(local_path: Path) -> str:
    """Upload an enhanced WAV under enhanced/{uuid}.wav; return the key."""
    key = f"enhanced/{uuid.uuid4().hex}.wav"
    get_client().upload_file(
        str(local_path),
        _env("R2_BUCKET"),
        key,
        ExtraArgs={"ContentType": "audio/wav"},
    )
    return key


def public_url(key: str) -> str:
    base = _env("R2_PUBLIC_BASE_URL").rstrip("/")
    return f"{base}/{key}"
