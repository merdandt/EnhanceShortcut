# EnhanceShortcut

Private repo for voice enhancement. A FastAPI microservice (`voice_enhance`)
wraps [resemble-enhance](https://github.com/resemble-ai/resemble-enhance) to
denoise + enhance phone-recorded voiceover to studio-like quality, deployed on
Cloud Run with an NVIDIA L4 GPU. Project layout and deploy scripts follow the
`gcp-cloudrun-microservices` skill conventions.

## API

Auth: every `/enhance` call needs header `X-API-Key: <API_KEY from .env>`.

### `POST /enhance`

Two input modes:

```bash
# a) direct file upload (wav/mp3/m4a/ogg; <=32 MB through Cloud Run)
curl -X POST "$URL/enhance" \
  -H "X-API-Key: $API_KEY" \
  -F "file=@recording.m4a"

# b) object already in the R2 bucket
curl -X POST "$URL/enhance" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"r2_key": "uploads/recording.wav"}'
```

Response: `{"url": "<R2_PUBLIC_BASE_URL>/enhanced/<uuid>.wav", "key": "enhanced/<uuid>.wav"}`
(result is a mono 44.1 kHz 16-bit WAV). Add `?inline=true` to stream the WAV
bytes back directly instead of uploading to R2.

Optional tuning query params: `nfe` (1-128, default 64), `solver`
(midpoint|rk4|euler), `lambd` (denoise strength, default 0.9), `tau` (default
0.5), `denoise_only=true`.

### `GET /health`

Reports `cuda_available` / `model_loaded` so you can verify the GPU actually
attached after a deploy.

### Telegram bot (@enhance_shortcut_bot)

Lives inside the same service as a webhook (`POST /telegram/webhook`, secured
by Telegram's secret token header — registered automatically by
`deploy_cloud_run.sh`). Send the bot a voice message or audio file (≤20 MB,
Telegram's bot limit) and it replies with the cleaned WAV. Only user IDs in
`TELEGRAM_USER_IDS` (root `.env`, comma-separated) are allowed; anyone else
gets a reply with their numeric ID so you can whitelist them, e.g.:

```
TELEGRAM_USER_IDS=123456789,987654321
```

After changing the whitelist, redeploy — or update just the env var (fast):

```bash
gcloud run services update voice-enhance-service --region=europe-west4 \
  --project=<PROJECT_ID> --update-env-vars=TELEGRAM_USER_IDS=123456789
```

## Setup

1. Fill the empty values in `.env` (PROJECT_ID + R2_*). `REGION`,
   `REPOSITORY_NAME` and a generated `API_KEY` are already there.
2. Local run: `make local` → http://localhost:8081 (CPU fallback; first run
   creates `.venv-voice_enhance/` and downloads the ~700 MB model).
3. Deploy: `make deploy` (or `./deploy_cloud_run.sh voice_enhance`).

## GPU deploy notes (voice_enhance only)

`deploy_cloud_run.sh` adds these flags just for `voice_enhance`:
`--gpu=1 --gpu-type=nvidia-l4 --no-gpu-zonal-redundancy --cpu=4 --memory=16Gi
--no-cpu-throttling --concurrency=1 --timeout=300 --max-instances=1`.
Region must be GPU-enabled (`europe-west4` configured; also us-central1,
us-east4, europe-west1, asia-southeast1, asia-south1 as of mid-2026).

The Docker image (`Dockerfile.voice_enhance`) deviates from the skill's
python-slim template: CUDA 12.1 base, torch 2.4.1+cu121 (torch >=2.6 breaks
resemble-enhance checkpoint loading), deepspeed installed with
`--no-build-isolation`, ffmpeg for decoding, and model weights baked at
`/models/enhancer_stage2` so cold starts don't re-download them.
