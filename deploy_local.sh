#!/bin/bash
# =============================================================================
# deploy_local.sh - Run microservices locally for development
# =============================================================================
# Usage:
#   ./deploy_local.sh                    # Run all services
#   ./deploy_local.sh voice_enhance     # Run specific services
#
# voice_enhance runs on CPU locally (no GPU expected) — slow but fine for
# basic testing. First run creates a dedicated venv and downloads the
# ~700 MB model on startup.
# =============================================================================

set -e

echo "Starting local development services..."

# Change to project root
cd "$(dirname "$0")"

# Load .env file if exists
if [ -f .env ]; then
  echo "Loading environment variables from .env..."
  set -o allexport
  source .env
  set +o allexport
fi

# =============================================================================
# Validate Required Environment Variables
# =============================================================================

if [ -z "${API_KEY}" ]; then
  echo "WARNING: API_KEY is not set in .env — /enhance will answer 503."
  echo "         Generate one with: openssl rand -hex 32"
fi
if [ -z "${R2_ENDPOINT}" ] || [ -z "${R2_BUCKET}" ]; then
  echo "NOTE: R2_* not fully configured — only ?inline=true responses will work."
fi

# =============================================================================
# Service Ports
# =============================================================================

VOICE_ENHANCE_PORT=8081

# =============================================================================
# voice_enhance venv (heavy ML deps, pinned install order — mirrors
# Dockerfile.voice_enhance but with CPU torch wheels)
# =============================================================================

VENV_DIR=".venv-voice_enhance"
RESEMBLE_COMMIT="8e978149bfe8abab3eb77d965d579a111afdb0ff"

setup_voice_enhance_venv() {
  if [ -f "${VENV_DIR}/.deps_installed" ]; then
    return 0
  fi

  echo "Setting up ${VENV_DIR} (first run: downloads CPU torch, ~15 min)..."
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.10 "${VENV_DIR}"
    PIP=(uv pip install --python "${VENV_DIR}/bin/python")
  else
    python3.10 -m venv "${VENV_DIR}" || python3 -m venv "${VENV_DIR}"
    "${VENV_DIR}/bin/pip" install --upgrade pip
    PIP=("${VENV_DIR}/bin/pip" install)
  fi

  # Same order as Dockerfile.voice_enhance:
  "${PIP[@]}" torch==2.4.1 torchaudio==2.4.1
  "${PIP[@]}" setuptools wheel ninja packaging py-cpuinfo numpy==1.26.4
  "${PIP[@]}" --no-build-isolation deepspeed==0.15.4
  "${PIP[@]}" -r voice_enhance/requirements.txt
  "${PIP[@]}" --no-deps "resemble-enhance @ git+https://github.com/resemble-ai/resemble-enhance.git@${RESEMBLE_COMMIT}"

  touch "${VENV_DIR}/.deps_installed"
  echo "venv ready."
}

# =============================================================================
# Start Services
# =============================================================================

if [ $# -gt 0 ]; then
  RUN_SERVICES="$@"
else
  RUN_SERVICES="voice_enhance"
fi

PIDS=""

cleanup() {
  echo ""
  echo "Stopping all services..."
  if [ -n "$PIDS" ]; then
    kill $PIDS 2>/dev/null || true
  fi
  exit 0
}

trap cleanup SIGINT SIGTERM

for service in ${RUN_SERVICES}; do
  case "${service}" in
    voice_enhance)
      setup_voice_enhance_venv
      echo "Starting voice_enhance on port ${VOICE_ENHANCE_PORT} (CPU fallback expected locally)..."
      PORT=${VOICE_ENHANCE_PORT} "${VENV_DIR}/bin/python" -m voice_enhance &
      PIDS="$PIDS $!"
      ;;
    *)
      echo "Unknown service: ${service}"
      ;;
  esac
done

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "=========================================="
echo "Local services started:"
echo "  voice_enhance: http://localhost:${VOICE_ENHANCE_PORT}  (health: /health)"
echo "=========================================="
echo ""
echo "Press Ctrl+C to stop all services"
echo "Or run: kill $PIDS"

# Wait for all background processes
wait
