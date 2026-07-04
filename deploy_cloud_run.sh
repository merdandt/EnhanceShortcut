#!/bin/bash
# =============================================================================
# deploy_cloud_run.sh - Deploy microservices to Google Cloud Run
# =============================================================================
# Usage:
#   ./deploy_cloud_run.sh                    # Deploy all services
#   ./deploy_cloud_run.sh voice_enhance      # Deploy specific services
#
# Pattern per gcp-cloudrun-microservices skill. Deviation: voice_enhance gets
# a conditional GPU block (NVIDIA L4) in deploy_service(); other services in
# this repo keep deploying CPU-only exactly as the skill template does.
# =============================================================================

set -e

cd "$(dirname "$0")"

# Load root .env if present
if [ -f .env ]; then
  echo "Loading environment variables from .env..."
  set -o allexport
  source .env
  set +o allexport
fi

# =============================================================================
# Configuration
# =============================================================================

export PROJECT_ID="${PROJECT_ID:-}"
export REGION="${REGION:-europe-west4}"
export REPOSITORY_NAME="${REPOSITORY_NAME:-voice-tools}"

# Derived Artifact Registry path
export AR_PREFIX="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY_NAME}"

# Cloud Run GPU (nvidia-l4) regions as of mid-2026 — used only for a warning.
GPU_REGIONS="us-central1 us-east4 europe-west1 europe-west4 asia-southeast1 asia-south1"

# =============================================================================
# Service Definitions
# Format: SERVICE_NAME -> directory of same name, Cloud Run name kebab-case
# =============================================================================

ALL_SERVICES="voice_enhance"

# =============================================================================
# Helper Functions
# =============================================================================

# Parse .env files and construct comma-separated KEY=VALUE pairs for gcloud
get_env_vars_string() {
  local service_dir="$1"
  local pairs=""

  # Include root .env variables first
  if [ -f .env ]; then
    while IFS='=' read -r key raw_val; do
      [[ "$key" =~ ^\s*# ]] && continue
      key=$(echo "$key" | xargs)
      [[ -z "$key" ]] && continue
      raw_val=$(echo "$raw_val" | xargs)
      # Strip surrounding quotes
      if [[ "$raw_val" =~ ^\".*\"$ ]]; then
        val="${raw_val:1:-1}"
      else
        val="$raw_val"
      fi
      pairs+="${key}=${val},"
    done < <(grep -v '^\s*#' .env | grep .)
  fi

  # Service-specific .env (overrides root)
  local env_file="${service_dir}/.env"
  if [ -f "$env_file" ]; then
    while IFS='=' read -r key raw_val; do
      [[ "$key" =~ ^\s*# ]] && continue
      key=$(echo "$key" | xargs)
      [[ -z "$key" ]] && continue
      raw_val=$(echo "$raw_val" | xargs)
      if [[ "$raw_val" =~ ^\".*\"$ ]]; then
        val="${raw_val:1:-1}"
      else
        val="$raw_val"
      fi
      # Remove duplicate keys
      pairs=$(echo "$pairs" | sed -E "s/${key}=[^,]*,?//g")
      pairs+="${key}=${val},"
    done < <(grep -v '^\s*#' "$env_file" | grep .)
  fi

  # Trim trailing comma
  echo "${pairs%,}"
}

deploy_service() {
  local service_name="$1"
  local service_dir="$2"
  local cloud_run_name="$3"
  local buildfile="cloudbuild-${service_name}.yaml"

  echo "=========================================="
  echo "Deploying ${service_name}..."
  echo "=========================================="

  # Build and push image
  echo "Building ${service_name} image..."
  gcloud builds submit . \
    --config="${buildfile}" \
    --substitutions=_REGION=${REGION},_REPO_NAME=${REPOSITORY_NAME} \
    --project=${PROJECT_ID} \
    --quiet

  # Get image digest
  echo "Getting image digest..."
  sleep 3
  local digest=$(gcloud artifacts docker images list "${AR_PREFIX}/${service_name}" \
    --sort-by=~UPDATE_TIME \
    --format="value(DIGEST)" \
    --limit=1 \
    --project=${PROJECT_ID} 2>/dev/null || echo "")

  local image=""
  if [ -z "$digest" ]; then
    echo "Warning: Could not get digest, using :latest tag"
    image="${AR_PREFIX}/${service_name}:latest"
  else
    image="${AR_PREFIX}/${service_name}@${digest}"
  fi

  # Get environment variables (root .env merged with service .env)
  local env_vars=$(get_env_vars_string "$service_dir")
  if [ -n "$ADDITIONAL_ENV_VARS" ]; then
    env_vars="${env_vars},${ADDITIONAL_ENV_VARS}"
  fi

  # ---------------------------------------------------------------------------
  # Per-service resource flags. voice_enhance runs resemble-enhance on an
  # NVIDIA L4 GPU; everything else keeps the skill's CPU defaults.
  # ---------------------------------------------------------------------------
  local resource_flags=()
  if [ "${service_name}" = "voice_enhance" ]; then
    if ! echo " ${GPU_REGIONS} " | grep -q " ${REGION} "; then
      echo "WARNING: REGION=${REGION} is not in the known Cloud Run GPU regions:"
      echo "         ${GPU_REGIONS}"
      echo "         The deploy will likely fail; set REGION in .env accordingly."
    fi
    resource_flags+=(
      --gpu=1
      --gpu-type=nvidia-l4
      # Best-effort single-zone GPU: halves the L4 quota burn; fine for a
      # low-traffic personal tool (no zonal failover guarantee).
      --no-gpu-zonal-redundancy
      # L4 minimums per Cloud Run docs (verified 2026-07): 4 CPU / 16 GiB
      --cpu=4
      --memory=16Gi
      # Keep CPU allocated outside requests (instance-based billing is
      # required for GPU; also keeps the model warm)
      --no-cpu-throttling
      # One request per GPU instance at a time
      --concurrency=1
      # Enhancement of multi-minute clips can take a while, especially cold
      --timeout=300
      # Stay well inside the default L4 quota (3 without zonal redundancy)
      --max-instances=1
      --min-instances=0
      --execution-environment=gen2
    )
  else
    resource_flags+=(
      --memory=1Gi
      --cpu=1
      --timeout=300
      --max-instances=10
      --min-instances=0
    )
  fi

  # Deploy to Cloud Run.
  # --allow-unauthenticated is deliberate: auth is enforced at the app layer
  # via the X-API-Key header (constant-time compare against API_KEY).
  echo "Deploying to Cloud Run..."
  gcloud run deploy "${cloud_run_name}" \
    --image="${image}" \
    --platform=managed \
    --region=${REGION} \
    --allow-unauthenticated \
    --port=8080 \
    "${resource_flags[@]}" \
    --set-env-vars="${env_vars}" \
    --project=${PROJECT_ID}

  # Get and export service URL
  local service_url=$(gcloud run services describe "${cloud_run_name}" \
    --platform=managed \
    --region=${REGION} \
    --format='value(status.url)' \
    --project=${PROJECT_ID})

  echo "${service_name} URL: ${service_url}"

  # Export URL for dependent services
  local url_var_name=$(echo "${service_name}" | tr '[:lower:]' '[:upper:]' | tr '-' '_')_SERVICE_URL
  export "${url_var_name}=${service_url}"

  echo "---"
}

update_service_env() {
  local cloud_run_name="$1"
  local env_var="$2"
  local value="$3"

  echo "Updating ${cloud_run_name} with ${env_var}..."
  gcloud run services update "${cloud_run_name}" \
    --update-env-vars="${env_var}=${value}" \
    --platform=managed \
    --region=${REGION} \
    --project=${PROJECT_ID}
}

# =============================================================================
# Pre-flight Checks
# =============================================================================

echo "=========================================="
echo "GCP Cloud Run Deployment"
echo "=========================================="
echo "Project ID: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Repository: ${REPOSITORY_NAME}"
echo "Artifact Registry: ${AR_PREFIX}"
echo "=========================================="

# Validate configuration
if [ "${PROJECT_ID}" == "your-gcp-project-id" ] || [ -z "${PROJECT_ID}" ]; then
  echo "ERROR: Please set PROJECT_ID in .env"
  exit 1
fi

if [ -z "${API_KEY}" ]; then
  echo "ERROR: API_KEY is not set in .env — the service would reject every request."
  echo "       Generate one with: openssl rand -hex 32"
  exit 1
fi

# Create Artifact Registry repository if needed
echo "Ensuring Artifact Registry repository exists..."
gcloud artifacts repositories create ${REPOSITORY_NAME} \
  --repository-format=docker \
  --location=${REGION} \
  --description="Docker repository for microservices" \
  --project=${PROJECT_ID} 2>/dev/null || echo "Repository already exists"

# =============================================================================
# Deploy Services
# =============================================================================

# Parse command line arguments for specific services
if [ $# -gt 0 ]; then
  DEPLOY_SERVICES="$@"
else
  DEPLOY_SERVICES="${ALL_SERVICES}"
fi

for service in ${DEPLOY_SERVICES}; do
  cloud_run_name="$(echo "${service}" | tr '_' '-')-service"
  deploy_service "${service}" "${service}" "${cloud_run_name}"
done

# =============================================================================
# Update Cross-Service URLs
# =============================================================================
# (single service today — add update_service_env calls here when services
# need each other's URLs)

# =============================================================================
# Grant Public Access
# =============================================================================

echo "Granting public access to services..."
for service in ${DEPLOY_SERVICES}; do
  cloud_run_name="$(echo "${service}" | tr '_' '-')-service"
  gcloud run services add-iam-policy-binding "${cloud_run_name}" \
    --member="allUsers" \
    --role="roles/run.invoker" \
    --platform=managed \
    --region=${REGION} \
    --project=${PROJECT_ID} >/dev/null 2>&1 || echo "Warning: Could not bind ${cloud_run_name}"
done

echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
if [ -n "${VOICE_ENHANCE_SERVICE_URL}" ]; then
  echo "voice_enhance: ${VOICE_ENHANCE_SERVICE_URL}"
  echo ""
  echo "Check GPU attached:"
  echo "  curl -s ${VOICE_ENHANCE_SERVICE_URL}/health   # expect \"cuda_available\": true"
fi
