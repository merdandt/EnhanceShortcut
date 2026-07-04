"""Voice Enhance Service"""

from pathlib import Path

from dotenv import load_dotenv

_SERVICE_DIR = Path(__file__).resolve().parent

# Real environment variables always win (override=False). Among files, the
# service-specific .env takes precedence over the repo-root .env, matching
# the merge order used by deploy_cloud_run.sh.
load_dotenv(_SERVICE_DIR / ".env", override=False)
load_dotenv(_SERVICE_DIR.parent / ".env", override=False)
