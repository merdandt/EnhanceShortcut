"""
Common configuration for EnhanceShortcut microservices.

This module provides shared constants and configuration values used across
multiple services. Import it in any service that needs shared configuration:

    import common.config as config
    url = config.VOICE_ENHANCE_URL
"""

import os

# =============================================================================
# Service Ports (for local development)
# =============================================================================

VOICE_ENHANCE_PORT = 8081

# =============================================================================
# Service URLs
# =============================================================================

VOICE_ENHANCE_URL = os.environ.get(
    "VOICE_ENHANCE_SERVICE_URL",
    f"http://localhost:{VOICE_ENHANCE_PORT}",
)

# =============================================================================
# Shared Configuration
# =============================================================================

# Extensions accepted by audio-processing services
ALLOWED_AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".ogg")
