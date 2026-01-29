"""Constants for EnSmart STT."""

import logging

DOMAIN = "ensmart_stt"
LOGGER = logging.getLogger(__name__)

DEFAULT_API_URL = "https://api.energysmarthome.cloud/ai/openai/stt"
TOKEN_ENTITY_ID = "input_text.ensmart_http_api_token"

# Placeholder required by OpenAI-compatible STT endpoints (multipart form field)
PLACEHOLDER_MODEL = "whisper-1"

DEFAULT_NAME = "EnSmart STT"
