from __future__ import annotations

DOMAIN = "ensmart_tts"

API_URL = "https://api.energysmarthome.cloud/ai/openai/tts"
TOKEN_ENTITY_ID = "input_text.ensmart_http_api_token"

DEFAULT_MODEL = "gpt-4o-mini-tts"
DEFAULT_VOICE = "alloy"
DEFAULT_RESPONSE_FORMAT = "mp3"
DEFAULT_STREAM_FORMAT = "sse"
DEFAULT_SPEED = 1.0
DEFAULT_INSTRUCTIONS = "Magyar kiejtés, természetes, barátságos hangnem."

PLATFORMS: list[str] = ["tts"]
SERVICE_SAY = "say"
