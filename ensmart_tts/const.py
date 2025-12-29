"""Constants for the enSmart TTS integration."""

DOMAIN = "ensmart_tts"

# Fixed endpoint (do not ask user for it)
API_URL = "https://api.energysmarthome.cloud/ai/openai/tts"

# Bearer token is read live from this Home Assistant entity
TOKEN_ENTITY_ID = "input_text.ensmart_http_api_token"

# Config keys (stored in config entry options)
CONF_MODEL = "model"
CONF_VOICE = "voice"
CONF_SPEED = "speed"
CONF_TIMEOUT = "timeout"
CONF_DEFAULT_LANGUAGE = "default_language"

DEFAULT_NAME = "enSmart TTS"
DEFAULT_MODEL = "gpt-4o-mini-tts"

# If selected, we do not send a `voice` field, letting the server decide.
VOICE_SERVER_DEFAULT = "server_default"
DEFAULT_VOICE = VOICE_SERVER_DEFAULT

DEFAULT_SPEED = 1.0
DEFAULT_TIMEOUT = 15
DEFAULT_LANGUAGE = "hu-HU"

# OpenAI TTS voices (and likely compatible with your endpoint)
SUPPORTED_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
SUPPORTED_VOICES_ALL = [VOICE_SERVER_DEFAULT, *SUPPORTED_VOICES]
