"""Constants for the OpenAI Compatible TTS integration."""

DOMAIN = "ensmart_tts"

CONF_API_URL = "api_url"
CONF_API_TOKEN = "api_token"
CONF_MODEL = "model"
CONF_VOICE = "voice"
CONF_SPEED = "speed"
CONF_TIMEOUT = "timeout"
CONF_DEFAULT_LANGUAGE = "default_language"

DEFAULT_NAME = "enSmart TTS"
DEFAULT_MODEL = "gpt-4o-mini-tts"
DEFAULT_VOICE = "alloy"
DEFAULT_SPEED = 1.0
DEFAULT_TIMEOUT = 15
DEFAULT_LANGUAGE = "hu-HU"

SUPPORTED_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
