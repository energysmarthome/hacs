"""Constants for enSmart képelemzés"""

DOMAIN = "ensmart_kepelemzes"

# Service
SERVICE_ELEMZES = "elemzes"

# Token entity (state value will be used as Bearer token)
TOKEN_ENTITY_ID = "input_text.ensmart_http_api_token"

# Defaults
DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_TEMPERATURE = 0.5
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_TOKENS = 3000
DEFAULT_TARGET_WIDTH = 1280

DEFAULT_SYSTEM_PROMPT = (
    "Analyze the images and give a concise, objective event summary (<255 chars). "
    "Focus on people, pets, and moving objects; track changes across images. "
    "Exclude static details, avoid speculation, and follow user instructions."
)

# Endpoint (uses the same URL that was previously configured for OpenAI in your fork)
ENDPOINT_ENSMART = "https://api.energysmarthome.cloud/ai/openai/kepelemzes"

# Field names (service call)
FIELD_MESSAGE = "message"
FIELD_IMAGE_ENTITY = "image_entity"
FIELD_IMAGE_FILE = "image_file"
FIELD_INCLUDE_FILENAME = "include_filename"
FIELD_TARGET_WIDTH = "target_width"
FIELD_MAX_TOKENS = "max_tokens"
FIELD_MODEL = "model"
FIELD_TEMPERATURE = "temperature"
FIELD_TOP_P = "top_p"
FIELD_SYSTEM_PROMPT = "system_prompt"
