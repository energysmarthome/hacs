"""Constants for the EnergySmartHome OpenAI Conversation integration."""

from __future__ import annotations

import logging

DOMAIN = "ensmart_conversation"

LOGGER = logging.getLogger(__name__)

API_URL = "https://api.energysmarthome.cloud/ai/openai/conversation"
TOKEN_ENTITY_ID = "input_text.ensmart_http_api_token"

DEFAULT_TITLE = "enSmart conversation"

# Network timeouts (seconds)
REQUEST_TIMEOUT = 60

# To prevent infinite tool loops
MAX_TOOL_ITERATIONS = 10
