import logging
from homeassistant.core import SupportsResponse
from homeassistant.exceptions import ServiceValidationError

from .const import (
    DOMAIN,
    SERVICE_ELEMZES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TARGET_WIDTH,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_SYSTEM_PROMPT,
    FIELD_MESSAGE,
    FIELD_IMAGE_ENTITY,
    FIELD_IMAGE_FILE,
    FIELD_INCLUDE_FILENAME,
    FIELD_TARGET_WIDTH,
    FIELD_MAX_TOKENS,
    FIELD_MODEL,
    FIELD_TEMPERATURE,
    FIELD_TOP_P,
    FIELD_SYSTEM_PROMPT,
)
from .providers import Request, EnSmartVisionClient
from .media_handlers import MediaProcessor

_LOGGER = logging.getLogger(__name__)

CONF_DEFAULT_MODEL = "default_model"
CONF_TEMPERATURE = "temperature"
CONF_TOP_P = "top_p"
CONF_SYSTEM_PROMPT = "system_prompt"


def _normalize_str_or_list(value):
    """Accept either a string (possibly multiline) or a list/tuple of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        # Services UI will often send a string; treat multiline as multiple paths
        return [ln.strip() for ln in value.splitlines() if ln.strip()]
    if isinstance(value, (list, tuple)):
        out = []
        for v in value:
            if v is None:
                continue
            s = str(v).strip()
            if s:
                out.append(s)
        return out
    # Unknown type
    return [str(value).strip()] if str(value).strip() else []


async def async_setup(hass, config):
    """Set up enSmart képelemzés."""

    async def _handle_elemzes(call):
        # Pick the first config entry (single-provider design)
        entries = hass.config_entries.async_entries(DOMAIN)
        if not entries:
            raise ServiceValidationError(
                "Integration not configured. Add 'enSmart képelemzés' first."
            )
        entry = entries[0]
        entry_conf = entry.data or {}

        message = call.data.get(FIELD_MESSAGE)
        if not message:
            raise ServiceValidationError("Missing 'message'.")

        image_entities = _normalize_str_or_list(call.data.get(FIELD_IMAGE_ENTITY))
        image_files = _normalize_str_or_list(call.data.get(FIELD_IMAGE_FILE))

        include_filename = bool(call.data.get(FIELD_INCLUDE_FILENAME, False))
        target_width = int(call.data.get(FIELD_TARGET_WIDTH, DEFAULT_TARGET_WIDTH))
        max_tokens = int(call.data.get(FIELD_MAX_TOKENS, DEFAULT_MAX_TOKENS))

        model = call.data.get(FIELD_MODEL) or entry_conf.get(CONF_DEFAULT_MODEL) or DEFAULT_MODEL

        temperature = call.data.get(FIELD_TEMPERATURE)
        if temperature is None:
            temperature = entry_conf.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)

        top_p = call.data.get(FIELD_TOP_P)
        if top_p is None:
            top_p = entry_conf.get(CONF_TOP_P, DEFAULT_TOP_P)

        system_prompt = (
            call.data.get(FIELD_SYSTEM_PROMPT)
            or entry_conf.get(CONF_SYSTEM_PROMPT)
            or DEFAULT_SYSTEM_PROMPT
        )

        if not image_entities and not image_files:
            raise ServiceValidationError("Provide 'image_entity' and/or 'image_file'.")

        # Collect frames (base64) via MediaProcessor
        req = Request(hass)
        processor = MediaProcessor(hass, req)
        req = await processor.add_images(
            image_entities=image_entities,
            image_paths=image_files,
            target_width=target_width,
            include_filename=include_filename,
            expose_images=False,
        )

        # Call your endpoint (token is pulled from input_text.ensmart_http_api_token)
        client = EnSmartVisionClient(hass)
        response_text = await client.analyze(
            request=req,
            message=message,
            model=model,
            max_tokens=max_tokens,
            temperature=float(temperature) if temperature is not None else None,
            top_p=float(top_p) if top_p is not None else None,
            system_prompt=system_prompt,
            include_filename=include_filename,
        )

        return {"response_text": response_text}

    hass.services.async_register(
        DOMAIN,
        SERVICE_ELEMZES,
        _handle_elemzes,
        supports_response=SupportsResponse.ONLY,
    )
    return True


async def async_setup_entry(hass, entry):
    """Set up from a config entry."""
    return True


async def async_unload_entry(hass, entry):
    return True
