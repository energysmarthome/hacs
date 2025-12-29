from __future__ import annotations

import voluptuous as vol

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import config_validation as cv

from .const import DOMAIN, PLATFORMS, SERVICE_SAY


async def async_setup(hass: HomeAssistant, config: dict) -> bool:
    """Set up the integration (YAML not used, but required entrypoint)."""
    hass.data.setdefault(DOMAIN, {})
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up enSmart TTS from a config entry."""
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN].setdefault("services_registered", False)

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    if not hass.data[DOMAIN]["services_registered"]:
        _register_services(hass)
        hass.data[DOMAIN]["services_registered"] = True

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


def _register_services(hass: HomeAssistant) -> None:
    """Register convenience services for this integration."""

    async def async_handle_say(call: ServiceCall) -> None:
        ent_reg = er.async_get(hass)

        # Our TTS entity lives in the "tts" domain, with platform == our integration DOMAIN.
        tts_entity_id = ent_reg.async_get_entity_id(
            domain="tts",
            platform=DOMAIN,
            unique_id=f"{DOMAIN}_tts",
        )
        if not tts_entity_id:
            raise HomeAssistantError(
                "Nem találom a tts.ensmart_tts entitást. Add hozzá az integrációt a UI-ban és indíts újra."
            )

        message: str = call.data["message"]
        media_player_entity_id = call.data["media_player_entity_id"]
        language = call.data.get("language")
        cache = call.data.get("cache", True)
        options = call.data.get("options") or {}

        data = {
            "message": message,
            "media_player_entity_id": media_player_entity_id,
            "cache": cache,
        }
        if language:
            data["language"] = language
        if options:
            data["options"] = options

        # Wrapper around core tts.speak
        await hass.services.async_call(
            "tts",
            "speak",
            service_data=data,
            target={"entity_id": tts_entity_id},
            blocking=True,
        )

    hass.services.async_register(
        DOMAIN,
        SERVICE_SAY,
        async_handle_say,
        schema=vol.Schema(
            {
                vol.Required("message"): cv.string,
                vol.Required("media_player_entity_id"): cv.entity_ids,
                vol.Optional("language"): cv.string,
                vol.Optional("cache", default=True): cv.boolean,
                vol.Optional("options"): dict,
            }
        ),
    )
