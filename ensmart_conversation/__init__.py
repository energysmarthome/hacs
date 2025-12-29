"""EnergySmartHome OpenAI Conversation integration.

Lightweight custom conversation agent that proxies Home Assistant Assist chat logs
to a custom HTTP API endpoint.
"""

from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import DOMAIN

PLATFORMS: list[str] = ["conversation"]


async def async_setup(hass: HomeAssistant, config: dict) -> bool:
    """Set up the integration (YAML is not supported)."""
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up from a config entry."""
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
