"""EnSmart STT integration."""

from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant

from .const import DOMAIN, LOGGER

PLATFORMS: list[Platform] = [Platform.STT]


async def async_setup(hass: HomeAssistant, config: dict) -> bool:
    """Set up the integration (required entry point)."""
    LOGGER.debug("async_setup called for %s", DOMAIN)
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up from a config entry."""
    LOGGER.debug("Setting up %s (%s)", DOMAIN, entry.entry_id)
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    LOGGER.debug("Unloading %s (%s)", DOMAIN, entry.entry_id)
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
