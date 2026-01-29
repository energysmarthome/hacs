"""Config flow for EnSmart STT."""

from __future__ import annotations

from homeassistant import config_entries
from homeassistant.data_entry_flow import FlowResult

from .const import DEFAULT_NAME, DOMAIN


class ConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for EnSmart STT."""

    VERSION = 1

    async def async_step_user(self, user_input=None) -> FlowResult:
        """Create entry without asking anything."""
        if self._async_current_entries():
            return self.async_abort(reason="single_instance_allowed")

        return self.async_create_entry(title=DEFAULT_NAME, data={})
