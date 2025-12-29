"""Config flow for EnergySmartHome OpenAI Conversation."""

from __future__ import annotations

from homeassistant.config_entries import ConfigFlow

from .const import DEFAULT_TITLE, DOMAIN


class ConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for the integration.

    The flow intentionally does not ask for any user input.
    Installing the integration (e.g. via HACS) and adding it from the UI
    is enough. Authentication is done at runtime using the bearer token
    stored in input_text.ensmart_http_api_token.
    """

    VERSION = 1

    async def async_step_user(self, user_input=None):
        """Handle the initial step."""
        if self._async_current_entries():
            return self.async_abort(reason="already_configured")

        # Create an entry with empty data; token is read from the entity at runtime.
        return self.async_create_entry(title=DEFAULT_TITLE, data={})
