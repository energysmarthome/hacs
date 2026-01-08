"""Config flow for enSmart conversation."""

from __future__ import annotations

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.helpers import selector

from .const import (
    DOMAIN,
    DEFAULT_TITLE,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_SYSTEM_PROMPT,
)


class ConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for the integration."""

    VERSION = 1

    async def async_step_user(self, user_input=None):
        """Handle the initial step."""
        if self._async_current_entries():
            return self.async_abort(reason="already_configured")

        # Create an entry with empty data; token is read elsewhere at runtime.
        return self.async_create_entry(title=DEFAULT_TITLE, data={})

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: config_entries.ConfigEntry):
        """Return the options flow handler."""
        return OptionsFlowHandler()


class OptionsFlowHandler(config_entries.OptionsFlowWithReload):
    """Handle options for the integration (auto-reload on save)."""

    async def async_step_init(self, user_input=None):
        """Manage the options."""
        if user_input is not None:
            # Normalize types
            user_input[CONF_TEMPERATURE] = float(
                user_input.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
            )
            user_input[CONF_TOP_P] = float(user_input.get(CONF_TOP_P, DEFAULT_TOP_P))

            # CRITICAL: always persist the key, even if user leaves it empty
            # so that empty "" can intentionally override HA default system prompt.
            user_input[CONF_SYSTEM_PROMPT] = str(
                user_input.get(CONF_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT)
            )

            return self.async_create_entry(title="", data=user_input)

        opts = self.config_entry.options

        schema = vol.Schema(
            {
                vol.Optional(
                    CONF_TEMPERATURE,
                    default=opts.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0,
                        max=2,
                        step=0.1,
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
                vol.Optional(
                    CONF_TOP_P,
                    default=opts.get(CONF_TOP_P, DEFAULT_TOP_P),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0,
                        max=1,
                        step=0.05,
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
                vol.Optional(
                    CONF_SYSTEM_PROMPT,
                    default=opts.get(CONF_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT),
                ): selector.TextSelector(selector.TextSelectorConfig(multiline=True)),
            }
        )

        return self.async_show_form(step_id="init", data_schema=schema)