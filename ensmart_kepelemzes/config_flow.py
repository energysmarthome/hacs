import voluptuous as vol
from homeassistant import config_entries
from homeassistant.helpers import selector
from homeassistant.helpers.selector import selector as selector_fn

from .const import (
    DOMAIN,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_SYSTEM_PROMPT,
)

CONF_DEFAULT_MODEL = "default_model"
CONF_TEMPERATURE = "temperature"
CONF_TOP_P = "top_p"
CONF_SYSTEM_PROMPT = "system_prompt"


class ConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    VERSION = 1

    async def async_step_user(self, user_input=None):
        if user_input is not None:
            return self.async_create_entry(title="enSmart", data=user_input)

        data_schema = vol.Schema(
            {
                vol.Required(CONF_DEFAULT_MODEL, default=DEFAULT_MODEL): selector_fn(
                    {"text": {"multiline": False}}
                ),
                vol.Required(CONF_TEMPERATURE, default=DEFAULT_TEMPERATURE): selector_fn(
                    {
                        "number": {
                            "min": 0,
                            "max": 1,
                            "step": 0.1,
                            "mode": "slider",
                        }
                    }
                ),
                vol.Required(CONF_TOP_P, default=DEFAULT_TOP_P): selector_fn(
                    {
                        "number": {
                            "min": 0,
                            "max": 1,
                            "step": 0.1,
                            "mode": "slider",
                        }
                    }
                ),
                vol.Optional(CONF_SYSTEM_PROMPT, default=DEFAULT_SYSTEM_PROMPT): selector_fn(
                    {"text": {"multiline": True}}
                ),
            }
        )

        return self.async_show_form(step_id="user", data_schema=data_schema)
