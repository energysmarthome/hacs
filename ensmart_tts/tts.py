from __future__ import annotations

import asyncio
from typing import Any

from homeassistant.components.tts import TextToSpeechEntity, TtsAudioType
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import (
    API_URL,
    DEFAULT_INSTRUCTIONS,
    DEFAULT_MODEL,
    DEFAULT_RESPONSE_FORMAT,
    DEFAULT_SPEED,
    DEFAULT_STREAM_FORMAT,
    DEFAULT_VOICE,
    DOMAIN,
    TOKEN_ENTITY_ID,
)


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback
) -> None:
    async_add_entities([EnSmartTTSEntity(hass)])


class EnSmartTTSEntity(TextToSpeechEntity):
    """enSmart TTS entity."""

    _attr_name = "enSmart TTS"
    _attr_unique_id = f"{DOMAIN}_tts"

    _attr_supported_languages = ["hu", "hu-HU"]
    _attr_default_language = "hu-HU"

    _attr_supported_options = [
        "model",
        "voice",
        "response_format",
        "stream_format",
        "speed",
        "instructions",
    ]
    _attr_default_options = {
        "model": DEFAULT_MODEL,
        "voice": DEFAULT_VOICE,
        "response_format": DEFAULT_RESPONSE_FORMAT,
        "stream_format": DEFAULT_STREAM_FORMAT,
        "speed": DEFAULT_SPEED,
        "instructions": DEFAULT_INSTRUCTIONS,
    }

    def __init__(self, hass: HomeAssistant) -> None:
        self.hass = hass
        self._last_input: str | None = None
        self._last_payload: dict[str, Any] | None = None

    @callback
    def async_get_supported_voices(self, language: str) -> list[str] | None:
        return [DEFAULT_VOICE]

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        payload = dict(self._attr_default_options)
        payload["input"] = self._last_input or ""

        return {
            "model": payload["model"],
            "voice": payload["voice"],
            "input": payload["input"],
            "response_format": payload["response_format"],
            "stream_format": payload["stream_format"],
            "speed": payload["speed"],
            "instructions": payload["instructions"],
            "api_url": API_URL,
            "token_entity_id": TOKEN_ENTITY_ID,
        }

    async def async_get_tts_audio(
        self, message: str, language: str, options: dict[str, Any]
    ) -> TtsAudioType:
        token_state = self.hass.states.get(TOKEN_ENTITY_ID)
        token = (token_state.state if token_state else "").strip()

        if not token or token in ("unknown", "unavailable"):
            raise HomeAssistantError(
                f"Hiányzó vagy érvénytelen token a {TOKEN_ENTITY_ID} entitásban."
            )

        payload: dict[str, Any] = dict(self._attr_default_options)
        payload.update(options or {})
        payload["input"] = message

        self._last_input = message
        self._last_payload = payload
        self.async_write_ha_state()

        session = async_get_clientsession(self.hass)

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "*/*",
        }

        try:
            async with session.post(API_URL, json=payload, headers=headers) as resp:
                resp.raise_for_status()

                # (1) If server returns raw audio bytes:
                ctype = (resp.headers.get("content-type") or "").lower()
                if ctype.startswith("audio/"):
                    audio_bytes = await resp.read()
                    ext = str(payload.get("response_format", "mp3"))
                    return (ext, audio_bytes)

                # (2) JSON with payload: [byte, byte, ...]
                data = await resp.json(content_type=None)
                raw = data.get("payload")

                if isinstance(raw, list) and raw and all(
                    isinstance(x, int) and 0 <= x <= 255 for x in raw
                ):
                    audio_bytes = bytes(raw)
                    ext = str(payload.get("response_format", "mp3"))
                    return (ext, audio_bytes)

                raise HomeAssistantError(
                    f"Ismeretlen válaszformátum az enSmart TTS API-tól. content-type={ctype}, keys={list(data.keys())}"
                )

        except asyncio.TimeoutError as err:
            raise HomeAssistantError("Időtúllépés az enSmart TTS API hívásakor.") from err
