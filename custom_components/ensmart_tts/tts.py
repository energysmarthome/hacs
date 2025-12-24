"""TTS platform for an OpenAI-compatible /v1/audio/speech endpoint.

This integration exposes a single **tts** entity.

We intentionally override `async_speak` to generate a normal `/local/...` URL
and play that back, instead of relying on the `/api/tts_proxy/...` streaming
route. In your logs we repeatedly saw:

    "TTS engine name is not set"

which prevents playback in certain setups.
"""

from __future__ import annotations

import hashlib
import logging
import pathlib
import secrets
from typing import Any, Mapping

import aiohttp

from homeassistant.components.tts import TextToSpeechEntity, TtsAudioType
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import (
    CONF_API_TOKEN,
    CONF_API_URL,
    CONF_DEFAULT_LANGUAGE,
    CONF_MODEL,
    CONF_SPEED,
    CONF_TIMEOUT,
    CONF_VOICE,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    DEFAULT_SPEED,
    DEFAULT_TIMEOUT,
    DEFAULT_VOICE,
    DOMAIN,
    SUPPORTED_VOICES,
)

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up TTS entity from a config entry."""
    async_add_entities([OpenAICompatTTSEntity(hass, entry)], update_before_add=False)


class OpenAICompatTTSEntity(TextToSpeechEntity):
    """Represent an OpenAI-compatible TTS service."""

    _attr_has_entity_name = True
    _attr_name = None  # Use config entry title

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self.hass = hass
        self.entry = entry
        self._session = async_get_clientsession(hass)

        self._attr_unique_id = entry.entry_id
        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)},
            "name": entry.title,
            "manufacturer": "OpenAI-compatible",
            "model": "TTS HTTP API",
        }

    def _cfg(self) -> dict[str, Any]:
        merged = self.hass.data.get(DOMAIN, {}).get(self.entry.entry_id)
        if isinstance(merged, dict):
            return merged
        return {**self.entry.data, **(self.entry.options or {})}

    # ---- Home Assistant required / expected properties ----

    @property
    def supported_languages(self) -> list[str]:
        # The OpenAI TTS API infers language from text, but HA requires a list.
        return [str(self._cfg().get(CONF_DEFAULT_LANGUAGE, DEFAULT_LANGUAGE))]

    @property
    def default_language(self) -> str:
        return str(self._cfg().get(CONF_DEFAULT_LANGUAGE, DEFAULT_LANGUAGE))

    @property
    def supported_options(self) -> list[str] | None:
        return ["model", "voice", "speed"]

    @property
    def default_options(self) -> Mapping[str, Any] | None:
        cfg = self._cfg()
        return {
            "model": cfg.get(CONF_MODEL, DEFAULT_MODEL),
            "voice": cfg.get(CONF_VOICE, DEFAULT_VOICE),
            "speed": cfg.get(CONF_SPEED, DEFAULT_SPEED),
        }

    def async_get_supported_voices(self, language: str) -> list[str] | None:  # noqa: ARG002
        return SUPPORTED_VOICES

    # ---- Core audio generation ----

    async def async_get_tts_audio(
        self, message: str, language: str, options: dict[str, Any]
    ) -> TtsAudioType:
        """Return (extension, bytes) for the given message."""
        cfg = self._cfg()

        url: str | None = cfg.get(CONF_API_URL)
        token: str | None = cfg.get(CONF_API_TOKEN)
        if not url:
            raise HomeAssistantError("enSmart TTS: hiányzó api_url")
        if token is None:
            # Token optional if user wants no-auth (custom endpoint). If they entered
            # empty string, treat as missing.
            token = ""

        model = str(options.get("model") or cfg.get(CONF_MODEL) or DEFAULT_MODEL)
        voice = str(options.get("voice") or cfg.get(CONF_VOICE) or DEFAULT_VOICE)
        speed = float(options.get("speed") or cfg.get(CONF_SPEED) or DEFAULT_SPEED)
        timeout_s = int(cfg.get(CONF_TIMEOUT) or DEFAULT_TIMEOUT)

        payload = {
            "model": model,
            "input": message,
            "voice": voice,
            "response_format": "mp3",
            "speed": speed,
        }

        headers = {
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
            "Accept-Encoding": "identity",
            "User-Agent": "enSmart-TTS",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"

        try:
            async with self._session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout_s),
            ) as resp:
                if resp.status in (401, 403):
                    raise HomeAssistantError(
                        "enSmart TTS: hibás token / nincs jogosultság"
                    )

                if resp.status >= 400:
                    try:
                        body = await resp.text()
                    except Exception:  # noqa: BLE001
                        body = "<nem olvasható>"
                    _LOGGER.error(
                        "TTS endpoint HTTP %s. URL=%s, body=%s",
                        resp.status,
                        url,
                        body[:500],
                    )
                    raise HomeAssistantError(f"enSmart TTS: HTTP {resp.status}")

                content_type = (resp.headers.get("Content-Type") or "").lower()

                # Some proxies return JSON that contains a byte array (Node-RED style).
                if "application/json" in content_type:
                    data = await resp.json(content_type=None)
                    payload_arr = data.get("payload")
                    if isinstance(payload_arr, list):
                        return ("mp3", bytes(payload_arr))
                    raise HomeAssistantError(
                        "enSmart TTS: JSON válasz, de hiányzik a 'payload' byte tömb"
                    )

                audio = await resp.read()
                if not audio:
                    raise HomeAssistantError("enSmart TTS: üres audio válasz")
                return ("mp3", audio)
        except (aiohttp.ClientError, TimeoutError) as err:
            raise HomeAssistantError(f"enSmart TTS: hálózati hiba: {err}") from err

    # ---- Playback (workaround for /api/tts_proxy engine problems) ----

    async def async_speak(
        self,
        media_player_entity_id: str,
        message: str,
        cache: bool = True,
        language: str | None = None,
        options: dict[str, Any] | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Generate audio, save under /config/www, and play via /local URL."""
        lang = language or self.default_language
        opts = options or {}

        extension, audio = await self.async_get_tts_audio(message, lang, opts)

        www_dir = pathlib.Path(self.hass.config.path("www")) / "ensmart_tts"
        await self.hass.async_add_executor_job(
            lambda: www_dir.mkdir(parents=True, exist_ok=True)
        )

        if cache:
            key_src = (
                f"{message}|{lang}|{opts.get('model')}|{opts.get('voice')}|"
                f"{opts.get('speed')}|{self.entry.entry_id}"
            )
            file_id = hashlib.sha1(key_src.encode("utf-8")).hexdigest()  # noqa: S324
        else:
            file_id = secrets.token_urlsafe(16)

        filename = f"{file_id}.{extension}"
        filepath = www_dir / filename

        if not filepath.exists():
            await self.hass.async_add_executor_job(filepath.write_bytes, audio)

        # Prefer internal_url (your LAN URL), fall back to helper if available.
        base_url = (self.hass.config.internal_url or "").rstrip("/")
        if not base_url:
            try:
                from homeassistant.helpers.network import get_url

                base_url = get_url(self.hass).rstrip("/")
            except Exception:  # noqa: BLE001
                base_url = ""

        media_url = (
            f"{base_url}/local/ensmart_tts/{filename}"
            if base_url
            else f"/local/ensmart_tts/{filename}"
        )

        _LOGGER.debug(
            "Playing TTS: player=%s url=%s bytes=%s",
            media_player_entity_id,
            media_url,
            len(audio),
        )

        await self.hass.services.async_call(
            "media_player",
            "play_media",
            {
                "entity_id": media_player_entity_id,
                "media_content_id": media_url,
                "media_content_type": "music",
            },
            blocking=True,
        )
