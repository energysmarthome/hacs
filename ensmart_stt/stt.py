"""EnSmart STT speech-to-text entity."""

from __future__ import annotations

import io
import wave
from collections.abc import AsyncIterable

from aiohttp import ClientError, FormData

from homeassistant.components.stt import (
    AudioBitRates,
    AudioChannels,
    AudioCodecs,
    AudioFormats,
    AudioSampleRates,
    SpeechMetadata,
    SpeechResult,
    SpeechResultState,
    SpeechToTextEntity,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import (
    DEFAULT_API_URL,
    DEFAULT_NAME,
    LOGGER,
    PLACEHOLDER_MODEL,
    TOKEN_ENTITY_ID,
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up STT entity."""
    async_add_entities([EnSmartSttEntity(hass, entry)])


class EnSmartSttEntity(SpeechToTextEntity):
    """STT entity that proxies audio to an OpenAI-compatible endpoint."""

    _attr_name = DEFAULT_NAME

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self.hass = hass
        self._attr_unique_id = entry.entry_id

    @property
    def supported_languages(self) -> list[str]:
        """Return supported languages for Assist UI selection.

        IMPORTANT: returning '*' can cause the Assist UI to disable selection.
        Provide explicit BCP-47 tags.
        """
        return ["hu", "hu-HU", "en", "en-US"]

    @property
    def supported_formats(self) -> list[AudioFormats]:
        return [AudioFormats.WAV]

    @property
    def supported_codecs(self) -> list[AudioCodecs]:
        return [AudioCodecs.PCM]

    @property
    def supported_bit_rates(self) -> list[AudioBitRates]:
        return [
            AudioBitRates.BITRATE_8,
            AudioBitRates.BITRATE_16,
            AudioBitRates.BITRATE_24,
            AudioBitRates.BITRATE_32,
        ]

    @property
    def supported_sample_rates(self) -> list[AudioSampleRates]:
        return [
            AudioSampleRates.SAMPLERATE_8000,
            AudioSampleRates.SAMPLERATE_16000,
            AudioSampleRates.SAMPLERATE_44100,
            AudioSampleRates.SAMPLERATE_48000,
        ]

    @property
    def supported_channels(self) -> list[AudioChannels]:
        return [AudioChannels.CHANNEL_MONO, AudioChannels.CHANNEL_STEREO]

    def _get_bearer_token(self) -> str | None:
        state = self.hass.states.get(TOKEN_ENTITY_ID)
        token = (state.state if state else "") or ""
        token = token.strip()
        return token or None

    async def async_process_audio_stream(
        self,
        metadata: SpeechMetadata,
        stream: AsyncIterable[bytes],
    ) -> SpeechResult:
        """Process audio stream by sending to proxy endpoint."""
        token = self._get_bearer_token()
        if not token:
            LOGGER.error("Missing/empty token in %s", TOKEN_ENTITY_ID)
            return SpeechResult("", SpeechResultState.ERROR)

        raw = bytearray()
        async for chunk in stream:
            raw.extend(chunk)
            if len(raw) > 25 * 1024 * 1024:
                LOGGER.error("Audio stream exceeds 25MB limit")
                return SpeechResult("", SpeechResultState.ERROR)

        if not raw:
            LOGGER.error("No audio data received")
            return SpeechResult("", SpeechResultState.ERROR)

        wav_buf = io.BytesIO()
        try:
            with wave.open(wav_buf, "wb") as wav_file:
                wav_file.setnchannels(int(metadata.channel or 1))
                wav_file.setframerate(int(metadata.sample_rate or 16000))
                wav_file.setsampwidth(2)  # 16-bit PCM
                wav_file.writeframes(bytes(raw))
            wav_buf.seek(0)
        except Exception as err:
            LOGGER.exception("Failed to build WAV: %s", err)
            return SpeechResult("", SpeechResultState.ERROR)

        form = FormData()
        form.add_field("file", wav_buf, filename="audio.wav", content_type="audio/wav")
        form.add_field("model", PLACEHOLDER_MODEL)

        # forward language if present
        if metadata.language:
            form.add_field("language", metadata.language)

        session = async_get_clientsession(self.hass)
        headers = {"Authorization": f"Bearer {token}"}

        try:
            async with session.post(DEFAULT_API_URL, data=form, headers=headers) as resp:
                data = await resp.json(content_type=None)

                if resp.status >= 400:
                    LOGGER.error("STT error %s: %s", resp.status, data)
                    return SpeechResult("", SpeechResultState.ERROR)

                text = (data or {}).get("text", "")
                if not text:
                    LOGGER.error("No 'text' in response: %s", data)
                    return SpeechResult("", SpeechResultState.ERROR)

                return SpeechResult(text, SpeechResultState.SUCCESS)

        except (ClientError, TimeoutError, ValueError) as err:
            LOGGER.error("Request failed: %s", err)
            return SpeechResult("", SpeechResultState.ERROR)
