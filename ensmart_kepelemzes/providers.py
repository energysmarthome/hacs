import logging
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import TOKEN_ENTITY_ID, ENDPOINT_ENSMART, DEFAULT_SYSTEM_PROMPT

_LOGGER = logging.getLogger(__name__)


def _extract_text_from_response(resp: dict) -> str:
    """Be tolerant: accept Chat Completions or Responses-style payloads."""
    # Chat Completions style
    try:
        choices = resp.get("choices")
        if choices and isinstance(choices, list):
            msg = choices[0].get("message") or {}
            content = msg.get("content")
            if isinstance(content, str):
                return content
    except Exception:
        pass

    # Responses style (OpenAI Responses API)
    try:
        out = resp.get("output")
        if isinstance(out, list):
            for item in out:
                if item.get("type") == "message":
                    for c in item.get("content", []):
                        if c.get("type") in ("output_text", "text") and isinstance(c.get("text"), str):
                            return c["text"]
        # Convenience field some clients add
        if isinstance(resp.get("output_text"), str):
            return resp["output_text"]
    except Exception:
        pass

    raise ServiceValidationError("Unexpected API response format (no text found).")


class Request:
    """Collects frames and carries call parameters."""
    def __init__(self, hass):
        self.hass = hass
        self.session = async_get_clientsession(hass)
        self.base64_images: list[str] = []
        self.filenames: list[str] = []

    def add_frame(self, base64_image: str, filename: str = "") -> None:
        self.base64_images.append(base64_image)
        self.filenames.append(filename or "")


class EnSmartVisionClient:
    """OpenAI Chat-Completions compatible client, with token from HA entity."""

    def __init__(self, hass, endpoint: str = ENDPOINT_ENSMART):
        self.hass = hass
        self.session = async_get_clientsession(hass)
        self.endpoint = endpoint

    def _get_bearer_token(self) -> str:
        st = self.hass.states.get(TOKEN_ENTITY_ID)
        token = (st.state if st else "") if st else ""
        token = (token or "").strip()
        if not token or token in ("unknown", "unavailable"):
            raise ServiceValidationError(
                f"Missing token: set {TOKEN_ENTITY_ID} to a valid API token."
            )
        return token

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self._get_bearer_token(),
        }

    def _build_payload(
        self,
        model: str,
        system_prompt: str,
        user_message: str,
        base64_images: list[str],
        filenames: list[str],
        max_tokens: int,
        temperature: float | None,
        top_p: float | None,
        include_filename: bool,
    ) -> dict:
        # Chat Completions compatible payload (what your HA provider expects to parse).
        payload: dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt or DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": []},
            ],
            "max_completion_tokens": max_tokens,
        }

        # Optional sampling params (skip for gpt-5 family which may reject them)
        if model not in ("gpt-5", "gpt-5-mini", "gpt-5-nano"):
            if temperature is not None:
                payload["temperature"] = temperature
            if top_p is not None:
                payload["top_p"] = top_p

        # Images first (each preceded with a tag like "Image 1:" or filename)
        user_content = payload["messages"][1]["content"]
        for idx, (img, fn) in enumerate(zip(base64_images, filenames), start=1):
            tag = fn if (include_filename and fn) else f"Image {idx}"
            user_content.append({"type": "text", "text": f"{tag}:"})
            user_content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
            )

        # User prompt last
        user_content.append({"type": "text", "text": user_message})
        return payload

    async def analyze(
        self,
        *,
        request: Request,
        message: str,
        model: str,
        max_tokens: int,
        temperature: float | None,
        top_p: float | None,
        system_prompt: str,
        include_filename: bool,
    ) -> str:
        if not request.base64_images:
            raise ServiceValidationError("No image input provided.")

        payload = self._build_payload(
            model=model,
            system_prompt=system_prompt,
            user_message=message,
            base64_images=request.base64_images,
            filenames=request.filenames,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            include_filename=include_filename,
        )

        async with self.session.post(
            self.endpoint,
            json=payload,
            headers=self._headers(),
            timeout=60,
        ) as r:
            text = await r.text()
            if r.status >= 400:
                _LOGGER.error("API error %s: %s", r.status, text)
                raise ServiceValidationError(f"API request failed ({r.status}).")
            try:
                data = await r.json()
            except Exception:
                _LOGGER.error("Non-JSON response: %s", text)
                raise ServiceValidationError("API returned non-JSON response.")
        return _extract_text_from_response(data)
