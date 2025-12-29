"""Conversation support for EnergySmartHome OpenAI proxy (with device control)."""

from __future__ import annotations

from collections.abc import AsyncGenerator
import json
from typing import Any, Literal
from uuid import uuid4

from aiohttp import ClientError, ClientTimeout
from voluptuous_serialize import convert

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, llm
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.entity import Entity

from .const import (
    API_URL,
    DEFAULT_TITLE,
    DOMAIN,
    LOGGER,
    MAX_TOOL_ITERATIONS,
    REQUEST_TIMEOUT,
    TOKEN_ENTITY_ID,
)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities,
) -> None:
    """Set up the conversation entity."""
    async_add_entities([EnergySmartHomeConversationEntity(hass, config_entry)])


def _format_tool(
    tool: llm.Tool, custom_serializer
) -> dict[str, Any]:
    """Format HA tool specification as OpenAI-compatible tool schema (chat.completions-style)."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": _make_json_safe(convert(tool.parameters, custom_serializer=custom_serializer)),
        },
    }


def _make_json_safe(value: Any) -> Any:
    """Recursively convert values to JSON-serializable primitives.

    Home Assistant LLM tool schemas may contain sentinel objects (e.g. `object()`)
    which break JSON encoding. We coerce unknown types to strings and drop/None
    for raw `object()` instances.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    # Avoid sending opaque sentinel objects
    if type(value) is object:
        return None

    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            sv = _make_json_safe(v)
            # Skip keys with None sentinel if you prefer: keep it though
            out[str(k)] = sv
        return out

    if isinstance(value, (list, tuple, set)):
        return [_make_json_safe(v) for v in value]

    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")

    # datetimes / dates
    iso = getattr(value, "isoformat", None)
    if callable(iso):
        try:
            return iso()
        except Exception:
            pass

    # Fallback: stringify
    try:
        return str(value)
    except Exception:
        return repr(value)

def _chat_log_to_messages(chat_log: conversation.ChatLog) -> list[dict[str, Any]]:
    """Convert HA chat log content into OpenAI-compatible messages.

    - user/system/assistant messages are passed through
    - assistant tool calls are represented via message.tool_calls
    - tool results are represented via role="tool" messages
    """
    messages: list[dict[str, Any]] = []

    for item in chat_log.content:
        role = getattr(item, "role", None)

        # Tool results (produced by HA LLM Hass API tool execution)
        if role == "tool_result":
            tool_call_id = getattr(item, "tool_call_id", None)
            tool_name = getattr(item, "tool_name", None)
            tool_result = getattr(item, "tool_result", None)
            if not tool_call_id:
                continue
            # OpenAI tool response message
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": json.dumps(tool_result, ensure_ascii=False),
                }
            )
            continue

        content = getattr(item, "content", None)
        if role in ("user", "assistant", "system") and content is not None:
            msg: dict[str, Any] = {"role": role, "content": content}

            # Preserve tool calls from assistant turns (if any)
            tool_calls = getattr(item, "tool_calls", None)
            if role == "assistant" and tool_calls:
                formatted: list[dict[str, Any]] = []
                for tc in tool_calls:
                    # tc is expected to be llm.ToolInput
                    tc_id = getattr(tc, "id", None) or uuid4().hex
                    tc_name = getattr(tc, "tool_name", None)
                    tc_args = getattr(tc, "tool_args", None)
                    if not tc_name:
                        continue
                    formatted.append(
                        {
                            "id": tc_id,
                            "type": "function",
                            "function": {
                                "name": tc_name,
                                "arguments": json.dumps(tc_args or {}, ensure_ascii=False),
                            },
                        }
                    )
                if formatted:
                    msg["tool_calls"] = formatted

            messages.append(msg)

    return messages


def _extract_text(data: Any) -> str:
    """Best-effort extract response text from various JSON response shapes."""
    if data is None:
        return ""

    if isinstance(data, str):
        return data

    # Custom shapes
    for key in ("response", "text", "content", "message"):
        val = data.get(key) if isinstance(data, dict) else None
        if isinstance(val, str) and val.strip():
            return val

    # OpenAI Chat Completions style
    if isinstance(data, dict):
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            msg = choices[0].get("message") if isinstance(choices[0], dict) else None
            if isinstance(msg, dict):
                txt = msg.get("content")
                if isinstance(txt, str) and txt.strip():
                    return txt

    # OpenAI Responses style (output_text)
    if isinstance(data, dict):
        output = data.get("output")
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue
                # message with content parts
                content = item.get("content")
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict):
                            if part.get("type") in ("output_text", "text") and isinstance(
                                part.get("text"), str
                            ):
                                txt = part["text"]
                                if txt.strip():
                                    return txt

    return ""


def _extract_tool_calls(data: Any) -> list[dict[str, Any]]:
    """Extract tool calls from various response shapes.

    Returns list of dicts with keys: id, name, arguments (dict)
    """
    tool_calls: list[dict[str, Any]] = []

    if not isinstance(data, dict):
        return tool_calls

    # Custom top-level
    raw = data.get("tool_calls")
    if isinstance(raw, list):
        for tc in raw:
            _append_tool_call(tool_calls, tc)

    # Chat Completions style
    choices = data.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        msg = choices[0].get("message")
        if isinstance(msg, dict) and isinstance(msg.get("tool_calls"), list):
            for tc in msg["tool_calls"]:
                _append_tool_call(tool_calls, tc)

    # Responses style: output items of type function_call
    output = data.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") in ("function_call", "tool_call"):
                _append_tool_call(tool_calls, item)

    # De-dup by id
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for tc in tool_calls:
        tc_id = tc.get("id") or uuid4().hex
        if tc_id in seen:
            continue
        seen.add(tc_id)
        tc["id"] = tc_id
        deduped.append(tc)
    return deduped


def _append_tool_call(target: list[dict[str, Any]], tc: Any) -> None:
    """Normalize one tool call entry into target list."""
    if not isinstance(tc, dict):
        return

    tc_id = tc.get("id") or tc.get("tool_call_id") or uuid4().hex

    # Chat Completions schema: {"type":"function","function":{"name","arguments"}}
    if isinstance(tc.get("function"), dict):
        name = tc["function"].get("name")
        args_raw = tc["function"].get("arguments")
    else:
        # Responses-like schema: {"type":"function_call","name","arguments"}
        name = tc.get("name") or tc.get("tool_name") or tc.get("function_name")
        args_raw = tc.get("arguments") or tc.get("tool_args")

    if not isinstance(name, str) or not name:
        return

    args: dict[str, Any] = {}
    if isinstance(args_raw, dict):
        args = args_raw
    elif isinstance(args_raw, str) and args_raw.strip():
        try:
            parsed = json.loads(args_raw)
            if isinstance(parsed, dict):
                args = parsed
        except json.JSONDecodeError:
            # Some models return non-JSON; keep empty dict
            args = {}

    target.append({"id": tc_id, "name": name, "arguments": args})


async def _response_to_delta_stream(
    text: str, tool_calls: list[dict[str, Any]]
) -> AsyncGenerator[
    conversation.AssistantContentDeltaDict | conversation.ToolResultContentDeltaDict, None
]:
    """Produce a HA delta stream from a response text + tool calls."""
    # Start / ensure assistant role
    yield {"role": "assistant"}

    if text:
        yield {"content": text}

    if tool_calls:
        yield {
            "tool_calls": [
                llm.ToolInput(
                    id=tc["id"],
                    tool_name=tc["name"],
                    tool_args=tc.get("arguments", {}),
                    external=False,
                )
                for tc in tool_calls
            ]
        }


class EnergySmartHomeConversationEntity(
    conversation.ConversationEntity,
    conversation.AbstractConversationAgent,
    Entity,
):
    """EnergySmartHome conversation agent."""

    _attr_has_entity_name = True
    _attr_name = None
    _attr_supports_streaming = True
    _attr_supported_features = conversation.ConversationEntityFeature.CONTROL

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the entity."""
        self.hass = hass
        self.entry = entry
        self._attr_unique_id = entry.entry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=DEFAULT_TITLE,
            manufacturer="Energy Smart Home",
            model="enSmart conversation",
            entry_type=dr.DeviceEntryType.SERVICE,
        )

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return supported languages."""
        return MATCH_ALL

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    def _get_bearer_token(self) -> str:
        """Read the bearer token from input_text.ensmart_http_api_token."""
        state = self.hass.states.get(TOKEN_ENTITY_ID)
        token = state.state.strip() if state and isinstance(state.state, str) else ""
        if not token or token.lower() in ("unknown", "unavailable"):
            raise HomeAssistantError(
                f"Missing bearer token in {TOKEN_ENTITY_ID}. "
                "Set that input_text entity to a valid token."
            )
        return token

    async def _call_proxy_api(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Call the EnergySmartHome proxy API and return JSON."""
        payload = _make_json_safe(payload)
        token = self._get_bearer_token()
        session = async_get_clientsession(self.hass)
        timeout = ClientTimeout(total=REQUEST_TIMEOUT)

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        try:
            async with session.post(
                API_URL,
                json=payload,
                headers=headers,
                timeout=timeout,
            ) as resp:
                if resp.status == 401:
                    raise HomeAssistantError("Unauthorized (401) - invalid bearer token")
                if resp.status == 403:
                    raise HomeAssistantError("Access denied (403)")
                if resp.status >= 400:
                    body = await resp.text()
                    raise HomeAssistantError(f"Proxy API error {resp.status}: {body[:2000]}")

                return await resp.json(content_type=None)

        except (ClientError, TimeoutError) as err:
            LOGGER.error("Error calling proxy API: %s", err)
            raise HomeAssistantError("Cannot connect to proxy API") from err

    async def _async_handle_chat_log(self, chat_log: conversation.ChatLog) -> None:
        """Generate an answer for the chat log (including tool call loops)."""
        # Build tool specs if HA LLM API tools are available
        tools: list[dict[str, Any]] = []
        if chat_log.llm_api:
            tools = [
                _format_tool(tool, chat_log.llm_api.custom_serializer)
                for tool in chat_log.llm_api.tools
            ]

        for _ in range(MAX_TOOL_ITERATIONS):
            messages = _chat_log_to_messages(chat_log)

            payload: dict[str, Any] = {
                "messages": messages,
                # keep conversation id if server wants it
                "conversation_id": str(chat_log.conversation_id),
            }
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"

            data = await self._call_proxy_api(payload)

            text = _extract_text(data)
            tool_calls = _extract_tool_calls(data)

            # Add assistant response (and any tool calls) into the chat log.
            _ = [
                content
                async for content in chat_log.async_add_delta_content_stream(
                    self.entity_id,
                    _response_to_delta_stream(text, tool_calls),
                )
            ]

            # If HA executed tools, it will have unresponded tool results.
            # Continue the loop so the LLM can respond using those results.
            if not getattr(chat_log, "unresponded_tool_results", None):
                break

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Process the user input and call the proxy API."""
        try:
            # Always enable HA device control tools (no UI configuration).
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(DOMAIN),
                [llm.LLM_API_ASSIST],  # enable Home Assistant API tools
                None,  # no custom prompt from UI
                user_input.extra_system_prompt,
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        await self._async_handle_chat_log(chat_log)

        return conversation.async_get_result_from_chat_log(user_input, chat_log)