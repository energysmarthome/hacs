"""Conversation support for EnergySmartHome proxy (with device control)."""

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
from homeassistant.helpers.chat_session import async_get_chat_session
from homeassistant.helpers.entity import Entity

from .const import (
    API_URL,
    DEFAULT_TITLE,
    DOMAIN,
    LOGGER,
    MAX_TOOL_ITERATIONS,
    REQUEST_TIMEOUT,
    TOKEN_ENTITY_ID,
    CONF_SYSTEM_PROMPT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities,
) -> None:
    """Set up the conversation entity."""
    async_add_entities([EnergySmartHomeConversationEntity(hass, config_entry)])


def _make_json_safe(value: Any) -> Any:
    """Recursively convert values to JSON-serializable primitives.

    HA tool schemas may contain opaque sentinel objects; convert unknowns to strings,
    and convert raw `object()` instances to None.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if type(value) is object:
        return None

    if isinstance(value, dict):
        return {str(k): _make_json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_make_json_safe(v) for v in value]

    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")

    iso = getattr(value, "isoformat", None)
    if callable(iso):
        try:
            return iso()
        except Exception:  # noqa: BLE001
            pass

    try:
        return str(value)
    except Exception:  # noqa: BLE001
        return repr(value)


def _fallback_tool_parameters(tool_name: str) -> dict[str, Any]:
    """Fallback JSON Schemas for tools when HA/serializer doesn't provide usable parameters.

    This is critical for color/brightness control: if parameters are null, the model won't
    reliably include fields like rgb_color in tool calls.
    """
    if tool_name == "HassLightSet":
        # Home Assistant built-in intent (ServiceIntentHandler) expects slots like:
        # name/area/floor + brightness (0-100) + color (a color name string).
        # It does NOT consume rgb_color/hs_color/color_name directly.
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Target light name (friendly name) as listed in the static context.",
                },
                "area": {
                    "type": "string",
                    "description": "Area name (e.g. Konyha). Prefer name over area when controlling a specific light.",
                },
                "floor": {
                    "type": "string",
                    "description": "Floor name (if used in your Home Assistant).",
                },
                "brightness": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "Brightness as a percentage (0-100).",
                },
                "color": {
                    "type": "string",
                    "description": "Color name (use English CSS-style names like red, green, blue).",
                },
            },
            "required": ["name"],
            "additionalProperties": False,
        }

    if tool_name == "HassSetPosition":
        # Built-in intent expects a numeric position (0-100). 0=closed/down, 100=open/up.
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Target cover/blind name (friendly name) as listed in the static context.",
                },
                "area": {
                    "type": "string",
                    "description": "Area name. Prefer name when controlling a specific cover.",
                },
                "floor": {
                    "type": "string",
                    "description": "Floor name (if used in your Home Assistant).",
                },
                "position": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "Target position percentage (0-100). 0=closed/down, 100=open/up.",
                },
            },
            "required": ["name", "position"],
            "additionalProperties": False,
        }



    # Minimal fallback for other tools: at least allow a name/entity selector.
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Target device/entity name (friendly name)."},
        },
        "required": ["name"],
        "additionalProperties": True,
    }


def _format_tool(tool: llm.Tool, custom_serializer) -> dict[str, Any]:
    """Format HA tool specification as OpenAI-compatible tool schema (chat.completions-style).

    Some HA tool schemas may not serialize cleanly (or may be absent). If we send
    `"parameters": null`, the model will typically avoid passing structured args such
    as `rgb_color`, which breaks light color control. We therefore fall back to a
    minimal, explicit JSON Schema when needed.
    """
    schema: Any = None

    # If parameters are already a dict JSON schema, keep them as-is.
    if isinstance(tool.parameters, dict):
        schema = tool.parameters
    elif tool.parameters is not None:
        try:
            schema = convert(tool.parameters, custom_serializer=custom_serializer)
        except Exception:  # noqa: BLE001
            schema = None

    schema = _make_json_safe(schema)

    # Ensure we always send a dict schema (never null) to the LLM.
    if not isinstance(schema, dict) or not schema:
        schema = _fallback_tool_parameters(tool.name)

    # Some serializers return dicts without a top-level "type".
    if "type" not in schema:
        schema = {"type": "object", **schema}
    # Guardrail: HassLightSet must target a specific device/area; otherwise HA raises
    # "Service handler cannot target all devices". Force the model to always include "name".
    if tool.name == "HassLightSet":
        req = schema.get("required")
        if not isinstance(req, list):
            req = []
        if "name" not in req:
            req.append("name")
        schema["required"] = req

    # Guardrail: HassSetPosition must include a numeric "position" (0-100) and a target "name".
    if tool.name == "HassSetPosition":
        req = schema.get("required")
        if not isinstance(req, list):
            req = []
        for needed in ("name", "position"):
            if needed not in req:
                req.append(needed)
        schema["required"] = req


    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": schema,
        },
    }



def _ensure_tool_speech(
    tool_name: str | None,
    tool_args: dict[str, Any] | None,
    tool_result: dict[str, Any],
) -> dict[str, Any]:
    """Ensure tool_result includes a minimal speech payload to help the LLM stop looping.

    Some HA tool results contain no speech (speech: {}), which can cause the model to retry
    the same tool call repeatedly and/or claim it couldn't do the action. We synthesize
    a short, truthful confirmation based on the tool arguments.
    """
    try:
        speech = tool_result.get("speech")
        if isinstance(speech, dict):
            plain = speech.get("plain")
            if isinstance(plain, dict):
                s = plain.get("speech")
                if isinstance(s, str) and s.strip():
                    return tool_result
    except Exception:  # noqa: BLE001
        pass

    name = ""
    if isinstance(tool_args, dict):
        name = str(tool_args.get("name") or tool_args.get("area") or "").strip()

    msg = "Rendben."
    if tool_name == "HassLightSet" and isinstance(tool_args, dict):
        if tool_args.get("color"):
            msg = f"{name or 'A lámpa'} színe {tool_args['color']}."
        elif tool_args.get("brightness") is not None:
            msg = f"{name or 'A lámpa'} fényereje {tool_args['brightness']}%."
        else:
            msg = f"{name or 'A lámpa'} be lett állítva."
    elif tool_name == "HassSetPosition" and isinstance(tool_args, dict):
        pos = tool_args.get("position")
        if isinstance(pos, (int, float)):
            msg = f"{name or 'A redőny'} pozíciója {int(pos)}%."
        else:
            msg = f"{name or 'A redőny'} pozíciója be lett állítva."
    elif tool_name in ("HassTurnOn", "HassTurnOff") and name:
        msg = f"{name} OK."

    tool_result["speech"] = {"plain": {"speech": msg, "extra_data": None}}
    return tool_result


def _chat_log_to_messages(chat_log: conversation.ChatLog) -> list[dict[str, Any]]:
    """Convert HA chat log content into OpenAI-compatible messages."""
    messages: list[dict[str, Any]] = []

    # Map tool_call_id -> (tool_name, tool_args) for later tool result enrichment
    tool_args_by_id: dict[str, tuple[str | None, dict[str, Any] | None]] = {}
    for it in chat_log.content:
        if getattr(it, "role", None) == "assistant":
            for tc in (getattr(it, "tool_calls", None) or []):
                tc_id = getattr(tc, "id", None)
                if tc_id:
                    tool_args_by_id[tc_id] = (getattr(tc, "tool_name", None), getattr(tc, "tool_args", None))

    for item in chat_log.content:
        role = getattr(item, "role", None)
        content = getattr(item, "content", None)

        # Normal messages
        if role in ("user", "assistant", "system") and content is not None:
            msg: dict[str, Any] = {"role": role, "content": content}

            # Preserve assistant tool calls if present (so model can see what it asked before)
            tool_calls = getattr(item, "tool_calls", None)
            if role == "assistant" and tool_calls:
                formatted: list[dict[str, Any]] = []
                for tc in tool_calls:
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
                                "arguments": json.dumps(
                                    _make_json_safe(tc_args or {}),
                                    ensure_ascii=False,
                                ),
                            },
                        }
                    )
                if formatted:
                    msg["tool_calls"] = formatted

            messages.append(msg)
            continue

        # Tool results: different HA versions represent these differently.
        # We support both role "tool" and "tool_result", and also look for tool_call_id attrs.
        if role in ("tool", "tool_result") or hasattr(item, "tool_call_id"):
            tool_call_id = getattr(item, "tool_call_id", None)
            tool_name = getattr(item, "tool_name", None)
            tool_result = getattr(item, "tool_result", None)
            if tool_call_id:
                mapped_name, mapped_args = tool_args_by_id.get(tool_call_id, (None, None))
                use_name = tool_name or mapped_name
                tr = tool_result
                if isinstance(tr, dict):
                    tr = _ensure_tool_speech(use_name, mapped_args, tr)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": use_name,
                        "content": json.dumps(_make_json_safe(tr), ensure_ascii=False),
                    }
                )
            continue

    return messages


def _extract_text(data: Any) -> str:
    """Extract assistant text from OpenAI-like response JSON."""
    if not isinstance(data, dict):
        return ""

    # Chat Completions style
    choices = data.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        msg = choices[0].get("message")
        if isinstance(msg, dict):
            txt = msg.get("content")
            if isinstance(txt, str):
                return txt or ""

    # Fallback keys
    for key in ("response", "text", "content", "message"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            return val

    return ""



def _rgb_to_basic_color(rgb: Any) -> str | None:
    """Map an RGB triplet to a basic CSS color name.

    HassLightSet (built-in intent) expects a color *name* (slot 'color'), not rgb tuples.
    We keep this intentionally simple and only map common primary/secondary colors.
    """
    if not isinstance(rgb, (list, tuple)) or len(rgb) != 3:
        return None
    try:
        r, g, b = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
    except (ValueError, TypeError):
        return None
    # Exact common colors
    mapping = {
        (255, 0, 0): "red",
        (0, 255, 0): "green",
        (0, 0, 255): "blue",
        (255, 255, 255): "white",
        (0, 0, 0): "black",
        (255, 255, 0): "yellow",
        (0, 255, 255): "cyan",
        (255, 0, 255): "magenta",
        (255, 165, 0): "orange",
        (128, 0, 128): "purple",
    }
    if (r, g, b) in mapping:
        return mapping[(r, g, b)]
    # Heuristic fallback: pick dominant channel
    if r >= g and r >= b:
        return "red"
    if g >= r and g >= b:
        return "green"
    return "blue"



def _normalize_color_name(value: Any) -> str | None:
    """Normalize Hungarian/English color strings to English CSS-style names.

    HassLightSet intent slot 'color' expects a color name string. We map common Hungarian
    names (and their -ra/-re suffix forms) to English.
    """
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    # strip common Hungarian suffixes like "pirosra", "zöldre", "lilára"
    for suf in ("ra", "re"):
        if v.endswith(suf) and len(v) > 2:
            v = v[:-2]
            break
    v = v.strip()

    mapping = {
        "piros": "red",
        "vörös": "red",
        "zöld": "green",
        "kék": "blue",
        "lila": "purple",
        "ibolya": "purple",
        "sárga": "yellow",
        "fehér": "white",
        "fekete": "black",
        "narancs": "orange",
        "rózsaszín": "pink",
        "turkiz": "cyan",
        "türkiz": "cyan",
        "cián": "cyan",
        "cyan": "cyan",
        "magenta": "magenta",
        "pink": "pink",
        "purple": "purple",
        "red": "red",
        "green": "green",
        "blue": "blue",
        "yellow": "yellow",
        "white": "white",
        "black": "black",
        "orange": "orange",
    }
    return mapping.get(v, value.strip())


def _normalize_hass_light_set_args(args: dict[str, Any]) -> dict[str, Any]:
    """Normalize various LLM-produced keys to the built-in HassLightSet intent slots.

    Built-in intent slots: name, area, floor, brightness (0-100), color (name).
    See HA built-in intents docs.
    """
    # brightness_pct -> brightness
    if "brightness" not in args and "brightness_pct" in args:
        try:
            args["brightness"] = int(args["brightness_pct"])
        except (ValueError, TypeError):
            pass
    # color_name -> color
    if "color" not in args and "color_name" in args and isinstance(args["color_name"], str):
        args["color"] = _normalize_color_name(args["color_name"]) or args["color_name"]
    # rgb_color -> color (basic mapping)
    if "color" not in args and "rgb_color" in args:
        color = _rgb_to_basic_color(args["rgb_color"])
        if color:
            args["color"] = color

    # Normalize color value (accept Hungarian names as well)
    if "color" in args:
        norm = _normalize_color_name(args.get("color"))
        if norm:
            args["color"] = norm

    # Clamp brightness to 0..100 if present
    if "brightness" in args:
        try:
            b = int(args["brightness"])
            args["brightness"] = max(0, min(100, b))
        except (ValueError, TypeError):
            args.pop("brightness", None)

    # Drop keys the built-in intent doesn't understand (to avoid "extra keys not allowed")
    for k in ("brightness_pct", "rgb_color", "color_name", "hs_color", "xy_color", "kelvin", "color_temp"):
        args.pop(k, None)

    return args


def _normalize_hass_set_position_args(args: dict[str, Any]) -> dict[str, Any]:
    """Normalize HassSetPosition args to match built-in intent slots.

    Built-in intent expects: name/area/floor + position (0-100 integer).
    We translate common direction strings like "down"/"up" into numeric values.
    """
    pos = args.get("position")

    # Accept numeric as string and percentage strings (e.g., "50%")
    if isinstance(pos, str):
        v = pos.strip().lower()
        if v.endswith("%"):
            v = v[:-1].strip()
        mapping: dict[str, int] = {
            "down": 0,
            "close": 0,
            "closed": 0,
            "lower": 0,
            "le": 0,
            "lent": 0,
            "zár": 0,
            "zárd": 0,
            "zárva": 0,
            "up": 100,
            "open": 100,
            "opened": 100,
            "raise": 100,
            "fel": 100,
            "fent": 100,
            "nyit": 100,
            "nyisd": 100,
            "nyitva": 100,
        }
        if v in mapping:
            args["position"] = mapping[v]
            pos = args["position"]
        else:
            try:
                args["position"] = int(v)
                pos = args["position"]
            except ValueError:
                pass

    if isinstance(pos, (int, float)):
        try:
            p = int(pos)
            args["position"] = max(0, min(100, p))
        except (ValueError, TypeError):
            args.pop("position", None)
    else:
        # Remove unusable position to prevent InvalidSlotInfo errors
        if "position" in args and not isinstance(args.get("position"), int):
            args.pop("position", None)

    return args


def _normalize_hass_media_search_and_play_args(args: dict[str, Any]) -> dict[str, Any]:
    """Normalize arguments for HassMediaSearchAndPlay to avoid InvalidSlotInfo.

    Home Assistant expects the searchable text in the slot named `search_query`.
    Models often send `query` (or `search`) instead. Convert those to `search_query`.
    """
    # Move common synonyms to the expected slot name
    if "search_query" not in args:
        if "query" in args:
            args["search_query"] = args.pop("query")
        elif "search" in args:
            args["search_query"] = args.pop("search")
        elif "q" in args:
            args["search_query"] = args.pop("q")

    # Ensure search_query is a string (best-effort)
    sq = args.get("search_query")
    if isinstance(sq, (list, tuple)):
        args["search_query"] = " ".join(str(x) for x in sq if x is not None).strip()
    elif sq is not None and not isinstance(sq, str):
        args["search_query"] = str(sq)

    # Drop empty search_query to prevent slot validation errors
    if isinstance(args.get("search_query"), str) and not args["search_query"].strip():
        args.pop("search_query", None)

    return args



def _extract_tool_calls(data: Any) -> list[dict[str, Any]]:
    """Extract tool calls from OpenAI-like response JSON.

    Returns list of dicts: {id, name, arguments(dict)}
    """
    out: list[dict[str, Any]] = []
    if not isinstance(data, dict):
        return out

    choices = data.get("choices")
    if not (isinstance(choices, list) and choices and isinstance(choices[0], dict)):
        return out

    msg = choices[0].get("message")
    if not isinstance(msg, dict):
        return out

    tool_calls = msg.get("tool_calls")
    if not isinstance(tool_calls, list):
        return out

    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        tc_id = tc.get("id") or uuid4().hex
        fn = tc.get("function")
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        args_raw = fn.get("arguments", "")
        if not isinstance(name, str) or not name:
            continue

        args: dict[str, Any] = {}
        if isinstance(args_raw, dict):
            args = args_raw
        elif isinstance(args_raw, str) and args_raw.strip():
            try:
                parsed = json.loads(args_raw)
                if isinstance(parsed, dict):
                    args = parsed
            except json.JSONDecodeError:
                args = {}

        # Normalize args for certain built-in intents to match HA slot names
        if name == "HassLightSet" and isinstance(args, dict):
            args = _normalize_hass_light_set_args(args)
        if name == "HassSetPosition" and isinstance(args, dict):
            args = _normalize_hass_set_position_args(args)
        if name == "HassMediaSearchAndPlay" and isinstance(args, dict):
            args = _normalize_hass_media_search_and_play_args(args)

        out.append({"id": tc_id, "name": name, "arguments": args})

    return out


async def _response_to_delta_stream(
    text: str, tool_calls: list[dict[str, Any]]
) -> AsyncGenerator[dict[str, Any], None]:
    """Produce a HA delta stream (dict-based) from response text + tool calls."""
    yield {"role": "assistant"}

    if text:
        yield {"content": text}

    if tool_calls:
        yield {
            "tool_calls": [
                llm.ToolInput(
                    id=tc["id"],
                    tool_name=tc["name"],
                    tool_args=_make_json_safe(tc.get("arguments", {})),
                    external=False,
                )
                for tc in tool_calls
            ]
        }


def _extract_static_context_from_system_messages(messages: list[dict[str, Any]]) -> str:
    """Find and return the 'Static Context:' block from HA-provided system messages.

    HA often embeds this inside a larger system prompt. We only keep the Static Context part
    because that contains the entity/area overview.
    """
    for m in messages:
        if m.get("role") != "system":
            continue
        content = m.get("content")
        if not isinstance(content, str):
            continue

        idx = content.find("Static Context:")
        if idx != -1:
            return content[idx:].strip()

        # sometimes it may start with "Static context:" (case variations)
        idx2 = content.lower().find("static context:")
        if idx2 != -1:
            return content[idx2:].strip()

    return ""


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

        self._attr_unique_id = f"{DOMAIN}_{entry.entry_id}"
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

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        """Process a conversation input (HA 2026.x: chat_log is a context manager)."""
        with (
            async_get_chat_session(self.hass, user_input.conversation_id) as session,
            conversation.async_get_chat_log(self.hass, session, user_input) as chat_log,
        ):
            return await self._async_handle_message(user_input, chat_log)

    def _get_bearer_token(self) -> str:
        """Read the bearer token from input_text entity."""
        state = self.hass.states.get(TOKEN_ENTITY_ID)
        token = state.state.strip() if state and isinstance(state.state, str) else ""
        if not token or token.lower() in ("unknown", "unavailable"):
            raise HomeAssistantError(
                f"Missing bearer token in {TOKEN_ENTITY_ID}. "
                "Set that input_text entity to a valid token."
            )
        return token

    async def _call_proxy_api(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Call the proxy API and return JSON."""
        payload = _make_json_safe(payload)
        token = self._get_bearer_token()
        session = async_get_clientsession(self.hass)
        timeout = ClientTimeout(total=REQUEST_TIMEOUT)

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Request-ID": str(uuid4()),
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
                "conversation_id": str(chat_log.conversation_id),
            }

            # --- Apply generation options ---
            opts = self.entry.options
            payload["temperature"] = float(opts.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE))
            payload["top_p"] = float(opts.get(CONF_TOP_P, DEFAULT_TOP_P))

            # --- Build a single system prompt:
            #     (user-configured prompt) + (Static Context block extracted from HA system messages)
            user_system_prompt = str(opts.get(CONF_SYSTEM_PROMPT, ""))

            static_context = _extract_static_context_from_system_messages(payload["messages"])

            combined_parts: list[str] = []
            if user_system_prompt:
                combined_parts.append(user_system_prompt)
            # even if user prompt is empty, we still want static context if available
            if static_context:
                combined_parts.append(static_context)

            combined_system = "\n\n".join(combined_parts)

            # Remove ALL existing system messages (including HA default prompt)
            payload["messages"] = [m for m in payload["messages"] if m.get("role") != "system"]
            # Always add one system message. If both parts empty, content will be "".
            payload["messages"].insert(0, {"role": "system", "content": combined_system})

            # Tools
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"

            data = await self._call_proxy_api(payload)

            text = _extract_text(data)
            tool_calls = _extract_tool_calls(data)

            _ = [
                _
                async for _ in chat_log.async_add_delta_content_stream(
                    self.entity_id,
                    _response_to_delta_stream(text, tool_calls),
                )
            ]

            if not getattr(chat_log, "unresponded_tool_results", None):
                break

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Process the user input and call the proxy API."""
        try:
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(DOMAIN),
                [llm.LLM_API_ASSIST],
                None,
                user_input.extra_system_prompt,
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        await self._async_handle_chat_log(chat_log)
        return conversation.async_get_result_from_chat_log(user_input, chat_log)