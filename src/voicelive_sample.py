"""Utilities and orchestration helpers for Voice Live Realtime tools.
This module provides a minimal, dependency-light sample that demonstrates how to
prepare the payloads required by Azure Voice Live Realtime when working with
tool calls.  The code is structured so that multiple tools can be registered in
the future and the sample uses the ``gpt-realtime`` model family by default.

Even though the unit tests run without the Azure SDK, the helpers are wired so
that the *real* ``azure-ai-voicelive`` types are used whenever they are
available.  This keeps the sample close to production usage – the
``build_azure_response_params`` function materialises ``FunctionTool`` and
``ResponseCreateParams`` instances with the required ``type="function"`` field
to address the regression the customer reported.

The important details that are showcased are:

* Every tool **must** serialise a ``type`` field with the value ``"function"``.
* ``tool_choice`` can be the literal strings ``"auto"``, ``"none"`` or
  ``"required"``.  The helper wraps these in a tiny enum-like class for
  discoverability.
* The payload structure follows the OpenAI-style JSON contract that the Voice
  Live backend expects and the builder keeps the shape identical between the
  pure-Python and SDK-backed versions.

The :func:`build_response_payload` function is the core entry point – it returns
the dictionary that would be sent through the WebSocket connection.  When the
Azure SDK is present, :func:`build_azure_response_params` can convert the
payload into ``ResponseCreateParams`` for direct use.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


LOG_LEVEL = os.getenv("AZURE_AGENTSDK_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


try:  # pragma: no cover - exercised via monkeypatched tests
    from azure.ai.voicelive.models import ToolChoiceLiteral as AzureToolChoiceLiteral
except (ModuleNotFoundError, ImportError):  # pragma: no cover - executed in test env without SDK
    AzureToolChoiceLiteral = None


class ToolChoice(str, Enum):
    """Literal helper matching the Voice Live ``tool_choice`` strings."""

    AUTO = "auto"
    NONE = "none"
    REQUIRED = "required"

    def to_azure(self) -> str:
        """Return the value or Azure literal when available."""

        if AzureToolChoiceLiteral is None:
            return self.value
        # ``ToolChoiceLiteral`` behaves like a string enum in the SDK.  Its
        # members can be accessed either via attribute or value lookup.
        try:
            return getattr(AzureToolChoiceLiteral, self.name).value
        except AttributeError:  # pragma: no cover - defensive guard for SDK drift
            return self.value


@dataclass(frozen=True)
class FunctionToolSpec:
    """Plain-Python representation of a function tool.

    The structure mirrors what the Azure SDK produces and is intentionally kept
    dependency free to make the minimal sample easy to test.
    """

    name: str
    description: str
    parameters: Mapping[str, Any]

    def as_payload(self) -> Dict[str, Any]:
        """Return the JSON payload expected by Voice Live."""

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": dict(self.parameters),
            },
        }


def build_tools_payload(tools: Iterable[FunctionToolSpec]) -> List[Dict[str, Any]]:
    """Convert :class:`FunctionToolSpec` objects into JSON payloads.

    Parameters
    ----------
    tools:
        An iterable of function tool specifications.

    Returns
    -------
    list of dict
        JSON ready payload for Voice Live.
    """

    payload = [tool.as_payload() for tool in tools]
    logger.debug("Converted %d tool specs into payload entries", len(payload))
    if not payload:
        raise ValueError("At least one tool must be provided for the sample.")
    logger.debug("Tool payload validation succeeded")
    return payload


def build_response_payload(
    *,
    instructions: str,
    tool_specs: Iterable[FunctionToolSpec],
    tool_choice: ToolChoice = ToolChoice.REQUIRED,
    modalities: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Assemble the JSON body for ``response.create``.

    The function keeps the payload explicit so that tests can assert the
    presence of the critical ``"type": "function"`` field.
    """

    logger.debug(
        "Building response payload | tool_choice=%s modalities=%s",
        tool_choice,
        list(modalities) if modalities is not None else None,
    )

    payload: MutableMapping[str, Any] = {
        "instructions": instructions,
        "tools": build_tools_payload(tool_specs),
        "tool_choice": tool_choice.to_azure(),
    }
    if modalities is not None:
        payload["modalities"] = list(modalities)
    logger.debug("Response payload built with keys: %s", list(payload.keys()))
    return dict(payload)


def build_decide_state_tool() -> FunctionToolSpec:
    """Return the tool specification used in the minimal sample."""

    return FunctionToolSpec(
        name="decide_state",
        description=(
            "Inspect the conversation and return stage, completed slots, "
            "missing slots, and the next utterance as JSON."
        ),
        parameters={
            "type": "object",
            "properties": {
                "stage": {"type": "string"},
                "filled": {
                    "type": "object",
                    "additionalProperties": {"type": "boolean"},
                },
                "missing": {"type": "array", "items": {"type": "string"}},
                "next_prompt": {"type": "string"},
            },
            "required": ["stage", "missing", "next_prompt"],
        },
    )


def build_azure_response_params(payload: Mapping[str, Any]) -> Any:
    """Convert a payload dict into Azure SDK objects when available.

    The function is intentionally lazy to keep unit tests independent from the
    SDK.  At runtime (when azure-ai-voicelive is installed) it will import the
    required classes and materialise ``FunctionTool`` and ``ResponseCreateParams``
    instances that the websocket client can send.
    """

    logger.debug("Attempting to build Azure SDK response params")

    try:
        from azure.ai.voicelive.models import FunctionTool, ResponseCreateParams, ToolType
    except ModuleNotFoundError as exc:  # pragma: no cover - requires SDK at runtime
        logger.warning("azure-ai-voicelive SDK not available: %s", exc)
        raise RuntimeError(
            "azure-ai-voicelive is required at runtime to build SDK objects"
        ) from exc

    sdk_tools = []
    for tool in payload["tools"]:
        sdk_tools.append(
            FunctionTool(
                type=ToolType.FUNCTION,
                name=tool["function"]["name"],
                description=tool["function"]["description"],
                parameters=tool["function"]["parameters"],
            )
        )
    logger.debug("Created %d Azure FunctionTool instances", len(sdk_tools))
    return ResponseCreateParams(
        instructions=payload["instructions"],
        tools=sdk_tools,
        tool_choice=payload["tool_choice"],
        modalities=payload.get("modalities"),
    )


def build_user_text_message(text: str) -> Any:
    """Return a user message payload compatible with ``conversation.item.create``.

    The helper produces plain dictionaries so that unit tests remain dependency
    free, while still upgrading to the Azure SDK message objects when the
    package is installed at runtime.
    """

    try:  # pragma: no cover - exercised only when SDK is installed
        from azure.ai.voicelive.models import InputTextContentPart, UserMessageItem

        logger.debug("Building UserMessageItem via Azure SDK")
        return UserMessageItem(content=[InputTextContentPart(text=text)])
    except ModuleNotFoundError:  # pragma: no cover - executed inside tests without SDK
        logger.debug("Azure SDK not available; falling back to dict message payload")
        return {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": text,
                }
            ],
        }


class PromptSequenceRunner:
    """High level helper that exercises session prompt updates.

    Parameters
    ----------
    connection:
        Voice Live realtime connection that exposes ``session.update`` and
        ``response.create`` coroutines.
    tool_spec:
        The function tool that must be invoked on each turn.  Defaults to the
        ``decide_state`` specification shipped with the sample.
    tool_choice:
        Literal that determines how the model selects tools.  ``"required"`` is
        used by default so the function call is guaranteed to be attempted on
        the first response.
    response_modalities:
        Optional list of modalities passed to ``response.create``.
    """

    def __init__(
        self,
        connection: Any,
        *,
        tool_spec: Optional[FunctionToolSpec] = None,
        tool_choice: ToolChoice = ToolChoice.REQUIRED,
        response_modalities: Optional[Iterable[str]] = None,
    ) -> None:
        self._connection = connection
        self._tool_spec = tool_spec or build_decide_state_tool()
        self._tool_choice = tool_choice
        self._response_modalities = list(response_modalities) if response_modalities else None
        self._session_defaults: Dict[str, Any] = {}
        logger.info(
            "PromptSequenceRunner initialised | tool_choice=%s response_modalities=%s",
            self._tool_choice,
            self._response_modalities,
        )

    def _response_kwargs(self, instructions: str) -> Mapping[str, Any]:
        payload = build_response_payload(
            instructions=instructions,
            tool_specs=[self._tool_spec],
            tool_choice=self._tool_choice,
            modalities=self._response_modalities,
        )
        logger.debug(
            "Constructed response payload for instructions snippet: %.60s",
            instructions,
        )
        try:
            response_obj = build_azure_response_params(payload)
        except RuntimeError:
            logger.debug("Returning raw payload because SDK objects could not be built")
            response_obj = payload
        else:
            logger.debug("Returning Azure SDK ResponseCreateParams object")
        return {"response": response_obj}

    async def initialise_session(
        self,
        *,
        instructions: str,
        modalities: Optional[Iterable[str]] = None,
        turn_detection: Optional[Mapping[str, Any]] = None,
        input_audio_transcription: Optional[Mapping[str, Any]] = None,
    ) -> None:
        session_payload: Dict[str, Any] = {"instructions": instructions}
        if modalities is not None:
            session_payload["modalities"] = list(modalities)
        if turn_detection is not None:
            session_payload["turn_detection"] = dict(turn_detection)
        if input_audio_transcription is not None:
            session_payload["input_audio_transcription"] = dict(
                input_audio_transcription
            )

        self._session_defaults = {
            key: value
            for key, value in session_payload.items()
            if key != "instructions"
        }
        logger.info(
            "Updating session with instructions='%.40s...' and options=%s",
            instructions,
            list(self._session_defaults.keys()),
        )
        await self._connection.session.update(session=session_payload)

    async def add_user_message(self, text: str) -> None:
        message = build_user_text_message(text)
        logger.info("Adding user message: %.80s", text)
        await self._connection.conversation.item.create(item=message)

    async def cycle_session_prompts(
        self,
        prompts: Iterable[str],
        *,
        response_instructions: Optional[Callable[[str], str]] = None,
    ) -> None:
        """Iterate over session prompts and request a response after each update."""

        if response_instructions is None:
            response_instructions = lambda prompt: (
                "Inspect the latest conversation state and respond by calling the "
                "decide_state function tool. Current session prompt: " + prompt
            )

        for prompt in prompts:
            session_payload = dict(self._session_defaults)
            session_payload["instructions"] = prompt
            logger.info("Cycling session prompt: %.80s", prompt)
            await self._connection.session.update(session=session_payload)
            await self._connection.response.create(
                **self._response_kwargs(response_instructions(prompt))
            )
            call_id, args_json, text_output = await pump_until_done(self._connection)
            print("FUNCTION CALL ID:", call_id)
            print("ARGS JSON:", args_json)
            if text_output:
                print("TEXT OUTPUT:", text_output)


async def pump_until_done(connection: Any) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Drain events from the connection until a terminal signal is received.

    The helper prints incremental deltas so the demo surfaces the model's
    decision making process while returning the collected artefacts for further
    handling by the caller.
    """

    def _maybe_get(event: Any, name: str) -> Any:
        if hasattr(event, name):
            return getattr(event, name)
        if isinstance(event, Mapping):
            return event.get(name)
        return None

    call_args: List[str] = []
    text_chunks: List[str] = []
    call_id: Optional[str] = None

    while True:
        event = await connection.recv()
        event_type = _maybe_get(event, "type")

        if event_type == "response.function_call_arguments.delta":
            call_id = call_id or _maybe_get(event, "call_id")
            chunk = _maybe_get(event, "arguments") or ""
            call_args.append(chunk)
            print("[function_call_arguments.delta]", chunk)
        elif event_type == "response.function_call_arguments.done":
            print("[function_call_arguments.done] call_id=", _maybe_get(event, "call_id"))
            break
        elif event_type == "response.text.delta":
            chunk = _maybe_get(event, "delta") or ""
            text_chunks.append(chunk)
            print("[text.delta]", chunk)
        elif event_type == "response.text.done":
            print("[text.done]")
        elif event_type == "response.done":
            print("[response.done]")
            break
        elif event_type == "error":
            print("ERROR:", event)
            break

    call_json = "".join(call_args) if call_args else None
    text_output = "".join(text_chunks) if text_chunks else None
    return call_id, call_json, text_output


async def demo_prompt_sequence() -> None:
    """Executable demo that cycles prompts if Azure credentials are available."""

    endpoint = os.getenv("AZURE_VOICELIVE_ENDPOINT")
    api_key = os.getenv("AZURE_VOICELIVE_API_KEY")
    model = os.getenv("VOICELIVE_MODEL", "gpt-realtime")

    logger.info("Starting demo prompt sequence")
    if not (endpoint and api_key):
        logger.warning("Azure credentials missing; aborting realtime demo")
        raise RuntimeError(
            "Set AZURE_VOICELIVE_ENDPOINT and AZURE_VOICELIVE_API_KEY to run the demo"
        )

    try:  # pragma: no cover - requires SDK at runtime
        from azure.ai.voicelive.aio import connect
        from azure.core.credentials import AzureKeyCredential
    except ModuleNotFoundError as exc:  # pragma: no cover - requires optional dependency
        logger.error("azure-ai-voicelive dependency missing: %s", exc)
        raise RuntimeError(
            "azure-ai-voicelive must be installed to run the realtime demo"
        ) from exc

    prompts = ["Prompt A", "Prompt B", "Prompt C"]
    logger.debug("Demo prompt sequence initialised with prompts: %s", prompts)

    async with connect(
        credential=AzureKeyCredential(api_key),
        endpoint=endpoint,
        model=model,
    ) as conn:
        logger.info("Connected to Azure VoiceLive endpoint: %s", endpoint)
        runner = PromptSequenceRunner(conn, response_modalities=["text", "audio"])
        await runner.initialise_session(
            instructions="Initial prompt",
            modalities=["text", "audio"],
            turn_detection={"type": "server_vad"},
            input_audio_transcription={"model": "whisper-1"},
        )
        await runner.add_user_message("保険の契約内容を確認したいです。")
        await runner.cycle_session_prompts(prompts)
        logger.info("Demo prompt sequence completed")


def main() -> None:  # pragma: no cover - convenience entry point
    endpoint = os.getenv("AZURE_VOICELIVE_ENDPOINT")
    api_key = os.getenv("AZURE_VOICELIVE_API_KEY")
    model = os.getenv("VOICELIVE_MODEL", "gpt-realtime")

    logger.debug(
        "Main entry | endpoint=%s key_present=%s model=%s",
        endpoint,
        bool(api_key),
        model,
    )

    if endpoint and api_key:
        logger.info("Azure credentials detected; launching realtime demo")
        asyncio.run(demo_prompt_sequence())
        return

    tool = build_decide_state_tool()
    payload = build_response_payload(
        instructions=(
            "直近の会話ログを見て、進行段階(stage)、埋まった項目(filled)、"
            "不足項目(missing)、次に話すべき文(next_prompt)を返答するJSONを返す"
        ),
        tool_specs=[tool],
        modalities=["text", "audio"],
    )
    logger.info("Azure credentials missing; displaying sample payload instead")
    print("Azure VoiceLive credentials were not found. Showing sample payload instead:")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print('Set AZURE_VOICELIVE_ENDPOINT and AZURE_VOICELIVE_API_KEY to run the realtime demo.')


__all__ = [
    "ToolChoice",
    "FunctionToolSpec",
    "build_decide_state_tool",
    "build_response_payload",
    "build_azure_response_params",
    "build_user_text_message",
    "PromptSequenceRunner",
    "pump_until_done",
    "demo_prompt_sequence",
]



if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(str(exc))
