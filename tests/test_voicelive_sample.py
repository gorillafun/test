"""Unit tests for the Voice Live minimal sample."""

import sys
from pathlib import Path

import pytest

import importlib
import os
import types

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - defensive path setup
    sys.path.insert(0, str(PROJECT_ROOT))

import src.voicelive_sample as sample

from src.voicelive_sample import FunctionToolSpec, ToolChoice
from src.voicelive_sample import (
    PromptSequenceRunner,
    build_decide_state_tool,
    build_response_payload,
    build_tools_payload,
    build_user_text_message,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_build_tools_payload_includes_type():
    tool = FunctionToolSpec(
        name="example",
        description="Example tool",
        parameters={"type": "object", "properties": {}},
    )

    payload = build_tools_payload([tool])

    assert payload[0]["type"] == "function"
    assert payload[0]["function"]["name"] == "example"


def test_build_tools_payload_requires_tool():
    with pytest.raises(ValueError):
        build_tools_payload([])


def test_build_response_payload_default_modalities_none():
    tool = build_decide_state_tool()

    payload = build_response_payload(
        instructions="Test instructions",
        tool_specs=[tool],
    )

    assert payload["tool_choice"] == ToolChoice.REQUIRED.value
    assert payload["tools"][0]["type"] == "function"
    assert "modalities" not in payload


def test_build_response_payload_with_modalities():
    tool = build_decide_state_tool()
    payload = build_response_payload(
        instructions="Test",
        tool_specs=[tool],
        tool_choice=ToolChoice.AUTO,
        modalities=["text", "audio"],
    )

    assert payload["modalities"] == ["text", "audio"]
    assert payload["tool_choice"] == "auto"


def test_build_azure_response_params_uses_sdk(monkeypatch):
    """Ensure ``build_azure_response_params`` emits SDK objects when available."""

    fake_models = types.SimpleNamespace()

    class FakeToolType:
        FUNCTION = "function"

    class FakeFunctionTool:
        def __init__(self, *, type, name, description, parameters):
            self.type = type
            self.name = name
            self.description = description
            self.parameters = parameters

    class FakeResponseCreateParams:
        def __init__(self, *, instructions, tools, tool_choice, modalities=None):
            self.instructions = instructions
            self.tools = tools
            self.tool_choice = tool_choice
            self.modalities = modalities

    fake_models.FunctionTool = FakeFunctionTool
    fake_models.ResponseCreateParams = FakeResponseCreateParams
    fake_models.ToolType = FakeToolType

    module_name = "azure.ai.voicelive.models"
    monkeypatch.setitem(sys.modules, module_name, fake_models)

    importlib.reload(sample)

    tool = sample.build_decide_state_tool()
    payload = sample.build_response_payload(
        instructions="Test instructions",
        tool_specs=[tool],
        modalities=["text"],
    )

    params = sample.build_azure_response_params(payload)

    assert isinstance(params, FakeResponseCreateParams)
    assert params.tool_choice == ToolChoice.REQUIRED.value
    assert params.tools[0].type == FakeToolType.FUNCTION

    monkeypatch.delitem(sys.modules, module_name, raising=False)
    importlib.reload(sample)


def test_build_user_text_message_without_sdk():
    payload = build_user_text_message("こんにちは")

    assert payload["role"] == "user"
    assert payload["content"][0]["text"] == "こんにちは"


class FakeConnection:
    def __init__(self):
        self.session_updates = []
        self.responses = []
        self.conversation_items = []

        class SessionClient:
            def __init__(self, parent):
                self._parent = parent

            async def update(self, *, session):
                self._parent.session_updates.append(session)

        class ResponseClient:
            def __init__(self, parent):
                self._parent = parent

            async def create(self, **kwargs):
                self._parent.responses.append(kwargs["response"])

        class ConversationItemClient:
            def __init__(self, parent):
                self._parent = parent

            async def create(self, *, item):
                self._parent.conversation_items.append(item)

        class ConversationClient:
            def __init__(self, parent):
                self.item = ConversationItemClient(parent)

        self.session = SessionClient(self)
        self.response = ResponseClient(self)
        self.conversation = ConversationClient(self)


@pytest.mark.anyio("asyncio")
async def test_prompt_sequence_runner_cycles_prompts():
    conn = FakeConnection()
    runner = PromptSequenceRunner(conn, response_modalities=["text"])

    await runner.initialise_session(
        instructions="base",
        modalities=["text"],
        turn_detection={"type": "server_vad"},
        input_audio_transcription={"model": "whisper-1"},
    )
    await runner.add_user_message("hello")
    await runner.cycle_session_prompts(["a", "b", "c"])

    assert [update["instructions"] for update in conn.session_updates] == [
        "base",
        "a",
        "b",
        "c",
    ]
    assert len(conn.responses) == 3
    for response_payload in conn.responses:
        assert response_payload["tools"][0]["type"] == "function"


@pytest.mark.integration
@pytest.mark.anyio("asyncio")
async def test_prompt_sequence_runner_live():
    if not (
        os.getenv("AZURE_VOICELIVE_ENDPOINT")
        and os.getenv("AZURE_VOICELIVE_API_KEY")
        and os.getenv("VOICELIVE_MODEL")
    ):
        pytest.skip("Azure Voice Live credentials not provided")

    try:
        from azure.ai.voicelive.aio import connect
        from azure.core.credentials import AzureKeyCredential
    except ModuleNotFoundError:
        pytest.skip("azure-ai-voicelive is not installed in this environment")

    async with connect(
        credential=AzureKeyCredential(os.environ["AZURE_VOICELIVE_API_KEY"]),
        endpoint=os.environ["AZURE_VOICELIVE_ENDPOINT"],
        model=os.environ["VOICELIVE_MODEL"],
    ) as conn:
        runner = PromptSequenceRunner(conn, response_modalities=["text"])
        await runner.initialise_session(
            instructions="integration test base",
            modalities=["text"],
            turn_detection={"type": "server_vad"},
            input_audio_transcription={"model": "whisper-1"},
        )
        await runner.add_user_message("integration test message")
        await runner.cycle_session_prompts(["prompt-a", "prompt-b", "prompt-c"])


