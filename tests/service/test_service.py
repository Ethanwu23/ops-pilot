import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import langsmith
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.types import Interrupt, StateSnapshot

from agents.agents import Agent
from schema import ChatHistory, ChatMessage, ServiceMetadata
from schema.models import AnthropicModelName, OpenAIModelName


def test_invoke(test_client, mock_agent) -> None:
    QUESTION = "What is the weather in Tokyo?"
    ANSWER = "The weather in Tokyo is 70 degrees."
    mock_agent.ainvoke.return_value = [("values", {"messages": [AIMessage(content=ANSWER)]})]

    response = test_client.post("/invoke", json={"message": QUESTION})
    assert response.status_code == 200

    mock_agent.ainvoke.assert_awaited_once()
    input_message = mock_agent.ainvoke.await_args.kwargs["input"]["messages"][0]
    assert input_message.content == QUESTION

    output = ChatMessage.model_validate(response.json())
    assert output.type == "ai"
    assert output.content == ANSWER


def test_invoke_custom_agent(test_client, mock_agent) -> None:
    """Test that /invoke works with a custom agent_id path parameter."""
    CUSTOM_AGENT = "custom_agent"
    QUESTION = "What is the weather in Tokyo?"
    CUSTOM_ANSWER = "The weather in Tokyo is sunny."
    DEFAULT_ANSWER = "This is from the default agent."

    # Create a separate mock for the default agent
    default_mock = AsyncMock()
    default_mock.ainvoke.return_value = [
        ("values", {"messages": [AIMessage(content=DEFAULT_ANSWER)]})
    ]

    # Configure our custom mock agent
    mock_agent.ainvoke.return_value = [("values", {"messages": [AIMessage(content=CUSTOM_ANSWER)]})]

    # Patch get_agent to return the correct agent based on the provided agent_id
    def agent_lookup(agent_id):
        if agent_id == CUSTOM_AGENT:
            return mock_agent
        return default_mock

    with patch("service.service.get_agent", side_effect=agent_lookup):
        response = test_client.post(f"/{CUSTOM_AGENT}/invoke", json={"message": QUESTION})
        assert response.status_code == 200

        # Verify custom agent was called and default wasn't
        mock_agent.ainvoke.assert_awaited_once()
        default_mock.ainvoke.assert_not_awaited()

        input_message = mock_agent.ainvoke.await_args.kwargs["input"]["messages"][0]
        assert input_message.content == QUESTION

        output = ChatMessage.model_validate(response.json())
        assert output.type == "ai"
        assert output.content == CUSTOM_ANSWER  # Verify we got the custom agent's response


def test_invoke_model_param(test_client, mock_agent) -> None:
    """Test that the model parameter is correctly passed to the agent if specified."""
    QUESTION = "What is the weather in Tokyo?"
    ANSWER = "The weather in Tokyo is sunny."
    CUSTOM_MODEL = OpenAIModelName.GPT_5_MINI
    mock_agent.ainvoke.return_value = [("values", {"messages": [AIMessage(content=ANSWER)]})]

    response = test_client.post("/invoke", json={"message": QUESTION, "model": CUSTOM_MODEL})
    assert response.status_code == 200

    # Verify the model was passed correctly in the config
    mock_agent.ainvoke.assert_awaited_once()
    config = mock_agent.ainvoke.await_args.kwargs["config"]
    assert config["configurable"]["model"] == CUSTOM_MODEL

    # Verify the response is still correct
    output = ChatMessage.model_validate(response.json())
    assert output.type == "ai"
    assert output.content == ANSWER

    # Verify a valid enum outside the configured allowlist returns a 400.
    unavailable_model = AnthropicModelName.SONNET_45
    response = test_client.post("/invoke", json={"message": QUESTION, "model": unavailable_model})
    assert response.status_code == 400
    assert "not available" in response.json()["detail"]

    # Verify a malformed model string still fails request validation.
    INVALID_MODEL = "gpt-7-notreal"
    response = test_client.post("/invoke", json={"message": QUESTION, "model": INVALID_MODEL})
    assert response.status_code == 422


def test_invoke_no_model_param_uses_none_default(test_client, mock_agent) -> None:
    """Test that when no model is specified, UserInput defaults to None and isn't passed to the runnable config (not hardcoded gpt-5-nano)."""
    QUESTION = "What is the weather in Tokyo?"
    ANSWER = "The weather in Tokyo is sunny."
    mock_agent.ainvoke.return_value = [("values", {"messages": [AIMessage(content=ANSWER)]})]

    # Don't specify model in the request
    response = test_client.post("/invoke", json={"message": QUESTION})
    assert response.status_code == 200

    mock_agent.ainvoke.assert_awaited_once()
    config = mock_agent.ainvoke.await_args.kwargs["config"]
    assert "model" not in config["configurable"]  # Should not be present when None

    # Verify the response is still correct
    output = ChatMessage.model_validate(response.json())
    assert output.type == "ai"
    assert output.content == ANSWER


def test_invoke_custom_agent_config(test_client, mock_agent) -> None:
    """Test that the agent_config parameter is correctly passed to the agent."""
    QUESTION = "What is the weather in Tokyo?"
    ANSWER = "The weather in Tokyo is sunny."
    CUSTOM_CONFIG = {"spicy_level": 0.1, "additional_param": "value_foo"}

    mock_agent.ainvoke.return_value = [("values", {"messages": [AIMessage(content=ANSWER)]})]

    response = test_client.post(
        "/invoke", json={"message": QUESTION, "agent_config": CUSTOM_CONFIG}
    )
    assert response.status_code == 200

    # Verify the agent_config was passed correctly in the config
    mock_agent.ainvoke.assert_awaited_once()
    config = mock_agent.ainvoke.await_args.kwargs["config"]
    assert config["configurable"]["spicy_level"] == 0.1
    assert config["configurable"]["additional_param"] == "value_foo"

    # Verify the response is still correct
    output = ChatMessage.model_validate(response.json())
    assert output.type == "ai"
    assert output.content == ANSWER

    # Verify a reserved key in agent_config throws a validation error
    INVALID_CONFIG = {"model": "gpt-5-nano"}
    response = test_client.post(
        "/invoke", json={"message": QUESTION, "agent_config": INVALID_CONFIG}
    )
    assert response.status_code == 422


def test_invoke_interrupt(test_client, mock_agent) -> None:
    QUESTION = "What is the weather in Tokyo?"
    ANSWER = "The weather in Tokyo is 70 degrees."
    INTERRUPT = "Confirm weather check"
    mock_agent.ainvoke.return_value = [
        ("values", {"messages": [AIMessage(content=ANSWER)]}),
        ("updates", {"__interrupt__": [Interrupt(value=INTERRUPT)]}),
    ]

    response = test_client.post("/invoke", json={"message": QUESTION})
    assert response.status_code == 200

    mock_agent.ainvoke.assert_awaited_once()
    input_message = mock_agent.ainvoke.await_args.kwargs["input"]["messages"][0]
    assert input_message.content == QUESTION

    output = ChatMessage.model_validate(response.json())
    assert output.type == "ai"
    assert output.content == INTERRUPT


@patch("service.service.LangsmithClient")
def test_feedback(mock_client: langsmith.Client, test_client) -> None:
    ls_instance = mock_client.return_value
    ls_instance.create_feedback.return_value = None
    body = {
        "run_id": "847c6285-8fc9-4560-a83f-4e6285809254",
        "key": "human-feedback-stars",
        "score": 0.8,
    }
    response = test_client.post("/feedback", json=body)
    assert response.status_code == 200
    assert response.json() == {"status": "success"}
    ls_instance.create_feedback.assert_called_once_with(
        run_id="847c6285-8fc9-4560-a83f-4e6285809254",
        key="human-feedback-stars",
        score=0.8,
    )


def test_history(test_client, mock_agent) -> None:
    QUESTION = "What is the weather in Tokyo?"
    ANSWER = "The weather in Tokyo is 70 degrees."
    user_question = HumanMessage(content=QUESTION)
    agent_response = AIMessage(content=ANSWER)
    mock_agent.aget_state.return_value = StateSnapshot(
        values={"messages": [user_question, agent_response]},
        next=(),
        config={},
        metadata=None,
        created_at=None,
        parent_config=None,
        tasks=(),
        interrupts=(),
    )

    mock_agent.checkpointer = SimpleNamespace(
        aget_tuple=AsyncMock(return_value=SimpleNamespace(
            metadata={"user_id": "user-123", "agent_id": "chatbot"},
            checkpoint={"channel_values": {}},
        )),
    )

    response = test_client.post(
        "/history", json={"thread_id": "7bcc7cc1-99d7-4b1d-bdb5-e6f90ed44de6"}
    )
    assert response.status_code == 200

    output = ChatHistory.model_validate(response.json())
    assert output.messages[0].type == "human"
    assert output.messages[0].content == QUESTION
    assert output.messages[1].type == "ai"
    assert output.messages[1].content == ANSWER


def test_history_custom_agent(test_client) -> None:
    """Test that /{agent_id}/history reads the thread through the requested agent's graph."""
    CUSTOM_AGENT = "custom_agent"
    QUESTION = "What is the weather in Tokyo?"
    ANSWER = "The weather in Tokyo is 70 degrees."

    custom_snapshot = StateSnapshot(
        values={"messages": [HumanMessage(content=QUESTION), AIMessage(content=ANSWER)]},
        next=(),
        config={},
        metadata=None,
        created_at=None,
        parent_config=None,
        tasks=(),
        interrupts=(),
    )
    # The default agent's graph doesn't know about this thread, so it returns no messages.
    default_snapshot = StateSnapshot(
        values={"messages": []},
        next=(),
        config={},
        metadata=None,
        created_at=None,
        parent_config=None,
        tasks=(),
        interrupts=(),
    )

    custom_mock = AsyncMock()
    custom_mock.aget_state.return_value = custom_snapshot
    custom_mock.checkpointer = SimpleNamespace(
        aget_tuple=AsyncMock(return_value=SimpleNamespace(
            metadata={"user_id": "user-123", "agent_id": CUSTOM_AGENT},
            checkpoint={"channel_values": {}},
        )),
    )
    default_mock = AsyncMock()
    default_mock.aget_state.return_value = default_snapshot
    default_mock.checkpointer = None

    def agent_lookup(agent_id):
        if agent_id == CUSTOM_AGENT:
            return custom_mock
        return default_mock

    with patch("service.service.get_agent", side_effect=agent_lookup):
        response = test_client.post(
            f"/{CUSTOM_AGENT}/history",
            json={"thread_id": "7bcc7cc1-99d7-4b1d-bdb5-e6f90ed44de6"},
        )
        assert response.status_code == 200

        # The custom agent's graph was used, not the default one.
        custom_mock.aget_state.assert_awaited_once()
        default_mock.aget_state.assert_not_awaited()

        output = ChatHistory.model_validate(response.json())
        assert output.messages[0].type == "human"
        assert output.messages[0].content == QUESTION
        assert output.messages[1].type == "ai"
        assert output.messages[1].content == ANSWER


@pytest.mark.asyncio
async def test_stream(test_client, mock_agent) -> None:
    """Test streaming tokens and messages."""
    QUESTION = "What is the weather in Tokyo?"
    TOKENS = ["The", " weather", " in", " Tokyo", " is", " sunny", "."]
    FINAL_ANSWER = "The weather in Tokyo is sunny."

    # Configure mock to use our async iterator function
    events = [
        (
            "messages",
            (
                AIMessageChunk(content=token),
                {"tags": []},
            ),
        )
        for token in TOKENS
    ] + [
        (
            "updates",
            {
                "finish_direct": {
                    "messages": [AIMessage(content=FINAL_ANSWER)],
                    "status": "succeeded",
                }
            },
        )
    ]

    async def mock_astream(**kwargs):
        for event in events:
            yield event

    mock_agent.astream = mock_astream

    # Make request with streaming
    with test_client.stream(
        "POST", "/stream", json={"message": QUESTION, "stream_tokens": True}
    ) as response:
        assert response.status_code == 200

        # Collect all SSE messages
        messages = []
        for line in response.iter_lines():
            if line.startswith("data: "):
                messages.append(json.loads(line.removeprefix("data: ")))

        # Verify streamed tokens
        token_messages = [msg for msg in messages if msg["type"] == "token"]
        assert len(token_messages) == len(TOKENS)
        for i, msg in enumerate(token_messages):
            assert msg["content"] == TOKENS[i]

        # Verify final message
        final_messages = [msg for msg in messages if msg["type"] == "message"]
        assert len(final_messages) == 1
        assert final_messages[0]["content"]["content"] == FINAL_ANSWER
        assert final_messages[0]["content"]["type"] == "ai"
        assert not any(event["type"] == "error" for event in messages)
        assert messages[-1]["type"] == "done"
        assert messages[-1]["status"] == "succeeded"


@pytest.mark.asyncio
async def test_stream_no_tokens(test_client, mock_agent) -> None:
    """Test streaming without tokens."""
    QUESTION = "What is the weather in Tokyo?"
    TOKENS = ["The", " weather", " in", " Tokyo", " is", " sunny", "."]
    FINAL_ANSWER = "The weather in Tokyo is sunny."

    # Configure mock to use our async iterator function
    events = [
        (
            "messages",
            (
                AIMessageChunk(content=token),
                {"tags": []},
            ),
        )
        for token in TOKENS
    ] + [
        (
            "updates",
            {
                "finish_direct": {
                    "messages": [AIMessage(content=FINAL_ANSWER)],
                    "status": "succeeded",
                }
            },
        )
    ]

    async def mock_astream(**kwargs):
        for event in events:
            yield event

    mock_agent.astream = mock_astream

    # Make request with streaming disabled
    with test_client.stream(
        "POST", "/stream", json={"message": QUESTION, "stream_tokens": False}
    ) as response:
        assert response.status_code == 200

        # Collect all SSE messages
        messages = []
        for line in response.iter_lines():
            if line.startswith("data: "):
                messages.append(json.loads(line.removeprefix("data: ")))

        # Verify no token messages
        token_messages = [msg for msg in messages if msg["type"] == "token"]
        assert len(token_messages) == 0

        # Verify final message
        final_messages = [
            event for event in messages
            if event["type"] == "message"
        ]
        assert len(final_messages) == 1
        assert final_messages[0]["content"]["content"] == FINAL_ANSWER
        assert final_messages[0]["content"]["type"] == "ai"
        assert not any(event["type"] == "error" for event in messages)
        assert messages[-1]["type"] == "done"
        assert messages[-1]["status"] == "succeeded"


def test_stream_interrupt(test_client, mock_agent) -> None:
    QUESTION = "What is the weather in Tokyo?"
    INTERRUPT = "Confirm weather check"
    # Configure mock to use our async iterator function
    events = [
        (
            "updates",
            {"__interrupt__": [Interrupt(value=INTERRUPT)]},
        )
    ]

    async def mock_astream(**kwargs):
        for event in events:
            yield event

    mock_agent.astream = mock_astream

    # Make request with streaming disabled
    with test_client.stream(
        "POST", "/stream", json={"message": QUESTION, "stream_tokens": False}
    ) as response:
        assert response.status_code == 200

        # 暂停是独立事件，当前流结束不等于业务成功。
        messages = [
            json.loads(line.removeprefix("data: "))
            for line in response.iter_lines()
            if line.startswith("data: ")
        ]
        assert [event["type"] for event in messages] == ["interrupt", "done"]
        interrupted, done = messages
        assert interrupted["content"] == INTERRUPT
        assert interrupted["interrupt_id"]
        assert interrupted["thread_id"]
        assert interrupted["diagnosis_run_id"]
        assert done["status"] == "paused"
        assert done["run_id"] == interrupted["run_id"]


def test_info(test_client, mock_settings) -> None:
    """Test that /info returns the correct service metadata."""

    base_agent = Agent(description="A base agent.", graph_like=None)
    mock_settings.DEFAULT_MODEL = OpenAIModelName.GPT_5_NANO
    mock_settings.AVAILABLE_MODELS = {OpenAIModelName.GPT_5_NANO, OpenAIModelName.GPT_5_MINI}
    with patch.dict("agents.agents.agents", {"base-agent": base_agent}, clear=True):
        response = test_client.get("/info")
        assert response.status_code == 200
        output = ServiceMetadata.model_validate(response.json())

    assert output.default_agent == "chatbot"
    assert len(output.agents) == 1
    assert output.agents[0].key == "base-agent"
    assert output.agents[0].description == "A base agent."

    assert output.default_model == OpenAIModelName.GPT_5_NANO
    assert output.models == [OpenAIModelName.GPT_5_MINI, OpenAIModelName.GPT_5_NANO]


@pytest.mark.parametrize("owner", [None, "other-user", "other-agent"])
def test_history_rejects_inaccessible_checkpoint(test_client, mock_agent, owner):
    """历史不存在或归属不匹配时，不得读取图状态。"""
    checkpoint = None
    if owner is not None:
        checkpoint = SimpleNamespace(
            metadata={
                "user_id": "other-user" if owner == "other-user" else "user-123",
                "agent_id": "other-agent" if owner == "other-agent" else "chatbot",
            },
            checkpoint={"channel_values": {}},
        )
    mock_agent.checkpointer = SimpleNamespace(aget_tuple=AsyncMock(return_value=checkpoint))

    response = test_client.post("/history", json={"thread_id": "private-thread"})

    assert response.status_code == 404
    assert response.json() == {"detail": "Conversation not found"}
    mock_agent.aget_state.assert_not_awaited()


@pytest.mark.parametrize("path", ["/invoke", "/stream"])
@pytest.mark.parametrize("business_status, expected_status", [(None, 404), ("failed", 409), ("succeeded", 409)])
def test_resume_rejects_ended_or_missing_diagnosis(
    test_client, mock_agent, path, business_status, expected_status,
):
    """恢复入口必须在执行工作流和发送 SSE 之前拒绝无效业务状态。"""
    diagnosis_id = uuid4()
    mock_agent.aget_state.return_value = SimpleNamespace(
        tasks=[SimpleNamespace(interrupts=("waiting",))],
        values={"diagnosis_run_id": diagnosis_id},
    )
    with patch("service.service.get_diagnosis_status", new_callable=AsyncMock) as lookup:
        lookup.return_value = business_status
        response = test_client.post(path, json={"message": "continue", "thread_id": "paused-thread"})

    assert response.status_code == expected_status
    assert response.headers["content-type"].startswith("application/json")
    assert "data: " not in response.text
    if expected_status == 404:
        assert response.json() == {"detail": "Conversation not found"}
    else:
        assert response.json()["detail"] == "诊断已结束，不能恢复，请使用新会话发起诊断"
    lookup.assert_awaited_once()
    assert lookup.await_args.kwargs == {
        "diagnosis_run_id": diagnosis_id,
        "user_id": "user-123",
        "thread_id": "paused-thread",
    }
    mock_agent.ainvoke.assert_not_awaited()
    mock_agent.astream.assert_not_called()


def test_stream_without_terminal_status_reports_error(test_client, mock_agent):
    """回答消息已输出，也不能把缺少业务终态的执行当成成功。"""
    async def events(**kwargs):
        yield ("updates", {"request_model": {"messages": [AIMessage(content="已有回答")]}})

    mock_agent.astream = events
    with patch("service.service.mark_unfinished_execution_failed", new_callable=AsyncMock) as finalize:
        response = test_client.post("/stream", json={"message": "测试缺失终态", "stream_tokens": False})

    assert response.status_code == 200
    frames = [
        json.loads(line.removeprefix("data: "))
        for line in response.iter_lines()
        if line.startswith("data: ")
    ]
    assert any(frame["type"] == "message" for frame in frames)
    assert frames[-2]["type"] == "error"
    assert frames[-1]["type"] == "done"
    assert frames[-1]["status"] == "error"
    finalize.assert_awaited_once()
    assert str(finalize.await_args.kwargs["diagnosis_run_id"]) == frames[-1]["run_id"]
