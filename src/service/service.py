import asyncio
import inspect
import json
import logging
import warnings
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal
from uuid import UUID, uuid4

import anyio
import httpx
from fastapi import APIRouter, Depends, FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRoute
from langchain_core._api import LangChainBetaWarning
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langfuse import Langfuse  # type: ignore[import-untyped]
from langfuse.langchain import (
    CallbackHandler,  # type: ignore[import-untyped]
)
from langgraph.types import Command, Interrupt
from langsmith import Client as LangsmithClient
from langsmith import uuid7

from agents import DEFAULT_AGENT, AgentGraph, get_agent, get_all_agent_info, load_agent
from agents.context import AgentContext
from core import settings
from core.observability import DiagnosisTracer, create_tracing_client
from memory import initialize_database, initialize_store
from persistence.database import (
    create_business_engine,
    create_business_session_factory,
)
from persistence.repositories import (
    get_diagnosis_status,
    mark_unfinished_execution_failed,
)
from schema import (
    ChatHistory,
    ChatHistoryInput,
    ChatMessage,
    Feedback,
    FeedbackResponse,
    ServiceMetadata,
    StreamInput,
    UserInput,
    UserThreads,
    UserThreadsInput,
)
from schema.auth import AuthenticatedUser
from service.agui import router as agui_router
from service.auth import get_authenticated_user
from service.threads import list_user_threads
from service.utils import (
    convert_message_content_to_string,
    ensure_model_available,
    langchain_to_chat_message,
    messages_from_checkpoint,
    remove_tool_calls,
)

warnings.filterwarnings("ignore", category=LangChainBetaWarning)
logger = logging.getLogger(__name__)
PROGRESS_BY_NODE: dict[str, tuple[str, str]] = {
    "start_diagnosis": (
        "diagnosis_record_created",
        "诊断记录已创建",
    ),
    "request_model": (
        "model_response_received",
        "已收到模型响应",
    ),
    "execute_tools": (
        "tools_finished",
        "工具执行已结束",
    ),
    "prepare_evidence": (
        "evidence_validated",
        "工具证据已校验",
    ),
    "analyze_evidence": (
        "analysis_generated",
        "结构化分析已生成",
    ),
}


def custom_generate_unique_id(route: APIRoute) -> str:
    """Generate idiomatic operation IDs for OpenAPI client generation."""
    return route.name


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """初始化并清理服务进程共享的资源。"""
    tracing_client = create_tracing_client()
    app.state.tracing_client = tracing_client
    business_engine = create_business_engine()
    business_session_factory = create_business_session_factory(business_engine)

    try:
        async with (
            initialize_database() as saver,
            initialize_store() as store,
            httpx.AsyncClient(
                base_url=settings.METRICS_BASE_URL,
                timeout=settings.METRICS_TIMEOUT_SECONDS,
                trust_env=False,
            ) as metrics_client,
        ):
            # 保存可安全共享的资源。
            app.state.metrics_client = metrics_client
            app.state.metrics_semaphore = asyncio.Semaphore(settings.METRICS_MAX_CONCURRENCY)
            app.state.business_session_factory = business_session_factory

            # 初始化 LangGraph Checkpoint。
            if hasattr(saver, "setup"):
                await saver.setup()

            # 初始化 LangGraph Store。
            if hasattr(store, "setup"):
                await store.setup()

            # 加载 Agent，并绑定 Checkpoint 和 Store。
            agents = get_all_agent_info()
            for agent_info in agents:
                try:
                    await load_agent(agent_info.key)
                    logger.info(f"Agent loaded: {agent_info.key}")
                except Exception as exc:
                    logger.error(f"Failed to load agent {agent_info.key}: {exc}")

                agent = get_agent(agent_info.key)
                agent.checkpointer = saver
                agent.store = store

            yield
    except Exception as exc:
        logger.error(f"Error during database/store/agents initialization: {exc}")
        raise
    finally:
        try:
            await business_engine.dispose()
        finally:
            if tracing_client is not None:
                await asyncio.to_thread(tracing_client.close, timeout=5.0)


app = FastAPI(lifespan=lifespan, generate_unique_id_function=custom_generate_unique_id)
router = APIRouter(
    dependencies=[Depends(get_authenticated_user)],
)
# AG-UI protocol endpoints inherit the same bearer auth - see service/agui.py
router.include_router(agui_router)


@router.get("/info")
async def info() -> ServiceMetadata:
    models = list(settings.AVAILABLE_MODELS)
    models.sort()
    return ServiceMetadata(
        agents=get_all_agent_info(),
        models=models,
        default_agent=DEFAULT_AGENT,
        default_model=settings.DEFAULT_MODEL,
    )


async def record_trace_status(kwargs: dict[str, Any], business_status: str) -> None:
    """追踪不是业务事务；上传失败只记录告警，不改变诊断结果。"""
    if not settings.LANGSMITH_TRACING:
        return
    try:
        project = await asyncio.to_thread(
            app.state.tracing_client.read_project,
            project_name=settings.LANGSMITH_PROJECT,
        )
        await asyncio.to_thread(
            app.state.tracing_client.create_feedback,
            kwargs["config"]["run_id"],
            key="business_status",
            value=business_status,
            session_id=project.id,
            trace_id=kwargs["config"]["run_id"],
            stop_after_attempt=1,
        )
    except Exception as exc:
        logger.warning("追踪状态上传失败：error_type=%s", type(exc).__name__)


async def _handle_input(
    user_input: UserInput,
    agent: AgentGraph,
    agent_id: str,
    current_user: AuthenticatedUser,
) -> tuple[dict[str, Any], UUID]:
    """
    Parse user input and handle any required interrupt resumption.
    Returns kwargs for agent invocation and the run_id.
    """
    run_id = uuid7()
    thread_id = user_input.thread_id or str(uuid4())
    user_id = current_user.user_id

    configurable: dict[str, Any] = {
        "thread_id": thread_id,
        "user_id": user_id,
        "diagnosis_run_id": run_id,
    }
    if user_input.model is not None:
        ensure_model_available(user_input.model)
        configurable["model"] = user_input.model

    callbacks: list[Any] = []
    if settings.LANGFUSE_TRACING:
        # Initialize Langfuse CallbackHandler for Langchain (tracing)
        langfuse_handler = CallbackHandler()

        callbacks.append(langfuse_handler)

    if user_input.agent_config:
        # Check for reserved keys (including 'model' even if not in configurable)
        reserved_keys = {
            "thread_id",
            "user_id",
            "model",
            "diagnosis_run_id",
        }
        if overlap := reserved_keys & user_input.agent_config.keys():
            raise HTTPException(
                status_code=422,
                detail=f"agent_config contains reserved keys: {overlap}",
            )
        configurable.update(user_input.agent_config)

    config = RunnableConfig(
        configurable=configurable,
        metadata={"user_id": user_id, "agent_id": agent_id},
        run_id=run_id,
        callbacks=callbacks,
    )

    checkpointer = getattr(agent, "checkpointer", None)
    if checkpointer is not None:
        checkpoint_tuple = await checkpointer.aget_tuple(config)
        if checkpoint_tuple is not None and (
            checkpoint_tuple.metadata.get("user_id") != user_id
            or checkpoint_tuple.metadata.get("agent_id") != agent_id
        ):
            raise HTTPException(
                status_code=404,
                detail="Conversation not found",
            )

    # Check for interrupts that need to be resumed
    state = await agent.aget_state(config=config)

    interrupted_tasks = [
        task for task in state.tasks if hasattr(task, "interrupts") and task.interrupts
    ]

    input: Command | dict[str, Any]
    if interrupted_tasks:
        # 恢复沿用图状态中的业务诊断 ID，请求追踪 ID 仍保持本次新值。
        diagnosis_run_id = state.values.get("diagnosis_run_id")
        if not isinstance(diagnosis_run_id, UUID):
            raise HTTPException(
                status_code=500,
                detail="中断状态缺少有效的诊断记录 ID",
            )
        # 业务已结束时，不允许借助遗留的图中断继续执行。
        diagnosis_status = await get_diagnosis_status(
            app.state.business_session_factory,
            diagnosis_run_id=diagnosis_run_id,
            user_id=user_id,
            thread_id=thread_id,
        )
        if diagnosis_status is None:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found",
            )
        if diagnosis_status != "running":
            raise HTTPException(
                status_code=409,
                detail="诊断已结束，不能恢复，请使用新会话发起诊断",
            )

        config["configurable"]["diagnosis_run_id"] = diagnosis_run_id
        input = Command(resume=user_input.message)
    else:
        input = {"messages": [HumanMessage(content=user_input.message)]}

    config["metadata"].update({
        "thread_id": thread_id,
        "request_run_id": str(run_id),
        "diagnosis_run_id": str(configurable["diagnosis_run_id"]),
        "request_mode": "resume" if interrupted_tasks else "new",
    })
    config["run_name"] = f"{agent_id}.diagnosis"
    if settings.LANGSMITH_TRACING:
        config["callbacks"].append(DiagnosisTracer(
            client=app.state.tracing_client,
            project_name=settings.LANGSMITH_PROJECT,
        ))

    kwargs = {
        "input": input,
        "config": config,
        "context": AgentContext(
            user_id=current_user.user_id,
            metrics_client=app.state.metrics_client,
            metrics_semaphore=app.state.metrics_semaphore,
            business_session_factory=app.state.business_session_factory,
        ),
    }

    logger.info(
        "诊断请求已准备：run_id=%s diagnosis_run_id=%s thread_id=%s mode=%s",
        run_id,
        config["configurable"]["diagnosis_run_id"],
        thread_id,
        "resume" if interrupted_tasks else "new",
    )
    return kwargs, run_id


@router.post("/{agent_id}/invoke", operation_id="invoke_with_agent_id")
@router.post("/invoke")
async def invoke(
    user_input: UserInput,
    current_user: Annotated[
        AuthenticatedUser,
        Depends(get_authenticated_user),
    ],
    agent_id: str = DEFAULT_AGENT,
) -> ChatMessage:
    """
    Invoke an agent with user input to retrieve a final response.

    If agent_id is not provided, the default agent will be used.
    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to messages for recording feedback.
    """
    # NOTE: Currently this only returns the last message or interrupt.
    # In the case of an agent outputting multiple AIMessages (such as the background step
    # in interrupt-agent, or a tool step in research-assistant), it's omitted. Arguably,
    # you'd want to include it. You could update the API to return a list of ChatMessages
    # in that case.
    agent: AgentGraph = get_agent(agent_id)
    kwargs, run_id = await _handle_input(
        user_input,
        agent,
        agent_id,
        current_user,
    )

    try:
        response_events: list[tuple[str, Any]] = await agent.ainvoke(**kwargs, stream_mode=["updates", "values"])  # type: ignore # fmt: skip
        response_type, response = response_events[-1]
        # A run that stops on an interrupt reports it on the final event of either stream
        # mode, so check for the interrupt before falling back to the last message.
        if "__interrupt__" in response:
            # Return the value of the first interrupt as an AIMessage
            output = langchain_to_chat_message(
                AIMessage(content=response["__interrupt__"][0].value)
            )
        elif response_type == "values":
            # Normal response, the agent completed successfully
            output = langchain_to_chat_message(response["messages"][-1])
        else:
            raise ValueError(f"Unexpected response type: {response_type}")

        output.run_id = str(run_id)
        logger.info(
            "普通请求响应已生成：run_id=%s diagnosis_run_id=%s status=%s",
            run_id,
            kwargs["config"]["configurable"]["diagnosis_run_id"],
            "paused" if "__interrupt__" in response else response.get("status"),
        )
        await record_trace_status(
            kwargs, "paused" if "__interrupt__" in response else response.get("status", "unknown")
        )
        return output
    except asyncio.CancelledError:
        with anyio.CancelScope(shield=True):
            await record_trace_status(kwargs, "cancelled")
        # 取消时完成业务收尾，然后继续传播取消信号。
        with anyio.CancelScope(shield=True):
            try:
                await mark_unfinished_execution_failed(
                    app.state.business_session_factory,
                    diagnosis_run_id=kwargs["config"]["configurable"]["diagnosis_run_id"],
                    error_message="请求执行已取消",
                )
            except Exception:
                logger.exception("普通请求取消收尾失败：run_id=%s", run_id)
        raise
    except Exception as exc:
        await record_trace_status(kwargs, "error")
        logger.error(
            "普通请求执行异常：run_id=%s diagnosis_run_id=%s error_type=%s",
            run_id, kwargs["config"]["configurable"]["diagnosis_run_id"], type(exc).__name__,
        )
        # 复用现有短事务，原子收尾诊断和未结束的工具日志。
        with anyio.CancelScope(shield=True):
            try:
                await mark_unfinished_execution_failed(
                    app.state.business_session_factory,
                    diagnosis_run_id=kwargs["config"]["configurable"]["diagnosis_run_id"],
                    error_message="执行发生未处理异常",
                )
            except Exception:
                logger.exception("普通请求异常收尾失败：run_id=%s", run_id)
        raise HTTPException(status_code=500, detail="Unexpected error") from exc


async def message_generator(
    user_input: StreamInput,
    agent: AgentGraph,
    kwargs: dict[str, Any],
    run_id: UUID,
) -> AsyncGenerator[str, None]:
    """使用已校验的执行参数运行工作流，并输出 SSE 事件。"""
    terminal_status: Literal["succeeded", "failed", "paused"] | None = None

    try:
        # Process streamed events from the graph and yield messages over the SSE stream.
        async for stream_event in agent.astream(  # type: ignore[no-matching-overload]
            **kwargs, stream_mode=["updates", "messages", "custom"], subgraphs=True
        ):
            if not isinstance(stream_event, tuple):
                continue
            # Handle different stream event structures based on subgraphs
            if len(stream_event) == 3:
                # With subgraphs=True: (node_path, stream_mode, event)
                _, stream_mode, event = stream_event
            else:
                # Without subgraphs: (stream_mode, event)
                stream_mode, event = stream_event
            new_messages: list[Any] = []
            if stream_mode == "updates":
                for node, updates in event.items():
                    # 暂停正常结束本次流，业务诊断仍等待恢复。
                    if node == "__interrupt__":
                        if not updates:
                            raise ValueError("中断事件缺少中断信息")

                        terminal_status = "paused"
                        interrupt: Interrupt
                        for interrupt in updates:
                            yield (
                                "data: "
                                + json.dumps(
                                    {
                                        "type": "interrupt",
                                        "content": interrupt.value,
                                        "interrupt_id": interrupt.id,
                                        "thread_id": kwargs["config"]["configurable"]["thread_id"],
                                        "diagnosis_run_id": str(
                                            kwargs["config"]["configurable"]["diagnosis_run_id"]
                                        ),
                                        "run_id": str(run_id),
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n\n"
                            )
                        continue
                    progress = PROGRESS_BY_NODE.get(node)
                    if progress is not None:
                        stage, content = progress
                        yield (
                            "data: "
                            + json.dumps(
                                {
                                    "type": "progress",
                                    "stage": stage,
                                    "content": content,
                                    "run_id": str(run_id),
                                },
                                ensure_ascii=False,
                            )
                            + "\n\n"
                        )
                    updates = updates or {}
                    status_update = updates.get("status")
                    if status_update in {"succeeded", "failed"}:
                        terminal_status = status_update
                    update_messages = updates.get("messages", [])
                    # special cases for using langgraph-supervisor library
                    if "supervisor" in node or "sub-agent" in node:
                        # the only tools that come from the actual agent are the handoff and handback tools
                        if isinstance(update_messages[-1], ToolMessage):
                            if "sub-agent" in node and len(update_messages) > 1:
                                # If this is a sub-agent, we want to keep the last 2 messages - the handback tool, and it's result
                                update_messages = update_messages[-2:]
                            else:
                                # If this is a supervisor, we want to keep the last message only - the handoff result. The tool comes from the 'agent' node.
                                update_messages = [update_messages[-1]]
                        else:
                            update_messages = []
                    new_messages.extend(update_messages)

            if stream_mode == "custom":
                new_messages = [event]

            # LangGraph streaming may emit tuples: (field_name, field_value)
            # e.g. ('content', <str>), ('tool_calls', [ToolCall,...]), ('additional_kwargs', {...}), etc.
            # We accumulate only supported fields into `parts` and skip unsupported metadata.
            # More info at: https://langchain-ai.github.io/langgraph/cloud/how-tos/stream_messages/
            processed_messages = []
            current_message: dict[str, Any] = {}
            for message in new_messages:
                if isinstance(message, tuple):
                    key, value = message
                    # Store parts in temporary dict
                    current_message[key] = value
                else:
                    # Add complete message if we have one in progress
                    if current_message:
                        processed_messages.append(_create_ai_message(current_message))
                        current_message = {}
                    processed_messages.append(message)

            # Add any remaining message parts
            if current_message:
                processed_messages.append(_create_ai_message(current_message))

            for message in processed_messages:
                try:
                    chat_message = langchain_to_chat_message(message)
                    chat_message.run_id = str(run_id)
                except Exception as e:
                    logger.error(f"Error parsing message: {e}")
                    yield f"data: {json.dumps({'type': 'error', 'content': 'Unexpected error'})}\n\n"
                    continue
                # LangGraph re-sends the input message, which feels weird, so drop it
                if chat_message.type == "human" and chat_message.content == user_input.message:
                    continue
                yield f"data: {json.dumps({'type': 'message', 'content': chat_message.model_dump()})}\n\n"

            if stream_mode == "messages":
                if not user_input.stream_tokens:
                    continue
                msg, metadata = event
                if "skip_stream" in metadata.get("tags", []):
                    continue
                # For some reason, astream("messages") causes non-LLM nodes to send extra messages.
                # Drop them.
                if not isinstance(msg, AIMessageChunk):
                    continue
                content = remove_tool_calls(msg.content)
                if content:
                    # Empty content in the context of OpenAI usually means
                    # that the model is asking for a tool to be invoked.
                    # So we only print non-empty content.
                    yield f"data: {json.dumps({'type': 'token', 'content': convert_message_content_to_string(content)})}\n\n"

        if terminal_status is None:
            raise RuntimeError("工作流结束但没有产生业务终态或中断")

        await record_trace_status(kwargs, terminal_status)
        logger.info(
            "流式执行结束：run_id=%s diagnosis_run_id=%s status=%s",
            run_id,
            kwargs["config"]["configurable"]["diagnosis_run_id"],
            terminal_status,
        )
        yield (
            "data: "
            + json.dumps(
                {
                    "type": "done",
                    "status": terminal_status,
                    "run_id": str(run_id),
                },
                ensure_ascii=False,
            )
            + "\n\n"
        )
    except asyncio.CancelledError:
        with anyio.CancelScope(shield=True):
            await record_trace_status(kwargs, "cancelled")
        logger.info(
            "SSE client disconnected or stream task was cancelled: run_id=%s",
            run_id,
        )

        # 客户端断开的取消作用域不能再次取消数据库收尾事务。
        with anyio.CancelScope(shield=True):
            try:
                await mark_unfinished_execution_failed(
                    app.state.business_session_factory,
                    diagnosis_run_id=kwargs["config"]["configurable"]["diagnosis_run_id"],
                    error_message="客户端断开，执行已取消",
                )
            except Exception:
                logger.exception(
                    "Failed to finalize cancelled execution: run_id=%s",
                    run_id,
                )
        raise
    except Exception as exc:
        await record_trace_status(kwargs, "error")
        logger.error(
            "流式执行异常：run_id=%s diagnosis_run_id=%s error_type=%s",
            run_id,
            kwargs["config"]["configurable"]["diagnosis_run_id"],
            type(exc).__name__,
        )

        # 流式响应发生未处理异常时，先完成数据库状态收尾。
        with anyio.CancelScope(shield=True):
            try:
                await mark_unfinished_execution_failed(
                    app.state.business_session_factory,
                    diagnosis_run_id=kwargs["config"]["configurable"]["diagnosis_run_id"],
                    error_message="执行发生未处理异常",
                )
            except Exception:
                logger.exception(
                    "Failed to finalize errored execution: run_id=%s",
                    run_id,
                )

        yield (
            "data: "
            + json.dumps(
                {
                    "type": "error",
                    "content": "Internal server error",
                    "run_id": str(run_id),
                },
                ensure_ascii=False,
            )
            + "\n\n"
        )

        yield (
            "data: "
            + json.dumps(
                {
                    "type": "done",
                    "status": "error",
                    "run_id": str(run_id),
                },
                ensure_ascii=False,
            )
            + "\n\n"
        )


def _create_ai_message(parts: dict) -> AIMessage:
    sig = inspect.signature(AIMessage)
    valid_keys = set(sig.parameters)
    filtered = {k: v for k, v in parts.items() if k in valid_keys}
    return AIMessage(**filtered)


def _sse_response_example() -> dict[int | str, Any]:
    return {
        status.HTTP_200_OK: {
            "description": "Server Sent Event Response",
            "content": {
                "text/event-stream": {
                    "example": (
                        'data: {"type": "progress", "stage": "diagnosis_record_created", '
                        '"content": "诊断记录已创建", "run_id": "example-run-id"}\n\n'
                        'data: {"type": "done", "status": "succeeded", '
                        '"run_id": "example-run-id"}\n\n'
                    ),
                    "schema": {"type": "string"},
                }
            },
        }
    }


@router.post(
    "/{agent_id}/stream",
    response_class=StreamingResponse,
    responses=_sse_response_example(),
    operation_id="stream_with_agent_id",
)
@router.post("/stream", response_class=StreamingResponse, responses=_sse_response_example())
async def stream(
    user_input: StreamInput,
    current_user: Annotated[
        AuthenticatedUser,
        Depends(get_authenticated_user),
    ],
    agent_id: str = DEFAULT_AGENT,
) -> StreamingResponse:
    """
    Stream an agent's response to a user input, including intermediate messages and tokens.

    If agent_id is not provided, the default agent will be used.
    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to all messages for recording feedback.

    Set `stream_tokens=false` to return intermediate messages but not token-by-token.
    """
    # 在响应开始前完成授权及恢复检查，使入口错误保留 HTTP 状态码。
    agent = get_agent(agent_id)
    kwargs, run_id = await _handle_input(
        user_input,
        agent,
        agent_id,
        current_user,
    )
    return StreamingResponse(
        message_generator(user_input, agent, kwargs, run_id),
        media_type="text/event-stream",
    )


@router.post("/feedback")
async def feedback(feedback: Feedback) -> FeedbackResponse:
    """
    Record feedback for a run to LangSmith.

    This is a simple wrapper for the LangSmith create_feedback API, so the
    credentials can be stored and managed in the service rather than the client.
    See: https://api.smith.langchain.com/redoc#tag/feedback/operation/create_feedback_api_v1_feedback_post
    """
    client = LangsmithClient()
    kwargs = feedback.kwargs or {}
    client.create_feedback(
        run_id=feedback.run_id,
        key=feedback.key,
        score=feedback.score,
        **kwargs,
    )
    return FeedbackResponse()


@router.post("/{agent_id}/history", operation_id="history_with_agent_id")
@router.post("/history")
async def history(
    input: ChatHistoryInput,
    current_user: Annotated[
        AuthenticatedUser,
        Depends(get_authenticated_user),
    ],
    agent_id: str = DEFAULT_AGENT,
) -> ChatHistory:
    """
    Get chat history for a thread and agent.

    If agent_id is not provided, the default agent will be used.
    """
    agent: AgentGraph = get_agent(agent_id)
    config = RunnableConfig(configurable={"thread_id": input.thread_id})
    try:
        messages: list[BaseMessage] = []
        # Functional-API agents keep the conversation in `__previous__`, which aget_state
        # doesn't return, so read the raw checkpoint first and only fall back for graphs.
        checkpointer = getattr(agent, "checkpointer", None)

        if checkpointer is None:
            raise RuntimeError("Agent 未配置 Checkpoint 存储")

        tup = await checkpointer.aget_tuple(config)

        if tup is None:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found",
            )

        stored_user_id = tup.metadata.get("user_id")
        stored_agent_id = tup.metadata.get("agent_id")

        if stored_user_id != current_user.user_id or stored_agent_id != agent_id:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found",
            )

        if "__previous__" in (tup.checkpoint.get("channel_values") or {}):
            messages = messages_from_checkpoint(tup.checkpoint)
        if not messages:
            state_snapshot = await agent.aget_state(config=config)
            messages = state_snapshot.values["messages"]
        chat_messages: list[ChatMessage] = [langchain_to_chat_message(m) for m in messages]
        return ChatHistory(messages=chat_messages)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"An exception occurred: {e}")
        raise HTTPException(
            status_code=500,
            detail="Unexpected error",
        ) from e


@router.get("/{agent_id}/threads", operation_id="threads_with_agent_id")
@router.get("/threads")
async def threads(
    current_user: Annotated[
        AuthenticatedUser,
        Depends(get_authenticated_user),
    ],
    input: UserThreadsInput = Depends(),
    agent_id: str = DEFAULT_AGENT,
) -> UserThreads:
    """列出当前认证用户在指定 Agent 下的会话。"""
    agent: AgentGraph = get_agent(agent_id)
    checkpointer = getattr(agent, "checkpointer", None)

    if not checkpointer:
        return UserThreads(threads=[])

    try:
        summaries = await list_user_threads(
            checkpointer,
            current_user.user_id,
            agent_id,
            input.limit,
        )
    except Exception as e:
        logger.error(f"An exception occurred: {e}")
        raise HTTPException(
            status_code=500,
            detail="Unexpected error",
        ) from e

    return UserThreads(threads=summaries)


@app.get("/health")
async def health_check():
    """Health check endpoint."""

    health_status = {"status": "ok"}

    if settings.LANGFUSE_TRACING:
        try:
            langfuse = Langfuse()
            health_status["langfuse"] = "connected" if langfuse.auth_check() else "disconnected"
        except Exception as e:
            logger.error(f"Langfuse connection error: {e}")
            health_status["langfuse"] = "disconnected"

    return health_status


app.include_router(router)
