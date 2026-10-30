from time import perf_counter
from typing import Literal
from uuid import UUID

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState
from langgraph.runtime import get_runtime

from persistence.repositories import create_diagnosis_run
from schema.diagnosis import AnalysisResult, Evidence


class DiagnosticState(MessagesState, total=False):
    """保存诊断工作流中的消息、执行状态和错误信息。"""

    status: Literal["succeeded", "failed", "running"]
    error: str | None
    evidence: list[Evidence]
    analysis: AnalysisResult | None
    diagnosis_run_id: UUID
    started_at_monotonic: float


async def start_diagnosis(
    state: DiagnosticState,
    config: RunnableConfig,
) -> DiagnosticState:
    """创建业务诊断记录，并初始化本轮工作流状态。"""
    started_at_monotonic = perf_counter()

    user_message = next(
        (message for message in reversed(state["messages"]) if isinstance(message, HumanMessage)),
        None,
    )
    if user_message is None:
        raise ValueError("缺少本轮用户问题")

    if not isinstance(user_message.content, str):
        raise ValueError("本轮用户问题必须是字符串")

    question = user_message.content.strip()
    if not question:
        raise ValueError("本轮用户问题不能为空")

    thread_id = config["configurable"].get("thread_id")
    if not isinstance(thread_id, str) or not thread_id:
        raise ValueError("缺少有效的 thread_id")

    user_id = config["configurable"].get("user_id")
    if not isinstance(user_id, str) or not user_id:
        raise ValueError("缺少有效的 user_id")

    diagnosis_run_id = config["configurable"].get("diagnosis_run_id")
    if not isinstance(diagnosis_run_id, UUID):
        raise ValueError("缺少有效的 run_id")

    runtime = get_runtime()
    diagnosis_run_id = await create_diagnosis_run(
        runtime.context.business_session_factory,
        diagnosis_run_id=diagnosis_run_id,
        user_id=user_id,
        thread_id=thread_id,
        question=question,
    )
    return {
        "status": "running",
        "error": None,
        "evidence": [],
        "analysis": None,
        "diagnosis_run_id": diagnosis_run_id,
        "started_at_monotonic": started_at_monotonic,
    }
