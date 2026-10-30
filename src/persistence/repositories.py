from typing import Any
from uuid import UUID

from sqlalchemy import Integer, cast, func, update

from persistence.database import BusinessSessionFactory
from persistence.models import DiagnosisRun, ToolExecutionLog


async def create_diagnosis_run(
    session_factory: BusinessSessionFactory,
    *,
    diagnosis_run_id: UUID,
    user_id: str,
    thread_id: str,
    question: str,
) -> UUID:
    """创建一条运行中的诊断记录，并在返回前提交事务。"""
    async with session_factory() as session:
        async with session.begin():
            diagnosis_run = DiagnosisRun(
                id=diagnosis_run_id,
                user_id=user_id,
                thread_id=thread_id,
                status="running",
                question=question,
            )
            session.add(diagnosis_run)

            # 立即执行 INSERT，在事务提交前验证约束。
            await session.flush()

    return diagnosis_run_id


async def mark_diagnosis_succeeded(
    session_factory: BusinessSessionFactory,
    *,
    diagnosis_run_id: UUID,
    result_summary: str,
    duration_ms: int,
) -> None:
    """把运行中的诊断记录更新为成功状态。"""
    if duration_ms < 0:
        raise ValueError("duration_ms 不能小于 0")

    async with session_factory() as session:
        async with session.begin():
            statement = (
                update(DiagnosisRun)
                .where(
                    DiagnosisRun.id == diagnosis_run_id,
                    DiagnosisRun.status == "running",
                )
                .values(
                    status="succeeded",
                    result_summary=result_summary,
                    error_message=None,
                    finished_at=func.now(),
                    duration_ms=duration_ms,
                )
                .returning(DiagnosisRun.id)
            )

            updated_id = await session.scalar(statement)

            if updated_id is None:
                raise ValueError(f"诊断记录不存在或状态已结束：{diagnosis_run_id}")


async def mark_diagnosis_failed(
    session_factory: BusinessSessionFactory,
    *,
    diagnosis_run_id: UUID,
    error_message: str,
    duration_ms: int,
) -> None:
    """把运行中的诊断记录更新为失败状态。"""
    if not error_message:
        raise ValueError("error_message 不能为空")

    if duration_ms < 0:
        raise ValueError("duration_ms 不能小于 0")

    async with session_factory() as session:
        async with session.begin():
            statement = (
                update(DiagnosisRun)
                .where(
                    DiagnosisRun.id == diagnosis_run_id,
                    DiagnosisRun.status == "running",
                )
                .values(
                    status="failed",
                    result_summary=None,
                    error_message=error_message,
                    finished_at=func.now(),
                    duration_ms=duration_ms,
                )
                .returning(DiagnosisRun.id)
            )

            updated_id = await session.scalar(statement)

            if updated_id is None:
                raise ValueError(f"诊断记录不存在或状态已结束：{diagnosis_run_id}")


async def create_pending_tool_log(
    session_factory: BusinessSessionFactory,
    *,
    diagnosis_run_id: UUID,
    tool_call_id: str,
    tool_name: str,
    input_payload: dict[str, Any],
) -> int:
    """为模型提出的工具请求创建待执行日志。"""
    if not tool_call_id:
        raise ValueError("tool_call_id 不能为空")

    if not tool_name:
        raise ValueError("tool_name 不能为空")

    async with session_factory() as session:
        async with session.begin():
            tool_log = ToolExecutionLog(
                diagnosis_run_id=diagnosis_run_id,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                status="pending",
                input_payload=input_payload,
            )
            session.add(tool_log)

            # 执行 INSERT，取得数据库生成的自增主键。
            await session.flush()
            tool_log_id = tool_log.id

    return tool_log_id


async def mark_tool_running(
    session_factory: BusinessSessionFactory,
    *,
    tool_log_id: int,
) -> None:
    """把待执行的工具日志更新为运行中。"""
    if tool_log_id <= 0:
        raise ValueError("tool_log_id 必须大于 0")

    async with session_factory() as session:
        async with session.begin():
            statement = (
                update(ToolExecutionLog)
                .where(
                    ToolExecutionLog.id == tool_log_id,
                    ToolExecutionLog.status == "pending",
                )
                .values(
                    status="running",
                    started_at=func.now(),
                )
                .returning(ToolExecutionLog.id)
            )

            updated_id = await session.scalar(statement)

            if updated_id is None:
                raise ValueError(f"工具日志不存在或状态不是 pending：{tool_log_id}")


async def mark_tool_succeeded(
    session_factory: BusinessSessionFactory,
    *,
    tool_log_id: int,
    result_summary: dict[str, Any],
    duration_ms: int,
) -> None:
    """把运行中的工具日志更新为成功状态。"""
    if tool_log_id <= 0:
        raise ValueError("tool_log_id 必须大于 0")

    if duration_ms < 0:
        raise ValueError("duration_ms 不能小于 0")

    async with session_factory() as session:
        async with session.begin():
            statement = (
                update(ToolExecutionLog)
                .where(
                    ToolExecutionLog.id == tool_log_id,
                    ToolExecutionLog.status == "running",
                )
                .values(
                    status="succeeded",
                    result_summary=result_summary,
                    error_message=None,
                    finished_at=func.now(),
                    duration_ms=duration_ms,
                )
                .returning(ToolExecutionLog.id)
            )

            updated_id = await session.scalar(statement)

            if updated_id is None:
                raise ValueError(f"工具日志不存在或状态不是 running：{tool_log_id}")


async def mark_tool_failed(
    session_factory: BusinessSessionFactory,
    *,
    tool_log_id: int,
    error_message: str,
    duration_ms: int,
) -> None:
    """把运行中的工具日志更新为失败状态。"""
    if tool_log_id <= 0:
        raise ValueError("tool_log_id 必须大于 0")

    if not error_message:
        raise ValueError("error_message 不能为空")

    if duration_ms < 0:
        raise ValueError("duration_ms 不能小于 0")

    async with session_factory() as session:
        async with session.begin():
            statement = (
                update(ToolExecutionLog)
                .where(
                    ToolExecutionLog.id == tool_log_id,
                    ToolExecutionLog.status == "running",
                )
                .values(
                    status="failed",
                    result_summary=None,
                    error_message=error_message,
                    finished_at=func.now(),
                    duration_ms=duration_ms,
                )
                .returning(ToolExecutionLog.id)
            )

            updated_id = await session.scalar(statement)

            if updated_id is None:
                raise ValueError(f"工具日志不存在或状态不是 running：{tool_log_id}")


async def mark_unfinished_execution_failed(
    session_factory: BusinessSessionFactory,
    *,
    diagnosis_run_id: UUID,
    error_message: str,
) -> None:
    """将诊断及其未结束的工具日志原子更新为失败。"""
    if not error_message:
        raise ValueError("error_message 不能为空")

    async with session_factory() as session:
        async with session.begin():
            tool_duration_ms = func.greatest(
                0,
                cast(
                    func.extract(
                        "epoch",
                        func.now()
                        - func.coalesce(
                            ToolExecutionLog.started_at,
                            ToolExecutionLog.created_at,
                        ),
                    )
                    * 1000,
                    Integer,
                ),
            )

            await session.execute(
                update(ToolExecutionLog)
                .where(
                    ToolExecutionLog.diagnosis_run_id == diagnosis_run_id,
                    ToolExecutionLog.status.in_(("pending", "running")),
                )
                .values(
                    status="failed",
                    result_summary=None,
                    error_message=error_message,
                    finished_at=func.now(),
                    duration_ms=tool_duration_ms,
                )
            )

            diagnosis_duration_ms = func.greatest(
                0,
                cast(
                    func.extract(
                        "epoch",
                        func.now() - DiagnosisRun.started_at,
                    )
                    * 1000,
                    Integer,
                ),
            )

            await session.execute(
                update(DiagnosisRun)
                .where(
                    DiagnosisRun.id == diagnosis_run_id,
                    DiagnosisRun.status == "running",
                )
                .values(
                    status="failed",
                    result_summary=None,
                    error_message=error_message,
                    finished_at=func.now(),
                    duration_ms=diagnosis_duration_ms,
                )
            )
