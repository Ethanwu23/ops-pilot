import asyncio
import json
import logging
import os
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import httpx
import pytest
import pytest_asyncio
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from openai import OpenAIError
from sqlalchemy import delete, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine

from agents.diagnostic_graph import builder
from persistence import repositories
from persistence.authorization import AssetAccessDeniedError, require_asset_access
from persistence.database import create_business_session_factory, get_business_database_url
from persistence.models import AssetAccessGrant, DiagnosisRun, ToolExecutionLog
from schema.auth import AuthenticatedUser
from schema.diagnosis import AnalysisResult, Hypothesis
from service import app
from service.auth import get_authenticated_user


@pytest_asyncio.fixture
async def diagnostic_case(test_client, monkeypatch):
    """保留真实图和工具，仅替换模型、网络及业务持久化边界。"""
    requests = []
    response_data = {
        "asset_id": 1,
        "source": "local_test_metrics",
        "is_test_data": True,
        "cpu_percent": 37.5,
        "memory_percent": 62.0,
    }
    case = SimpleNamespace(
        response_data=response_data,
        http_error=None,
        http_status=200,
        before_http=None,
        requests=requests,
    )

    async def metrics_endpoint(request):
        requests.append(request)
        if case.before_http is not None:
            await case.before_http()
        if case.http_error is not None:
            raise case.http_error
        return httpx.Response(case.http_status, json=case.response_data)

    async def create_run(session_factory, **kwargs):
        return kwargs["diagnosis_run_id"]

    repositories = {
        "create_diagnosis_run": AsyncMock(side_effect=create_run),
        "create_pending_tool_log": AsyncMock(return_value=1),
        "mark_tool_running": AsyncMock(),
        "mark_tool_succeeded": AsyncMock(),
        "mark_tool_failed": AsyncMock(),
        "mark_diagnosis_succeeded": AsyncMock(),
        "mark_diagnosis_failed": AsyncMock(),
        "mark_unfinished_execution_failed": AsyncMock(),
        "get_diagnosis_status": AsyncMock(return_value="running"),
    }
    monkeypatch.setattr(
        "agents.diagnostic_state.create_diagnosis_run", repositories["create_diagnosis_run"]
    )
    for name in (
        "create_pending_tool_log", "mark_tool_running", "mark_tool_succeeded",
        "mark_tool_failed", "mark_diagnosis_succeeded", "mark_diagnosis_failed",
    ):
        monkeypatch.setattr(f"agents.diagnostic_nodes.{name}", repositories[name])
    for name in ("mark_unfinished_execution_failed", "get_diagnosis_status"):
        monkeypatch.setattr(f"service.service.{name}", repositories[name])

    case.authorize = AsyncMock()
    monkeypatch.setattr("agents.metrics_tool.require_asset_access", case.authorize)
    case.repositories = repositories
    case.analysis = AnalysisResult(
        summary="仅有本地测试指标，无法判断真实健康状态。",
        hypotheses=[],
        missing_information=["真实指标及时间范围"],
    )
    case.structured = SimpleNamespace(ainvoke=AsyncMock(return_value=case.analysis))
    case.model = Mock()
    case.model.bind_tools.return_value = SimpleNamespace(
        ainvoke=AsyncMock(return_value=AIMessage(
            content="",
            tool_calls=[{"name": "get_metrics", "args": {"asset_id": 1}, "id": "call-test"}],
        )),
    )
    case.model.with_structured_output.return_value = case.structured
    monkeypatch.setattr("agents.diagnostic_nodes.get_model", Mock(return_value=case.model))
    case.graph = builder.compile(checkpointer=MemorySaver())
    monkeypatch.setattr("service.service.get_agent", Mock(return_value=case.graph))
    case.thread_id = "isolated-diagnostic-test"

    async with httpx.AsyncClient(
        base_url="http://metrics.test", transport=httpx.MockTransport(metrics_endpoint),
        trust_env=False,
    ) as metrics_client:
        monkeypatch.setattr(app.state, "metrics_client", metrics_client)
        monkeypatch.setattr(app.state, "metrics_semaphore", asyncio.Semaphore(1))
        async with httpx.AsyncClient(
            base_url="http://service.test", transport=httpx.ASGITransport(app=app),
            trust_env=False,
        ) as client:
            case.client = client
            yield case


async def send(case, message="查询资产 1 的测试指标", path="invoke"):
    """通过真实接口执行请求，不直接调用节点函数。"""
    return await case.client.post(
        f"/chatbot/{path}",
        json={"message": message, "thread_id": case.thread_id, "stream_tokens": False},
    )


def sse_events(response):
    return [
        json.loads(line.removeprefix("data: "))
        for line in response.iter_lines()
        if line.startswith("data: ")
    ]


def set_tool_args(case, args):
    case.model.bind_tools.return_value.ainvoke.return_value = AIMessage(
        content="", tool_calls=[{"name": "get_metrics", "args": args, "id": "call-test"}],
    )


@pytest.mark.asyncio
async def test_direct_answer_uses_no_tools(diagnostic_case):
    case = diagnostic_case
    case.model.bind_tools.return_value.ainvoke.return_value = AIMessage(content="你好")
    response = await send(case, message="你好，不要调用工具")
    assert response.status_code == 200
    assert response.json()["content"] == "你好"
    assert not case.requests
    case.authorize.assert_not_awaited()
    case.repositories["mark_diagnosis_succeeded"].assert_awaited_once()
    case.repositories["create_pending_tool_log"].assert_not_awaited()


@pytest.mark.asyncio
async def test_real_graph_pauses_and_resumes_without_repeating_tool(diagnostic_case):
    case = diagnostic_case
    first = await send(case, path="stream")
    assert first.status_code == 200
    frames = sse_events(first)
    assert not any(frame["type"] == "error" for frame in frames)
    assert frames[-2]["type"] == "interrupt"
    assert frames[-1]["status"] == "paused"
    diagnosis_id = frames[-2]["diagnosis_run_id"]
    case.structured.ainvoke.assert_not_awaited()
    case.repositories["mark_diagnosis_succeeded"].assert_not_awaited()

    resumed = await send(case, message="continue", path="stream")
    frames = sse_events(resumed)
    assert frames[-1]["status"] == "succeeded"
    assert not any(frame["type"] in {"error", "interrupt"} for frame in frames)
    result = json.loads(next(frame["content"]["content"] for frame in frames if frame["type"] == "message"))
    assert result["status"] == "succeeded"
    assert result["evidence"][0]["data"] == case.response_data
    assert result["analysis"]["hypotheses"] == []
    assert result["evidence"][0]["is_test_data"] is True
    assert len(case.requests) == 1
    assert case.requests[0].url.path == "/metrics/1"
    case.repositories["create_diagnosis_run"].assert_awaited_once()
    case.repositories["create_pending_tool_log"].assert_awaited_once()
    saved = case.repositories["mark_diagnosis_succeeded"]
    saved.assert_awaited_once()
    assert str(saved.await_args.kwargs["diagnosis_run_id"]) == diagnosis_id
    assert frames[-1]["run_id"] != diagnosis_id


@pytest.mark.asyncio
@pytest.mark.parametrize("asset_id", [0, -1, "1", True])
async def test_invalid_tool_input_never_reaches_authorization_or_network(diagnostic_case, asset_id):
    case = diagnostic_case
    set_tool_args(case, {"asset_id": asset_id})
    response = await send(case)
    assert response.status_code == 200
    assert json.loads(response.json()["content"])["status"] == "failed"
    assert not case.requests
    case.authorize.assert_not_awaited()
    case.structured.ainvoke.assert_not_awaited()
    case.repositories["mark_tool_failed"].assert_awaited_once()
    case.repositories["mark_diagnosis_failed"].assert_awaited_once()
    case.repositories["mark_diagnosis_succeeded"].assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("fault, expected", [
    ("connection", "指标服务调用失败"),
    ("timeout", "工具调用超时"),
    ("server", "指标服务调用失败"),
    ("wrong_asset", "工具返回的数据不符合约定"),
])
async def test_tool_fault_produces_failed_business_result(diagnostic_case, fault, expected):
    case = diagnostic_case
    if fault == "connection":
        case.http_error = httpx.ConnectError("测试连接故障")
    elif fault == "timeout":
        case.http_error = httpx.ReadTimeout("测试读取超时")
    elif fault == "server":
        case.http_status = 500
    else:
        case.response_data = {**case.response_data, "asset_id": 2}
    response = await send(case, path="stream")
    assert response.status_code == 200
    frames = sse_events(response)
    assert frames[-1]["status"] == "failed"
    assert not any(frame["type"] == "interrupt" for frame in frames)
    assert len(case.requests) == 1
    assert case.repositories["mark_tool_failed"].await_args.kwargs["error_message"] == expected
    case.repositories["mark_diagnosis_failed"].assert_awaited_once()
    case.repositories["mark_diagnosis_succeeded"].assert_not_awaited()
    case.structured.ainvoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_asset_access_denial_stops_network_call(diagnostic_case):
    case = diagnostic_case
    case.authorize.side_effect = AssetAccessDeniedError("测试无权访问")
    response = await send(case)
    assert json.loads(response.json()["content"])["status"] == "failed"
    assert not case.requests
    assert case.authorize.await_args.kwargs["user_id"] == "user-123"
    assert case.authorize.await_args.kwargs["asset_id"] == 1
    assert case.repositories["mark_tool_failed"].await_args.kwargs["error_message"] == "资产不存在或无权访问"
    case.structured.ainvoke.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["wrong_type", "unknown_evidence", "provider"])
async def test_analysis_fault_cannot_be_saved_as_success(diagnostic_case, fault):
    case = diagnostic_case
    await send(case)
    if fault == "wrong_type":
        case.structured.ainvoke.return_value = {"summary": "未校验的字典"}
    elif fault == "unknown_evidence":
        case.structured.ainvoke.return_value = AnalysisResult(
            summary="测试推测", hypotheses=[Hypothesis(
                statement="可能异常", evidence_ids=["unknown-call"], verification="查询真实指标",
            )], missing_information=[],
        )
    else:
        case.structured.ainvoke.side_effect = OpenAIError("测试供应商故障")
    response = await send(case, message="continue")
    result = json.loads(response.json()["content"])
    assert result["status"] == "failed"
    assert "测试供应商故障" not in result["error"]
    assert len(case.requests) == 1
    case.repositories["mark_diagnosis_failed"].assert_awaited_once()
    case.repositories["mark_diagnosis_succeeded"].assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("save_point", ["tool", "diagnosis"])
async def test_save_failure_never_emits_success(diagnostic_case, save_point):
    case = diagnostic_case
    if save_point == "tool":
        case.repositories["mark_tool_succeeded"].side_effect = RuntimeError("测试工具保存故障")
    else:
        await send(case)
        case.repositories["mark_diagnosis_succeeded"].side_effect = RuntimeError("测试诊断保存故障")
    response = await send(case, message="continue" if save_point == "diagnosis" else "查询指标", path="stream")
    frames = sse_events(response)
    assert response.status_code == 200
    assert frames[-2]["type"] == "error"
    assert frames[-1]["status"] == "error"
    assert not any(frame["type"] == "message" and '"status": "succeeded"' in frame["content"]["content"] for frame in frames)
    assert len(case.requests) == 1
    case.repositories["mark_unfinished_execution_failed"].assert_awaited_once()


@pytest_asyncio.fixture
async def postgres_case(diagnostic_case, monkeypatch):
    """连接显式指定的独立测试库，仓储和授权使用真实实现。"""
    target = os.environ.get("TEST_POSTGRES_DB")
    if target != "ops_pilot_test":
        pytest.fail("必须显式设置 TEST_POSTGRES_DB=ops_pilot_test，不允许使用学习库")
    case = diagnostic_case
    url = get_business_database_url().set(database=target)
    engine = create_async_engine(url, pool_size=2, max_overflow=0, pool_timeout=2, hide_parameters=True)
    case.engine = engine
    case.session_factory = create_business_session_factory(engine)
    case.owner = f"test-{uuid4().hex}"
    case.thread_id = f"test-{uuid4().hex}"
    try:
        async with engine.connect() as connection:
            assert await connection.scalar(text("SELECT current_database()")) == target
        monkeypatch.setattr(app.state, "business_session_factory", case.session_factory)
        monkeypatch.setitem(
            app.dependency_overrides, get_authenticated_user,
            lambda: AuthenticatedUser(user_id=case.owner),
        )
        monkeypatch.setattr("agents.diagnostic_state.create_diagnosis_run", repositories.create_diagnosis_run)
        for name in (
            "create_pending_tool_log", "mark_tool_running", "mark_tool_succeeded",
            "mark_tool_failed", "mark_diagnosis_succeeded", "mark_diagnosis_failed",
        ):
            monkeypatch.setattr(f"agents.diagnostic_nodes.{name}", getattr(repositories, name))
        for name in ("mark_unfinished_execution_failed", "get_diagnosis_status"):
            monkeypatch.setattr(f"service.service.{name}", getattr(repositories, name))
        monkeypatch.setattr("agents.metrics_tool.require_asset_access", require_asset_access)
        async with case.session_factory.begin() as session:
            session.add(AssetAccessGrant(user_id=case.owner, asset_id=1))
        yield case
    finally:
        # 仅清理本用例的记录，保留测试库迁移和其他用例数据。
        async with case.session_factory.begin() as session:
            await session.execute(delete(DiagnosisRun).where(DiagnosisRun.thread_id == case.thread_id))
            await session.execute(delete(AssetAccessGrant).where(AssetAccessGrant.user_id == case.owner))
        await engine.dispose()


async def read_business_rows(case):
    async with case.session_factory() as session:
        run = (await session.scalars(select(DiagnosisRun).where(DiagnosisRun.thread_id == case.thread_id))).one()
        logs = (await session.scalars(select(ToolExecutionLog).where(ToolExecutionLog.diagnosis_run_id == run.id))).all()
    return run, logs


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_postgres_success_uses_short_transactions_and_preserves_tool_result(postgres_case):
    case = postgres_case

    async def inspect_wait_boundary():
        # 执行外部 HTTP 前，授权和日志短事务应已归还连接。
        assert case.engine.pool.checkedout() == 0
        run, logs = await read_business_rows(case)
        assert run.status == "running"
        assert len(logs) == 1 and logs[0].status == "running"

    case.before_http = inspect_wait_boundary
    response = await send(case, path="stream")
    assert sse_events(response)[-1]["status"] == "paused"
    run, logs = await read_business_rows(case)
    original_id = run.id
    original_log_id = logs[0].id
    assert run.status == "running" and run.finished_at is None and run.duration_ms is None
    assert logs[0].status == "succeeded" and logs[0].result_summary == case.response_data
    assert case.engine.pool.checkedout() == 0
    response = await send(case, message="continue", path="stream")
    assert sse_events(response)[-1]["status"] == "succeeded"
    run, logs = await read_business_rows(case)
    assert run.id == original_id and run.status == "succeeded"
    assert run.finished_at is not None and run.duration_ms >= 0
    assert json.loads(run.result_summary)["summary"] == case.analysis.summary
    assert len(logs) == 1 and logs[0].id == original_log_id
    assert len(case.requests) == 1
    async with case.engine.connect() as connection:
        measured = await connection.scalar(text(
            "SELECT ROUND(EXTRACT(EPOCH FROM (finished_at-started_at))*1000)::bigint "
            "FROM diagnosis_runs WHERE id=:id"
        ), {"id": run.id})
    assert run.duration_ms == measured
    assert case.engine.pool.checkedout() == 0


@pytest.mark.postgres
@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["invalid", "connection", "timeout", "server", "wrong_asset", "denied"])
async def test_postgres_failed_execution_has_consistent_records(postgres_case, fault):
    case = postgres_case
    if fault == "invalid":
        set_tool_args(case, {"asset_id": 0})
    elif fault == "connection":
        case.http_error = httpx.ConnectError("测试连接故障")
    elif fault == "timeout":
        case.http_error = httpx.ReadTimeout("测试读取超时")
    elif fault == "server":
        case.http_status = 500
    elif fault == "wrong_asset":
        case.response_data = {**case.response_data, "asset_id": 2}
    else:
        async with case.session_factory.begin() as session:
            await session.execute(delete(AssetAccessGrant).where(AssetAccessGrant.user_id == case.owner))
    response = await send(case, path="stream")
    assert response.status_code == 200
    assert sse_events(response)[-1]["status"] == "failed"
    run, logs = await read_business_rows(case)
    assert run.status == "failed" and run.finished_at is not None
    assert run.result_summary is None and run.error_message
    assert run.duration_ms >= 0
    assert len(logs) == 1 and logs[0].status == "failed"
    assert logs[0].error_message and logs[0].finished_at is not None
    assert logs[0].result_summary is None
    assert len(case.requests) == (0 if fault in {"invalid", "denied"} else 1)
    case.structured.ainvoke.assert_not_awaited()
    assert case.engine.pool.checkedout() == 0


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_postgres_final_save_failure_rolls_back_and_reports_error(postgres_case):
    case = postgres_case
    await send(case)
    run, _ = await read_business_rows(case)
    constraint = f"test_reject_success_{uuid4().hex}"
    # 仅对本测试诊断拒绝成功写入，制造真实 PostgreSQL 约束异常。
    async with case.engine.begin() as connection:
        await connection.execute(text(
            f"ALTER TABLE diagnosis_runs ADD CONSTRAINT {constraint} "
            f"CHECK (id <> '{run.id}'::uuid OR status <> 'succeeded')"
        ))
    try:
        response = await send(case, message="continue", path="stream")
        frames = sse_events(response)
        assert frames[-2]["type"] == "error" and frames[-1]["status"] == "error"
        run, logs = await read_business_rows(case)
        assert run.status == "failed" and run.result_summary is None
        assert run.error_message == "执行发生未处理异常"
        assert run.finished_at is not None
        assert len(logs) == 1 and logs[0].status == "succeeded"
        assert logs[0].result_summary == case.response_data
        assert len(case.requests) == 1
        assert case.engine.pool.checkedout() == 0
    finally:
        async with case.engine.begin() as connection:
            await connection.execute(text(f"ALTER TABLE diagnosis_runs DROP CONSTRAINT {constraint}"))


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_postgres_failure_finalizer_is_atomic(postgres_case):
    case = postgres_case
    run_id = uuid4()
    await repositories.create_diagnosis_run(
        case.session_factory, diagnosis_run_id=run_id, user_id=case.owner,
        thread_id=case.thread_id, question="测试原子收尾",
    )
    log_id = await repositories.create_pending_tool_log(
        case.session_factory, diagnosis_run_id=run_id, tool_call_id="atomic-call",
        tool_name="get_metrics", input_payload={"asset_id": 1},
    )
    await repositories.mark_tool_running(case.session_factory, tool_log_id=log_id)
    constraint = f"test_reject_failure_{uuid4().hex}"
    async with case.engine.begin() as connection:
        await connection.execute(text(
            f"ALTER TABLE diagnosis_runs ADD CONSTRAINT {constraint} "
            f"CHECK (id <> '{run_id}'::uuid OR status <> 'failed')"
        ))
    try:
        with pytest.raises(IntegrityError):
            await repositories.mark_unfinished_execution_failed(
                case.session_factory, diagnosis_run_id=run_id, error_message="测试收尾异常",
            )
        run, logs = await read_business_rows(case)
        # 工具更新先执行，但诊断更新失败后应随同一事务一起回滚。
        assert run.status == "running" and run.finished_at is None
        assert len(logs) == 1 and logs[0].status == "running"
        assert logs[0].finished_at is None
        assert case.engine.pool.checkedout() == 0
    finally:
        async with case.engine.begin() as connection:
            await connection.execute(text(f"ALTER TABLE diagnosis_runs DROP CONSTRAINT {constraint}"))


@pytest.mark.asyncio
async def test_trace_logs_link_resume_without_logging_question(diagnostic_case, caplog):
    """请求、诊断和工具标识可关联，但不记录问题及工具原始数据。"""
    case = diagnostic_case
    caplog.set_level(logging.INFO, logger="service.service")
    caplog.set_level(logging.INFO, logger="agents.diagnostic_nodes")
    private_question = "这是仅用于检查日志的私密问题-marker-7812"
    first = await send(case, message=private_question, path="stream")
    first_frames = sse_events(first)
    diagnosis_id = first_frames[-2]["diagnosis_run_id"]
    second = await send(case, message="continue", path="stream")
    second_frames = sse_events(second)
    first_request_id = first_frames[-1]["run_id"]
    second_request_id = second_frames[-1]["run_id"]
    assert first_request_id != second_request_id
    messages = [record.getMessage() for record in caplog.records if record.name in {"service.service", "agents.diagnostic_nodes"}]
    assert any(f"run_id={first_request_id}" in message and "mode=new" in message for message in messages)
    assert any(f"run_id={second_request_id}" in message and f"diagnosis_run_id={diagnosis_id}" in message and "mode=resume" in message for message in messages)
    assert any(f"diagnosis_run_id={diagnosis_id}" in message and "tool_call_id=call-test" in message for message in messages)
    assert any("流式执行结束" in message and "status=paused" in message for message in messages)
    assert any("流式执行结束" in message and "status=succeeded" in message for message in messages)
    assert private_question not in "\n".join(messages)
    assert "cpu_percent" not in "\n".join(messages)


@pytest.mark.asyncio
async def test_unhandled_error_logs_type_not_exception_payload(diagnostic_case, caplog):
    """外部异常可能携带敏感文本，普通错误日志仅记录类型和标识。"""
    case = diagnostic_case
    case.repositories["mark_tool_succeeded"].side_effect = RuntimeError("private-exception-marker-4621")
    response = await send(case, path="stream")
    frames = sse_events(response)
    assert frames[-1]["status"] == "error"
    logs = [record.getMessage() for record in caplog.records if record.name == "service.service"]
    assert any("error_type=RuntimeError" in message and f"run_id={frames[-1]['run_id']}" in message for message in logs)
    assert "private-exception-marker-4621" not in "\n".join(logs)


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", [False, True], ids=["success", "timeout"])
async def test_tool_timing_logs_include_times_without_sensitive_data(
    diagnostic_case, caplog, monkeypatch, fault,
):
    """用户主导验收：起止时间齐全，成功失败耗时一致且不输出敏感原文。"""
    case = diagnostic_case
    caplog.set_level(logging.INFO, logger="agents.diagnostic_nodes")
    caplog.set_level(logging.INFO, logger="service.service")
    secrets = ["test-token-marker-917", "test-key-marker-284", "test-password-marker-563"]
    case.client.headers["Authorization"] = f"Bearer {secrets[0]}"
    if fault:
        case.http_error = httpx.ReadTimeout(" ".join(secrets))
    # 固定单调时钟间隔，验证记录的是实际计算值而非仅存在一个字段。
    clock_values = iter([100.0, 100.125])
    monkeypatch.setattr("agents.diagnostic_nodes.perf_counter", lambda: next(clock_values))
    response = await send(case, message="测试问题 " + " ".join(secrets), path="stream")
    frames = sse_events(response)
    assert frames[-1]["status"] == ("failed" if fault else "paused")
    tool_logs = [record.getMessage() for record in caplog.records if record.name == "agents.diagnostic_nodes"]
    start_logs = [message for message in tool_logs if message.startswith("工具开始执行")]
    finish_logs = [message for message in tool_logs if message.startswith("工具执行结束")]
    assert len(start_logs) == len(finish_logs) == 1
    start_fields = dict(item.split("=", 1) for item in start_logs[0].split("：", 1)[1].split() if "=" in item)
    finish_fields = dict(item.split("=", 1) for item in finish_logs[0].split("：", 1)[1].split() if "=" in item)
    assert start_fields["started_at"] == finish_fields["started_at"]
    assert start_fields["tool_call_id"] == finish_fields["tool_call_id"] == "call-test"
    assert start_fields["diagnosis_run_id"] == finish_fields["diagnosis_run_id"]
    assert finish_fields["status"] == ("failed" if fault else "succeeded")
    assert int(finish_fields["duration_ms"]) == 125
    for field in ("started_at", "finished_at"):
        timestamp = datetime.fromisoformat(finish_fields[field])
        assert timestamp.utcoffset().total_seconds() == 0
    logs = "\n".join(record.getMessage() for record in caplog.records if record.name in {"agents.diagnostic_nodes", "service.service"})
    assert all(secret not in logs for secret in secrets)
