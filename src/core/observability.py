from typing import Any
from uuid import UUID

from langchain_core.tracers.langchain import LangChainTracer
from langchain_core.tracers.schemas import Run
from langgraph.errors import GraphInterrupt
from langsmith import Client

from core.settings import settings

TRACE_METADATA_KEYS = {
    "agent_id", "thread_id", "diagnosis_run_id", "request_run_id", "request_mode",
    "business_status", "langgraph_node", "langgraph_step", "ls_provider",
    "ls_model_name", "ls_model_type", "ls_temperature", "ls_max_tokens",
}


def filter_trace_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """只上传追踪必需字段，排除用户身份及可配置资源对象。"""
    return {key: value for key, value in metadata.items() if key in TRACE_METADATA_KEYS}


def create_tracing_client() -> Client | None:
    """显式使用 Settings，避免只写在本地文件中的配置未被 SDK 读取。"""
    if not settings.LANGSMITH_TRACING:
        return None
    return Client(
        api_url=settings.LANGSMITH_ENDPOINT,
        timeout_ms=3000,
        api_key=settings.LANGSMITH_API_KEY.get_secret_value(),
        workspace_id=settings.LANGSMITH_WORKSPACE_ID,
        hide_inputs=settings.LANGSMITH_HIDE_INPUTS,
        hide_outputs=settings.LANGSMITH_HIDE_OUTPUTS,
        hide_metadata=filter_trace_metadata,
    )


class DiagnosisTracer(LangChainTracer):
    """保留框架追踪，区分预期中断并隐藏异常原文。"""

    def on_chain_error(
        self, error: BaseException, *, run_id: UUID,
        inputs: dict[str, Any] | None = None, **kwargs: Any,
    ) -> Run:
        # interrupt 是可恢复的流程控制，不是业务执行失败。
        if isinstance(error, GraphInterrupt):
            return super().on_chain_end(
                {"interrupted": True}, run_id=run_id, inputs=inputs,
            )
        return super().on_chain_error(error, run_id=run_id, inputs=inputs, **kwargs)

    def _update_run_single(self, run: Run) -> None:
        # 异常原文可能包含外部响应内容；远端仅保留异常发生标记。
        if run.error:
            run.error = "执行异常，详情请根据关联 ID 查询本地日志"
        super()._update_run_single(run)
