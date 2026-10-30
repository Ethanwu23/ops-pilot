from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from pydantic import ConfigDict

from agents.context import AgentContext
from agents.metrics_query import lookup_metrics
from persistence.authorization import require_asset_access
from schema.metrics import MetricsLookupInput


class MetricsToolInput(MetricsLookupInput):
    """保留业务参数校验，并声明后端注入的运行上下文。"""

    runtime: ToolRuntime[AgentContext]

    model_config = ConfigDict(arbitrary_types_allowed=True)


@tool(args_schema=MetricsToolInput)
async def get_metrics(
    asset_id: int,
    runtime: ToolRuntime[AgentContext],
) -> dict[str, object]:
    """查询当前用户有权访问资产的本地测试指标。"""
    await require_asset_access(
        runtime.context.business_session_factory,
        user_id=runtime.context.user_id,
        asset_id=asset_id,
    )

    result = await lookup_metrics(
        asset_id,
        client=runtime.context.metrics_client,
        semaphore=runtime.context.metrics_semaphore,
    )

    return result.model_dump(mode="json")
