from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

from agents.context import AgentContext
from agents.metrics_query import lookup_metrics
from schema.metrics import MetricsLookupInput
from pydantic import ConfigDict

class MetricsToolInput(MetricsLookupInput):
    """保留业务参数校验，并声明后端注入的运行上下文。"""

    runtime: ToolRuntime[AgentContext]

    model_config = ConfigDict(arbitrary_types_allowed=True)

@tool(args_schema=MetricsToolInput)
async def get_metrics(
    asset_id: int,
    runtime: ToolRuntime[AgentContext],
) -> dict[str, object]:
    """按资产 ID 查询本地测试指标；数据为模拟值，不代表实时运行状态。"""
    result = await lookup_metrics(
        asset_id,
        client=runtime.context.metrics_client,
        semaphore=runtime.context.metrics_semaphore,
    )
    return result.model_dump(mode="json")
