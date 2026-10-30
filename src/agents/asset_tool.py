from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from pydantic import ConfigDict

from agents.asset_query import lookup_asset
from agents.context import AgentContext
from persistence.authorization import require_asset_access
from schema.assets import AssetLookupInput


class AssetToolInput(AssetLookupInput):
    """保留资产参数校验，并声明服务端注入的运行上下文。"""

    runtime: ToolRuntime[AgentContext]

    model_config = ConfigDict(arbitrary_types_allowed=True)


@tool(args_schema=AssetToolInput)
async def get_asset(
    asset_id: int,
    runtime: ToolRuntime[AgentContext],
) -> dict[str, object]:
    """查询当前用户有权访问的资产登记信息。"""
    await require_asset_access(
        runtime.context.business_session_factory,
        user_id=runtime.context.user_id,
        asset_id=asset_id,
    )

    asset = await lookup_asset(asset_id)

    if asset is None:
        return {
            "source": "local_learning_database",
            "found": False,
            "asset": None,
        }

    return {
        "source": "local_learning_database",
        "found": True,
        "asset": asset.model_dump(mode="json"),
    }
