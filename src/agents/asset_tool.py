from langchain_core.tools import tool

from agents.asset_query import lookup_asset
from schema.assets import AssetLookupInput


@tool(args_schema=AssetLookupInput)
async def get_asset(asset_id: int) -> dict[str, object]:
    """按资产 ID 查询本地学习库中的资产登记信息，不代表实时健康状态。"""
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

