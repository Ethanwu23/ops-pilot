from sqlalchemy import select

from persistence.database import BusinessSessionFactory
from persistence.models import AssetAccessGrant


class AssetAccessDeniedError(Exception):
    """表示当前用户无权读取目标资产。"""


async def require_asset_access(
    session_factory: BusinessSessionFactory,
    *,
    user_id: str,
    asset_id: int,
) -> None:
    """确认用户拥有资产读取权限，否则拒绝访问。"""
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("user_id 必须是非空字符串")

    if isinstance(asset_id, bool) or not isinstance(asset_id, int) or asset_id <= 0:
        raise ValueError("asset_id 必须是严格正整数")

    statement = select(AssetAccessGrant.user_id).where(
        AssetAccessGrant.user_id == user_id,
        AssetAccessGrant.asset_id == asset_id,
    )
    async with session_factory() as session:
        async with session.begin():
            granted_user_id = await session.scalar(statement)

        if granted_user_id is None:
            raise AssetAccessDeniedError("资产不存在或无权访问")
