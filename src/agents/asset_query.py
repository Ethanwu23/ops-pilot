from psycopg import AsyncConnection
from psycopg.rows import dict_row

from memory.postgres import get_postgres_connection_string, validate_postgres_config
from schema.assets import AssetLookupInput, AssetResponse


async def lookup_asset(asset_id: int) -> AssetResponse | None:
    """按资产 ID 查询登记信息，未找到时返回 None。"""
    # 在建立连接前校验输入和数据库配置。
    request = AssetLookupInput(asset_id=asset_id)
    validate_postgres_config()

    async with await AsyncConnection.connect(
        get_postgres_connection_string(),
        row_factory=dict_row,
        autocommit=True,
        connect_timeout=5,
        options="-c default_transaction_read_only=on",
    ) as conn:
        async with conn.cursor() as cursor:
            # 参数单独传递给驱动，不拼接到 SQL 中。
            await cursor.execute(
                """
                SELECT id, name, asset_type, ip_address, status, created_at
                FROM public.assets_asset
                WHERE id = %s
                """,
                (request.asset_id,),
            )
            row = await cursor.fetchone()

    # 此时连接已关闭；仅无记录时返回 None，其他错误向上传递。
    if row is None:
        return None
    return AssetResponse.model_validate(row)
