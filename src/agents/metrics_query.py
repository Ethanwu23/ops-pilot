import asyncio
import httpx

from schema.metrics import MetricsResponse, MetricsLookupInput

async def lookup_metrics(
    asset_id: int,
    *,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
) -> MetricsResponse:
    """查询本地测试指标，校验结果并确认资产 ID 一致。"""
    request = MetricsLookupInput(asset_id=asset_id)

    async with semaphore:
        response = await client.get(
            f"/metrics/{request.asset_id}"
        )
        response.raise_for_status()

    # 响应正文已读取，释放并发名额后再解析和校验。
    result = MetricsResponse.model_validate(response.json())

    if result.asset_id != request.asset_id:
        raise ValueError(
            f"指标响应资产 ID 不匹配："
            f"请求 {request.asset_id}，返回 {result.asset_id}"
        )
    return result
