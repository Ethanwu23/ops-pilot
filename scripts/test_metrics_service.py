import asyncio
from typing import Annotated
import logging
from uuid import uuid4

from fastapi import FastAPI, Path, Query
from schema.metrics import MetricsResponse

logger = logging.getLogger("uvicorn.error")
app = FastAPI(title="本地测试指标服务")

@app.get("/metrics/{asset_id}", response_model=MetricsResponse)
async def get_metrics(
    asset_id: Annotated[int, Path(gt=0)],
    delay_seconds: Annotated[float, Query(ge=0, le=10)] = 0
) -> MetricsResponse:

    """返回固定测试指标，不代表资产的真实运行状态。"""
    request_id = uuid4().hex[:8]
    logger.info(
        "指标请求开始 request_id=%s asset_id=%s delay=%s",
        request_id,
        asset_id,
        delay_seconds,
    )

    try:
        await asyncio.sleep(delay_seconds)
        result = MetricsResponse(
            asset_id=asset_id,
            source="local_test_metrics",
            is_test_data=True,
            cpu_percent=37.5,
            memory_percent=62.0,
        )
        logger.info("指标数据生成完成 request_id=%s", request_id)
        return result
    except asyncio.CancelledError:
        logger.info("指标处理协程收到取消 request_id=%s", request_id)
        raise
    finally:
        logger.info("指标处理函数退出 request_id=%s", request_id)
