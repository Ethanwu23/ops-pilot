import asyncio
from dataclasses import dataclass

import httpx

@dataclass
class AgentContext:
    """保存服务端提供的运行资源引用。"""

    metrics_client: httpx.AsyncClient
    metrics_semaphore: asyncio.Semaphore
