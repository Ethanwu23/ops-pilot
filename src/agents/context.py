import asyncio
from dataclasses import dataclass

import httpx

from persistence.database import BusinessSessionFactory


@dataclass
class AgentContext:
    """保存服务端提供的运行资源引用。"""

    metrics_client: httpx.AsyncClient
    metrics_semaphore: asyncio.Semaphore
    business_session_factory: BusinessSessionFactory
