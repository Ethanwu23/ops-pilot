import logging
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI

from schema import AgentInfo


@pytest.mark.asyncio
async def test_lifespan(monkeypatch, caplog) -> None:
    """Test that the lifespan sets up the database and store, loads the agents, and logs errors."""
    from service import service

    # 不创建真实数据库 Engine；本测试只验证资源生命周期。
    engine = Mock(dispose=AsyncMock())
    session_factory = Mock()
    monkeypatch.setattr(service, "create_business_engine", Mock(return_value=engine))
    monkeypatch.setattr(service, "create_business_session_factory", Mock(return_value=session_factory))
    monkeypatch.setattr(service.settings, "METRICS_BASE_URL", "http://metrics.test:9000")
    monkeypatch.setattr(service.settings, "METRICS_TIMEOUT_SECONDS", 1.5)
    monkeypatch.setattr(service.settings, "METRICS_MAX_CONCURRENCY", 3)

    fake_saver_setup = False
    fake_store_setup = False

    class FakeSaver:
        async def setup(self) -> None:
            nonlocal fake_saver_setup
            fake_saver_setup = True

    class FakeStore:
        async def setup(self) -> None:
            nonlocal fake_store_setup
            fake_store_setup = True

    fake_saver = FakeSaver()
    fake_store = FakeStore()

    @asynccontextmanager
    async def fake_initialize_database():
        yield fake_saver

    @asynccontextmanager
    async def fake_initialize_store():
        yield fake_store

    agents = {
        "good": type("Agent", (), {"checkpointer": None, "store": None})(),
        "bad": type("Agent", (), {"checkpointer": None, "store": None})(),
    }

    async def fake_load_agent(agent_key: str) -> None:
        if agent_key == "bad":
            raise RuntimeError("boom")

    def fake_get_agent(agent_key: str):
        return agents[agent_key]

    monkeypatch.setattr(service, "initialize_database", fake_initialize_database)
    monkeypatch.setattr(service, "initialize_store", fake_initialize_store)
    monkeypatch.setattr(service, "load_agent", fake_load_agent)
    monkeypatch.setattr(service, "get_agent", fake_get_agent)
    monkeypatch.setattr(
        service,
        "get_all_agent_info",
        lambda: [
            AgentInfo(key="good", description=""),
            AgentInfo(key="bad", description=""),
        ],
    )

    caplog.set_level(logging.INFO, logger=service.logger.name)

    app = FastAPI()
    async with service.lifespan(app):
        client = app.state.metrics_client
        assert str(client.base_url) == "http://metrics.test:9000"
        assert client.timeout.read == 1.5
        assert app.state.business_session_factory is session_factory
        semaphore = app.state.metrics_semaphore
        for _ in range(3):
            await semaphore.acquire()
        assert semaphore.locked()
        for _ in range(3):
            semaphore.release()
        assert not client.is_closed

    assert client.is_closed
    engine.dispose.assert_awaited_once()

    assert fake_saver_setup
    assert fake_store_setup
    assert agents["good"].checkpointer is fake_saver
    assert agents["good"].store is fake_store
    assert agents["bad"].checkpointer is fake_saver
    assert agents["bad"].store is fake_store

    assert "Agent loaded: good" in caplog.text
    assert "Failed to load agent bad: boom" in caplog.text


@pytest.mark.parametrize("field, value", [
    ("METRICS_TIMEOUT_SECONDS", 0),
    ("METRICS_TIMEOUT_SECONDS", -1),
    ("METRICS_MAX_CONCURRENCY", 0),
    ("METRICS_MAX_CONCURRENCY", -1),
    ("METRICS_BASE_URL", "not-a-url"),
])
def test_invalid_metrics_configuration_is_rejected(field, value):
    """无效资源配置应在启动前校验失败。"""
    from pydantic import ValidationError

    from core.settings import Settings

    with pytest.raises(ValidationError) as error:
        Settings(_env_file=None, JWT_SECRET="test-only-secret", OPENAI_API_KEY="test-key", **{field: value})
    assert any(item["loc"] == (field,) for item in error.value.errors())
