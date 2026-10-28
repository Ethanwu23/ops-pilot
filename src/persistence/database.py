from typing import cast

from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from core import settings
from memory.postgres import validate_postgres_config

BusinessSessionFactory = async_sessionmaker[AsyncSession]


def get_business_database_url() -> URL:
    """构造业务数据库使用的 SQLAlchemy 异步连接 URL。"""
    validate_postgres_config()

    if settings.POSTGRES_PASSWORD is None:
        raise ValueError("POSTGRES_PASSWORD is not set")

    return URL.create(
        drivername="postgresql+psycopg",
        username=cast(str, settings.POSTGRES_USER),
        password=settings.POSTGRES_PASSWORD.get_secret_value(),
        host=cast(str, settings.POSTGRES_HOST),
        port=cast(int, settings.POSTGRES_PORT),
        database=cast(str, settings.POSTGRES_DB),
    )


def create_business_engine() -> AsyncEngine:
    """创建业务数据库异步 Engine 和连接池。"""
    return create_async_engine(
        get_business_database_url(),
        pool_pre_ping=True,
        hide_parameters=True,
        pool_size=settings.POSTGRES_MAX_CONNECTIONS_PER_POOL,
        max_overflow=0,
        pool_timeout=5.0,
    )


def create_business_session_factory(
    engine: AsyncEngine,
) -> BusinessSessionFactory:
    """创建用于短事务的异步 Session 工厂。"""
    return async_sessionmaker(
        bind=engine,
        expire_on_commit=False,
    )
