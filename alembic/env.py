import asyncio
from logging.config import fileConfig

from sqlalchemy import Connection, pool
from sqlalchemy.ext.asyncio import create_async_engine

from alembic import context
from persistence.database import get_business_database_url
from persistence.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata
BUSINESS_TABLE_NAMES = set(target_metadata.tables)


def include_name(
    name: str | None,
    type_: str,
    parent_names: dict[str, str | None],
) -> bool:
    """仅允许 Alembic 管理当前业务 ORM 元数据中的表。"""
    if type_ == "table":
        return name in BUSINESS_TABLE_NAMES

    return True


def run_migrations_offline() -> None:
    """在不连接数据库的情况下生成迁移 SQL。"""
    context.configure(
        url=get_business_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
        include_name=include_name,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """使用 Alembic 当前连接执行迁移。"""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
        include_name=include_name,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """创建独立迁移连接并在线执行迁移。"""
    engine = create_async_engine(
        get_business_database_url(),
        poolclass=pool.NullPool,
    )

    try:
        async with engine.connect() as connection:
            await connection.run_sync(do_run_migrations)
    finally:
        await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
