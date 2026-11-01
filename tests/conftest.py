import os
from unittest.mock import patch

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-docker", action="store_true", default=False, help="run docker integration tests"
    )


    parser.addoption(
        "--run-postgres", action="store_true", default=False,
        help="run isolated PostgreSQL integration tests",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "docker: mark test as requiring docker containers")
    config.addinivalue_line("markers", "postgres: requires explicit isolated test database")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-docker"):
        skip_docker = pytest.mark.skip(reason="need --run-docker option to run")
        for item in items:
            if "docker" in item.keywords:
                item.add_marker(skip_docker)
    if not config.getoption("--run-postgres"):
        for item in items:
            if "postgres" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="need --run-postgres option to run"))


@pytest.fixture
def mock_env():
    """Fixture to ensure environment is clean for each test."""
    with patch.dict(os.environ, {}, clear=True):
        yield
