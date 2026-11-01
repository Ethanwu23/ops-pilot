from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import patch

import jwt
import pytest
from pydantic import SecretStr

from service import app
from service.auth import JWT_ALGORITHM, get_authenticated_user

TEST_SECRET = "test-only-signing-key-at-least-32-bytes"


@pytest.fixture
def jwt_client(test_client):
    """复用资源替身，但取消身份替换，执行真实 JWT 认证。"""
    overrides = dict(app.dependency_overrides)
    app.dependency_overrides.pop(get_authenticated_user)
    settings = SimpleNamespace(
        JWT_SECRET=SecretStr(TEST_SECRET),
        JWT_ISSUER="test-issuer",
        JWT_AUDIENCE="test-audience",
    )
    try:
        with patch("service.auth.settings", settings):
            yield test_client
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(overrides)


def make_token(**changes):
    """使用固定测试密钥签发短期令牌，不读取真实密钥。"""
    now = int(datetime.now(UTC).timestamp())
    claims = {
        "sub": "user-123",
        "iss": "test-issuer",
        "aud": "test-audience",
        "iat": now - 10,
        "exp": now + 600,
    }
    claims.update(changes)
    return jwt.encode(claims, TEST_SECRET, algorithm=JWT_ALGORITHM)


def test_valid_jwt_establishes_user(jwt_client, mock_agent):
    """身份来自签名声明，并传入工作流配置和上下文。"""
    response = jwt_client.post(
        "/invoke", json={"message": "test"},
        headers={"Authorization": f"Bearer {make_token()}"},
    )
    assert response.status_code == 200
    kwargs = mock_agent.ainvoke.await_args.kwargs
    assert kwargs["config"]["configurable"]["user_id"] == "user-123"
    assert kwargs["context"].user_id == "user-123"


@pytest.mark.parametrize("authorization", [None, "Bearer invalid-token", "Basic invalid"])
def test_missing_or_malformed_token_is_rejected(jwt_client, mock_agent, authorization):
    """认证失败必须在进入工作流前拒绝。"""
    headers = {} if authorization is None else {"Authorization": authorization}
    response = jwt_client.post("/invoke", json={"message": "test"}, headers=headers)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json() == {"detail": "无效或缺失的访问令牌"}
    mock_agent.aget_state.assert_not_awaited()
    mock_agent.ainvoke.assert_not_awaited()


@pytest.mark.parametrize("claims", [
    {"exp": 1},
    {"iss": "wrong-issuer"},
    {"aud": "wrong-audience"},
    {"sub": ""},
])
def test_invalid_claims_are_rejected(jwt_client, mock_agent, claims):
    """过期、声明不匹配和无效身份均不能建立用户身份。"""
    response = jwt_client.post(
        "/invoke", json={"message": "test"},
        headers={"Authorization": f"Bearer {make_token(**claims)}"},
    )
    assert response.status_code == 401
    mock_agent.ainvoke.assert_not_awaited()


def test_wrong_signing_key_is_rejected(jwt_client, mock_agent):
    """结构正确但签名不可信的令牌必须被拒绝。"""
    claims = jwt.decode(make_token(), options={"verify_signature": False})
    token = jwt.encode(claims, "another-test-only-key-at-least-32-bytes", algorithm=JWT_ALGORITHM)
    response = jwt_client.post(
        "/invoke", json={"message": "test"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 401
    mock_agent.ainvoke.assert_not_awaited()
