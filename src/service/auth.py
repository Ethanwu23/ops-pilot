from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import InvalidTokenError
from pydantic import ValidationError

from core import settings
from schema.auth import AccessTokenClaims, AuthenticatedUser

JWT_ALGORITHM = "HS256"
bearer_scheme = HTTPBearer(
    description="请提供用户访问令牌",
    auto_error=False,
)


class AuthenticationError(Exception):
    """表示访问令牌不能建立可信用户身份。"""


def decode_access_token(token: str) -> AuthenticatedUser:
    """验证 JWT，并从受信任的 sub 声明建立用户身份。"""
    if not token:
        raise AuthenticationError("无效的访问令牌")

    try:
        payload = jwt.decode(
            token,
            key=settings.JWT_SECRET.get_secret_value(),
            algorithms=[JWT_ALGORITHM],
            issuer=settings.JWT_ISSUER,
            audience=settings.JWT_AUDIENCE,
            options={
                "require": ["sub", "iss", "aud", "iat", "exp"],
            },
        )
        claims = AccessTokenClaims.model_validate(payload)
    except (InvalidTokenError, ValidationError) as exc:
        raise AuthenticationError("无效的访问令牌") from exc

    return AuthenticatedUser(user_id=claims.sub)


def get_authenticated_user(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(bearer_scheme),
    ],
) -> AuthenticatedUser:
    """从 Bearer Token 建立当前请求的可信用户身份。"""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效或缺失的访问令牌",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        return decode_access_token(credentials.credentials)
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效或缺失的访问令牌",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
