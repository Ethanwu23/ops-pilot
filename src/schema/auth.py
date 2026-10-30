from pydantic import BaseModel, ConfigDict, Field


class AuthenticatedUser(BaseModel):
    """表示由服务端认证依赖确认的用户身份。"""

    user_id: str = Field(
        min_length=1,
        max_length=100,
    )

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


class AccessTokenClaims(BaseModel):
    """表示通过签名校验后允许使用的 JWT 声明。"""

    sub: str = Field(
        min_length=1,
        max_length=100,
    )
    iss: str = Field(
        min_length=1,
        max_length=100,
    )
    aud: str = Field(
        min_length=1,
        max_length=100,
    )
    iat: int = Field(
        gt=0,
        strict=True,
    )
    exp: int = Field(
        gt=0,
        strict=True,
    )

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )
