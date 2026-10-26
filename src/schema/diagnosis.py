from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Hypothesis(BaseModel):
    statement: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)
    verification: str = Field(min_length=1)

    model_config = ConfigDict(extra="forbid")


class AnalysisResult(BaseModel):
    summary: str = Field(min_length=1)
    hypotheses: list[Hypothesis]
    missing_information: list[str]

    model_config = ConfigDict(extra="forbid")


class Evidence(BaseModel):
    """保存程序从工具结果中提取的证据，不由模型改写。"""

    evidence_id: str = Field(min_length=1)
    tool_name: Literal["get_asset", "get_metrics"]
    source: str = Field(min_length=1)
    is_test_data: bool = Field(strict=True)
    data: dict[str, Any]

    model_config = ConfigDict(extra="forbid")
