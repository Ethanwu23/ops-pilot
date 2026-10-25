from typing import Literal

from pydantic import BaseModel, Field, ConfigDict



class MetricsLookupInput(BaseModel):
    asset_id: int = Field(gt=0, strict=True)

    model_config = ConfigDict(extra='forbid')



class MetricsResponse(BaseModel):
    asset_id: int = Field(gt=0, strict=True)
    source: Literal["local_test_metrics"]
    is_test_data: Literal[True]
    cpu_percent: float = Field(ge=0, le=100)
    memory_percent: float = Field(ge=0, le=100)

    model_config = ConfigDict(extra="forbid")
