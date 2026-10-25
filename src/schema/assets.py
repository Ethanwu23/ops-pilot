from typing import Literal
from datetime import datetime

from pydantic import BaseModel, Field, IPvAnyAddress, ConfigDict


class AssetResponse(BaseModel):
    id: int = Field(gt=0)
    name: str = Field(min_length=1, max_length=100)
    asset_type: str = Field(min_length=1, max_length=50)
    ip_address: IPvAnyAddress
    status: Literal["unknown","online", "offline"]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class AssetLookupInput(BaseModel):
    asset_id: int = Field(gt=0, strict=True)
    model_config = ConfigDict(extra='forbid')
      
    
