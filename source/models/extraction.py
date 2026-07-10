from datetime import datetime, timezone
from pydantic import BaseModel, Field
from typing import List, Optional,Any

class Extraction(BaseModel):
    id: Optional[str]= None
    document_id: str
    file_name: Optional[str]= None
    model_name: str
    prompt_version: Optional[str]= None
    extracted_data: Optional[dict[str,Any]]= None
    raw_response: Optional[str] = None
    error: Optional[str] = None

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )