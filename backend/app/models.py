from pydantic import BaseModel
from typing import Any, Optional


class GenerateDesignResponse(BaseModel):
    status: str
    generated_image_url: str
    llm_prompt: str
    prompt_strategy: str
    generation_metadata: Optional[dict[str, Any]] = None
    notes: Optional[str] = None
