from pydantic import BaseModel
from typing import Optional


class GenerateDesignResponse(BaseModel):
    status: str
    generated_image_url: str
    llm_prompt: str
    prompt_strategy: str
    notes: Optional[str] = None
