from typing import Any, Dict, List, Optional, Tuple, Union, get_args, get_origin
from uuid import UUID

from pydantic import BaseModel, Field


class DeleteResponse(BaseModel):
    id: UUID = Field(description="ID of the task")
    status: str = Field(description="Status of the task after deletion attempt")
    message: str = Field(description="Additional information about the deletion result")
