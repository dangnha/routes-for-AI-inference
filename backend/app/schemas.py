from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class Role(StrEnum):
    USER = "user"
    ADMIN = "admin"


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    email: str | None = None
    role: Role = Role.USER


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class PresignRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    content_type: str = Field(default="image/png", max_length=100)


class PresignResponse(BaseModel):
    object_key: str
    upload_url: str


class TaskCreate(BaseModel):
    object_key: str = Field(min_length=1, max_length=512)
    model: str = Field(default="trocr", max_length=64)
    filename: str = Field(default="", max_length=255)


class ModelInfo(BaseModel):
    slug: str
    name: str
    description: str


class TaskRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    owner_id: str
    filename: str
    model: str
    status: TaskStatus
    progress: int = Field(ge=0, le=100)
    text: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    lines: list[dict] | None = None
    error: str | None = None
    created_at: datetime


class InferenceCallback(BaseModel):
    """Payload the AI service POSTs back to /webhooks/inference."""

    task_id: str
    model: str
    status: TaskStatus
    text: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    error: str | None = None
    lines: list[dict] = Field(default_factory=list)
