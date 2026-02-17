from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class TaskStatus(str, Enum):
    PROCESSING = "processing"
    COMPLETED  = "completed"
    FAILED     = "failed"


# ── File upload ────────────────────────────────────────────────────

class UploadedFileInfo(BaseModel):
    original_name: str
    saved_name:    str
    path:          str          # server-side absolute path
    size:          int          # bytes
    type:          str          # e.g. ".pdf"


class FileUploadResponse(BaseModel):
    success: bool
    message: str
    files:   List[UploadedFileInfo]
    errors:  Optional[List[str]] = None


# ── Guide generation ───────────────────────────────────────────────

class GuideGenerationResponse(BaseModel):
    task_id: str
    status:  str
    message: str


class TaskStatusResponse(BaseModel):
    task_id:      str
    status:       TaskStatus
    message:      str
    created_at:   datetime
    completed_at: Optional[datetime] = None

    # Set once the flow finishes
    research_report: Optional[str] = None   # raw research output
    download_url:    Optional[str] = None   # e.g. /api/download/{task_id}

    # Internal — not exposed in JSON responses
    html_file_path: Optional[str] = None

    error: Optional[str] = None

    model_config = {"json_schema_extra": {
        "example": {
            "task_id":      "a1b2c3d4-...",
            "status":       "completed",
            "message":      "Guide generated successfully!",
            "created_at":   "2024-02-12T10:30:00",
            "completed_at": "2024-02-12T10:35:00",
            "download_url": "/api/download/a1b2c3d4",
        }
    }}