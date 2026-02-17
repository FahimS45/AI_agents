import os
import sys

# ── Path fix ──────────────────────────────────────────────────────
# When uvicorn is launched as `uvicorn backend.main:app` from the
# project root, Python puts the project root on sys.path but NOT
# the backend/ subdirectory.  We add it here so that the plain
# `from models import ...` / `from services import ...` /
# `from config import ...` sibling imports all resolve correctly.
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import uuid
import shutil
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from models import (
    FileUploadResponse,
    GuideGenerationResponse,
    TaskStatusResponse,
    UploadedFileInfo,
)
from services import FlowService
from config import settings

app = FastAPI(
    title="Guide Generator API",
    description="Generate getting-started guides via CrewAI. Upload docs, poll for status, download the HTML result.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)

flow_service = FlowService()


# ──────────────────────────────────────────────────────────
#  Health
# ──────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    return {"status": "online", "version": "1.0.0"}


# ──────────────────────────────────────────────────────────
#  Upload documents
# ──────────────────────────────────────────────────────────

@app.post("/api/upload-documents", response_model=FileUploadResponse, tags=["Files"])
async def upload_documents(
    files: List[UploadFile] = File(...),
):
    """
    Upload PDF / TXT / MD / MDX files before generation.
    Returns the server-side `path` for each file — pass these as
    `document_paths` (comma-separated) to `/api/generate-guide`.
    """
    uploaded: list[UploadedFileInfo] = []
    errors:   list[str]              = []

    for file in files:
        suffix = Path(file.filename).suffix.lower()

        if suffix not in settings.ALLOWED_EXTENSIONS:
            errors.append(
                f"{file.filename}: unsupported type "
                f"(allowed: {', '.join(settings.ALLOWED_EXTENSIONS)})"
            )
            continue

        file.file.seek(0, 2)
        size = file.file.tell()
        file.file.seek(0)

        if size > settings.MAX_FILE_SIZE_BYTES:
            errors.append(
                f"{file.filename}: exceeds "
                f"{settings.MAX_FILE_SIZE_BYTES // 1024 // 1024} MB limit"
            )
            continue

        try:
            uid       = uuid.uuid4().hex[:8]
            safe_name = f"{uid}_{file.filename}"
            dest      = os.path.join(settings.UPLOAD_DIR, safe_name)

            with open(dest, "wb") as buf:
                shutil.copyfileobj(file.file, buf)

            uploaded.append(
                UploadedFileInfo(
                    original_name=file.filename,
                    saved_name=safe_name,
                    path=dest,
                    size=size,
                    type=suffix,
                )
            )
        except Exception as exc:
            errors.append(f"{file.filename}: failed to save — {exc}")

    if not uploaded and errors:
        raise HTTPException(status_code=400, detail={"errors": errors})

    return FileUploadResponse(
        success=True,
        message=f"{len(uploaded)} file(s) uploaded.",
        files=uploaded,
        errors=errors or None,
    )


# ──────────────────────────────────────────────────────────
#  Start guide generation  (async background task)
# ──────────────────────────────────────────────────────────

@app.post("/api/generate-guide", response_model=GuideGenerationResponse, tags=["Guide"])
async def generate_guide(
    background_tasks: BackgroundTasks,
    youtube_links:        Optional[str]        = Form(None),
    webpage_links:        Optional[str]        = Form(None),
    research_paper_links: Optional[str]        = Form(None),
    document_paths:       Optional[str]        = Form(
        None,
        description="Comma-separated paths returned by /api/upload-documents",
    ),
    uploaded_files: Optional[List[UploadFile]] = File(
        None,
        description="Upload files directly alongside other sources",
    ),
):
    """
    Start guide generation as a background task.
    Returns a `task_id` immediately — poll `/api/status/{task_id}`.
    When `status == completed` use `/api/download/{task_id}`.
    """
    # Save any inline-uploaded files
    extra_paths: list[str] = []
    if uploaded_files:
        for f in uploaded_files:
            suffix = Path(f.filename).suffix.lower()
            if suffix not in settings.ALLOWED_EXTENSIONS:
                continue
            uid  = uuid.uuid4().hex[:8]
            dest = os.path.join(settings.UPLOAD_DIR, f"{uid}_{f.filename}")
            with open(dest, "wb") as buf:
                shutil.copyfileobj(f.file, buf)
            extra_paths.append(dest)

    all_doc_paths = []
    if document_paths:
        all_doc_paths += [p.strip() for p in document_paths.split(",") if p.strip()]
    all_doc_paths += extra_paths
    final_doc_paths = ",".join(all_doc_paths)

    if not any([youtube_links, webpage_links, research_paper_links, final_doc_paths]):
        raise HTTPException(
            status_code=422,
            detail="Provide at least one source: youtube_links, webpage_links, "
                   "research_paper_links, or documents.",
        )

    task_id = str(uuid.uuid4())

    background_tasks.add_task(
        flow_service.run,
        task_id=task_id,
        youtube_links=youtube_links              or "",
        webpage_links=webpage_links              or "",
        research_paper_links=research_paper_links or "",
        document_paths=final_doc_paths,
    )

    return GuideGenerationResponse(
        task_id=task_id,
        status="processing",
        message="Started. Poll /api/status/{task_id} for updates.",
    )


# ──────────────────────────────────────────────────────────
#  Poll status
# ──────────────────────────────────────────────────────────

@app.get("/api/status/{task_id}", response_model=TaskStatusResponse, tags=["Guide"])
async def get_status(task_id: str):
    """
    Returns the task's current state.

    `status` values: `processing` | `completed` | `failed`

    When `completed`, the response includes `download_url`.
    """
    task = flow_service.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found.")
    return task


# ──────────────────────────────────────────────────────────
#  Download finished HTML guide
# ──────────────────────────────────────────────────────────

@app.get("/api/download/{task_id}", tags=["Guide"])
async def download_guide(task_id: str):
    """
    Download the finished guide as an HTML file.
    Only available when `status == completed`.
    """
    task = flow_service.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found.")

    if task.status != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Not ready yet (current status: {task.status}).",
        )

    if not task.html_file_path or not os.path.exists(task.html_file_path):
        raise HTTPException(status_code=500, detail="Output file is missing on server.")

    filename = f"getting-started-guide-{task_id[:8]}.html"

    return FileResponse(
        path=task.html_file_path,
        media_type="text/html",
        filename=filename,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ──────────────────────────────────────────────────────────
#  Cleanup
# ──────────────────────────────────────────────────────────

@app.delete("/api/cleanup/{task_id}", tags=["Guide"])
async def cleanup(task_id: str):
    """Delete task state and the generated HTML file from the server."""
    if not flow_service.cleanup(task_id):
        raise HTTPException(status_code=404, detail="Task not found.")
    return {"message": f"Task {task_id} cleaned up."}


# ──────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)