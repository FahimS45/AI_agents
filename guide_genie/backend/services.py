from __future__ import annotations

import os
import sys
import traceback
from datetime import datetime
from typing import Dict, Optional

# ── Path fix ──────────────────────────────────────────────────────
# Add backend/ so sibling imports (config, models) resolve.
# Add project root + src/ so guide_generator_flow package resolves.
_backend_dir  = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_backend_dir)   # guide_generator_flow/
_src_dir      = os.path.join(_project_root, "src")

for _p in (_backend_dir, _project_root, _src_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# ──────────────────────────────────────────────────────────────────

# ── Import the CrewAI flow ─────────────────────────────────────────
try:
    from guide_generator_flow.main import GuideGeneratorFlow
    _FLOW_AVAILABLE = True
except Exception as _err:
    print(f"[WARNING] Could not import GuideGeneratorFlow: {_err}")
    _FLOW_AVAILABLE = False

from config import settings
from models import TaskStatus, TaskStatusResponse


class FlowService:
    """In-memory task registry + CrewAI flow runner."""

    def __init__(self) -> None:
        self._tasks: Dict[str, TaskStatusResponse] = {}

    # ── Public API ─────────────────────────────────────────────────

    def run(
        self,
        *,
        task_id: str,
        youtube_links: str,
        webpage_links: str,
        research_paper_links: str,
        document_paths: str,
    ) -> None:
        """Called as a FastAPI BackgroundTask."""

        self._tasks[task_id] = TaskStatusResponse(
            task_id=task_id,
            status=TaskStatus.PROCESSING,
            message="Guide generation started…",
            created_at=datetime.now(),
        )

        try:
            if not _FLOW_AVAILABLE:
                raise RuntimeError(
                    "GuideGeneratorFlow could not be imported. "
                    "Run `uv sync` and check the startup warnings."
                )

            flow = GuideGeneratorFlow()
            flow.kickoff(
                inputs={
                    "youtube_links":        youtube_links        or "Not provided",
                    "webpage_links":        webpage_links        or "Not provided",
                    "research_paper_links": research_paper_links or "Not provided",
                    "document_paths":       document_paths       or "Not provided",
                }
            )

            html_content    = flow.state.final_guide      # HTML string from WritingCrew
            research_report = flow.state.research_report

            if not html_content:
                raise RuntimeError(
                    "WritingCrew returned an empty guide. "
                    "Check your agent/task config."
                )

            # ── Persist the HTML file ──────────────────────────────
            os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
            html_path = os.path.join(settings.OUTPUT_DIR, f"guide_{task_id[:8]}.html")

            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            # ── Mark complete ──────────────────────────────────────
            task = self._tasks[task_id]
            task.status          = TaskStatus.COMPLETED
            task.message         = "Guide generated successfully!"
            task.completed_at    = datetime.now()
            task.research_report = research_report
            task.html_file_path  = html_path
            task.download_url    = f"/api/download/{task_id}"

        except Exception as exc:
            print(f"[ERROR] task {task_id}:\n{traceback.format_exc()}")
            task = self._tasks[task_id]
            task.status       = TaskStatus.FAILED
            task.message      = "Guide generation failed."
            task.error        = str(exc)
            task.completed_at = datetime.now()

    def get(self, task_id: str) -> Optional[TaskStatusResponse]:
        return self._tasks.get(task_id)

    def cleanup(self, task_id: str) -> bool:
        task = self._tasks.pop(task_id, None)
        if task is None:
            return False
        if task.html_file_path and os.path.exists(task.html_file_path):
            try:
                os.remove(task.html_file_path)
            except OSError:
                pass
        return True