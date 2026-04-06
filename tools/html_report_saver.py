"""
tools/html_report_saver.py — Save a generated HTML report to disk.

WHAT THIS TOOL DOES
-------------------
After the html_report_generator node produces an HTML string, this tool:
  1. Creates a session-specific output directory: outputs/{session_id}/
  2. Generates a timestamped filename to avoid overwriting previous reports.
  3. Writes the HTML to disk as a UTF-8 file.
  4. Returns the URL path where the file will be served by the FastAPI static
     file server (Phase 2).

WHY SESSION-SCOPED DIRECTORIES?
--------------------------------
Organising reports under outputs/{session_id}/ keeps each user's reports
separate on disk, mirroring the per-session isolation enforced at the API level.
The FastAPI endpoint GET /api/reports/{session_id}/{filename} can verify the
session_id before serving the file, preventing one user from accessing another
user's reports.

RETURN VALUE FORMAT
-------------------
Returns a URL-style path string: "/api/reports/{session_id}/{filename}"
In Phase 1 (script mode) this path is just informational.
In Phase 2 (FastAPI) it becomes a real URL the frontend can open or download.
"""

import asyncio
import os
from datetime import datetime
from pathlib import Path

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class HTMLReportInput(BaseModel):
    """Input schema for HTMLReportSaverTool."""
    html_text: str = Field(description="The complete HTML document string to save")
    session_id: str = Field(
        default="default",
        description="Session ID used to organise files under outputs/{session_id}/"
    )


class HTMLReportSaverTool(BaseTool):
    """
    Save an HTML report string to the local filesystem.

    Creates the output directory if it does not exist, writes the file,
    and returns the URL path that the Phase 2 static file server will use
    to serve it.
    """

    name: str = "html_report_saver"
    description: str = "Saves an HTML string to disk and returns the URL path to the saved file."
    args_schema: type[BaseModel] = HTMLReportInput

    def _run(self, html_text: str, session_id: str = "default") -> str:
        """
        Write the HTML report to outputs/{session_id}/{timestamped_filename}.

        Args:
            html_text:  The full HTML document as a string.
            session_id: Used to create a session-specific subfolder.

        Returns:
            URL path string: "/api/reports/{session_id}/{filename}"
            Returns an error string prefixed with "ERROR:" if writing fails.
        """
        try:
            # Build the output directory path and create it if needed.
            out_dir = Path(os.getenv("OUTPUTS_DIR", "outputs")) / session_id
            out_dir.mkdir(parents=True, exist_ok=True)

            # Timestamp the filename so repeated reports don't overwrite each other.
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"job_market_report_{timestamp}.html"
            path = out_dir / filename

            # Write the HTML file with UTF-8 encoding (important for non-ASCII characters).
            path.write_text(html_text, encoding="utf-8")

            # Return the URL path format that Phase 2's FastAPI server will expose.
            return f"/api/reports/{session_id}/{filename}"

        except Exception as e:
            return f"ERROR: Failed to save HTML report: {e}"

    async def _arun(self, html_text: str, session_id: str = "default") -> str:
        """
        Async wrapper for _run().

        File I/O is blocking, so we offload it to a thread pool to avoid
        blocking the async event loop in the Phase 2 FastAPI server.
        """
        return await asyncio.to_thread(self._run, html_text, session_id)
