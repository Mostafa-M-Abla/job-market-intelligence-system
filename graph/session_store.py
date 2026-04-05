"""
graph/session_store.py — Temporary in-memory storage for uploaded resume files.

PHASE 1 IMPLEMENTATION
In Phase 1 there is no database-backed API layer, so uploaded resume PDFs are
kept in a simple Python dictionary keyed by session ID.

PHASE 2 REPLACEMENT
In Phase 2, the FastAPI backend will receive resume uploads via
POST /api/chat/resume and store the raw bytes in the `sessions` table in
SQLite/PostgreSQL.  This module will then be replaced by DB queries in
the resume_parser node.

Why store bytes instead of a file path?
  - It avoids filesystem dependencies — the API server might be stateless or
    containerised without a persistent volume.
  - The ResumePDFExtractorTool accepts raw bytes directly, so no temp file is
    needed at any stage.
"""

from typing import Optional

# In-memory store: { session_id -> pdf_bytes }
# This is intentionally a plain module-level dict.  In Phase 2 it will be
# replaced by database calls; for now simplicity beats durability.
_store: dict[str, bytes] = {}


def save_resume(session_id: str, pdf_bytes: bytes) -> None:
    """
    Store the raw bytes of a PDF resume for a given session.

    Args:
        session_id: The unique identifier for this user's session.
        pdf_bytes:  The raw binary content of the uploaded PDF file.
    """
    _store[session_id] = pdf_bytes


def get_resume(session_id: str) -> Optional[bytes]:
    """
    Retrieve the stored resume bytes for a session, or None if not uploaded yet.

    Args:
        session_id: The unique identifier for this user's session.

    Returns:
        Raw PDF bytes, or None if no resume has been uploaded for this session.
    """
    return _store.get(session_id)


def has_resume(session_id: str) -> bool:
    """
    Check whether a non-empty resume has been uploaded for a session.

    Used by the check_resume node and the route_after_check_resume routing
    function to decide whether to proceed with analysis or prompt the user
    to upload their resume first.

    Args:
        session_id: The unique identifier for this user's session.

    Returns:
        True if a resume with non-zero content exists for this session.
    """
    return session_id in _store and bool(_store[session_id])
