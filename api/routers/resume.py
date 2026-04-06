"""
api/routers/resume.py — Resume PDF upload endpoint.

Endpoint:
  POST /api/chat/{session_id}/resume — Upload a PDF resume for a session.

The PDF bytes are stored in the in-memory session store (graph/session_store.py).
Since the fly.io machine is kept running continuously, the in-memory store
persists across requests without needing DB persistence.

The graph's check_resume node calls has_resume(session_id) to decide whether
to proceed with resume analysis, and resume_parser reads the bytes from
get_resume(session_id).
"""

from fastapi import APIRouter, File, HTTPException, UploadFile

from graph.session_store import save_resume

router = APIRouter()

_MAX_PDF_BYTES = 10 * 1024 * 1024  # 10 MB


@router.post("/chat/{session_id}/resume")
async def upload_resume(session_id: str, file: UploadFile = File(...)):
    """
    Upload a PDF resume for the given session.

    The file is validated (PDF only, ≤10 MB) and stored in the in-memory
    session store keyed by session_id. The client should call this before
    sending a resume_analysis chat message.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    pdf_bytes = await file.read()

    if len(pdf_bytes) > _MAX_PDF_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 10 MB).")

    save_resume(session_id, pdf_bytes)

    return {
        "session_id": session_id,
        "filename": file.filename,
        "status": "uploaded",
    }
