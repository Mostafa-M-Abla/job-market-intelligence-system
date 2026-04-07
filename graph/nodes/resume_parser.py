"""
graph/nodes/resume_parser.py — Extract readable text from the uploaded resume PDF.

ROLE IN THE GRAPH
-----------------
This node runs only for the resume_analysis intent, immediately after check_resume
confirms that a PDF is available for this session.

Flow position:
  check_resume (PDF exists) -> resume_parser -> cache_lookup -> ...

Its job is simple: fetch the raw PDF bytes from the in-memory session store and
hand them to ResumePDFExtractorTool, which uses the pypdf library to pull out
all readable text.

The extracted text is stored in state["resume_text"] and later consumed by:
  - skill_gap_analyzer: compares resume skills against market demand
  - html_report_generator: includes personalised gap analysis in the HTML report

WHY BYTES, NOT A FILE PATH?
----------------------------
The session store holds raw bytes (the PDF file content, not a path on disk).
This design is intentional — in Phase 2 (FastAPI), uploaded files are stored
directly in the database as BLOBs, and there is no guarantee of a persistent
filesystem in a containerised environment.  Passing bytes keeps the tool
stateless and portable.

ERROR HANDLING
--------------
If (by some race condition) the PDF has disappeared from the session store
between check_resume and this node, we set final_text_response to an upload
prompt and let the respond node surface it — the same graceful handling used
by check_resume.

LAZY TOOL INITIALISATION
-------------------------
`_tool` is created on the first call, not at import time.  This means importing
this module doesn't require pypdf or any API key to be available yet, which
simplifies testing and startup.
"""

import logging

from graph.session_store import get_resume
from graph.state import JobMarketState
from tools.resume_pdf_tool import ResumePDFExtractorTool

logger = logging.getLogger(__name__)

# Module-level singleton — created once, reused for the lifetime of the process.
# None until first call to _get_tool().
_tool = None


def _get_tool() -> ResumePDFExtractorTool:
    """
    Return the shared ResumePDFExtractorTool instance, creating it on first call.

    Lazy initialisation avoids loading pypdf and instantiating the tool object
    on every import.  Since node files are imported when the graph is compiled,
    this pattern keeps startup fast and avoids side-effects at import time.
    """
    global _tool
    if _tool is None:
        _tool = ResumePDFExtractorTool()
    return _tool


def resume_parser(state: JobMarketState) -> dict:
    """
    Fetch the uploaded resume PDF from the session store and extract its text.

    Steps:
      1. Look up pdf_bytes for this session_id in the in-memory session store.
      2. If not found (race condition / session expired), return an upload prompt
         so the respond node can tell the user what to do.
      3. Call ResumePDFExtractorTool with the bytes to extract readable text.
      4. Return the text in the "resume_text" state field.

    Args:
        state: Current graph state.  Reads session_id.

    Returns:
        {"resume_text": "<extracted text>"} on success.
        {"final_text_response": "<upload prompt>"} if PDF is missing.

    Note:
        ResumePDFExtractorTool will raise ValueError if the PDF contains no
        extractable text (e.g. it's a scanned image PDF).  That error is not
        caught here — it will propagate and surface as an unhandled exception.
        In Phase 2, the API layer will wrap node calls in try/except and return
        a 500 error response.
    """
    session_id = state.get("session_id", "")

    # Fetch the raw PDF bytes that were stored when the user uploaded their resume.
    # get_resume() returns None if no PDF has been uploaded for this session yet.
    pdf_bytes = get_resume(session_id)

    if not pdf_bytes:
        logger.warning("resume_parser: no PDF bytes for session %s — sending upload prompt", session_id[:8])
        return {
            "final_text_response": (
                "Resume not found in session. Please upload your resume PDF first."
            )
        }

    logger.info("resume_parser: extracting text from PDF (%d bytes, session %s)", len(pdf_bytes), session_id[:8])
    resume_text = _get_tool().run({"pdf_bytes": pdf_bytes})
    logger.info("resume_parser: extracted %d chars of resume text", len(resume_text))

    return {"resume_text": resume_text}
