"""
graph/nodes/check_resume.py — Verify that a resume PDF has been uploaded.

ROLE IN THE GRAPH
-----------------
This is the first node on the resume_analysis path.  It runs before any LLM
calls or market data lookups, because there is no point doing any of that work
if the user has not yet provided their resume.

The check is purely deterministic — it looks up the session ID in the in-memory
store (Phase 1) or database (Phase 2).  No LLM is involved.

OUTPUTS written to state
------------------------
  final_text_response : set to an upload prompt message if resume is missing,
                        otherwise cleared to None so downstream nodes don't see it.

HOW ROUTING USES THIS
---------------------
After this node, route_after_check_resume in routing.py reads the session store
directly to decide: resume found -> resume_parser, missing -> respond.
Setting final_text_response here means the respond node already has the message
it needs to show the user without any further work.
"""

from graph.session_store import has_resume
import logging

from graph.state import JobMarketState

logger = logging.getLogger(__name__)


def check_resume(state: JobMarketState) -> dict:
    """
    Check whether this session has an uploaded resume.

    If a resume exists, clear final_text_response (so it doesn't accidentally
    surface an old message) and let routing proceed to resume_parser.

    If no resume exists, write a friendly upload prompt into final_text_response
    so the respond node can display it to the user immediately.

    Args:
        state: Current graph state.  Only session_id is read.

    Returns:
        {"final_text_response": None}           — resume found, proceed normally
        {"final_text_response": "<prompt>"}     — resume missing, ask user to upload
    """
    session_id = state.get("session_id", "")

    if has_resume(session_id):
        logger.info("check_resume: resume found for session %s", session_id[:8])
        return {"final_text_response": None}

    logger.info("check_resume: no resume for session %s — prompting upload", session_id[:8])
    return {
        "final_text_response": (
            "I need your resume to compare it against the job market. "
            "Please upload your resume PDF and try again. "
            "You can also tell me which job titles and country you'd like me to analyse."
        ),
    }
