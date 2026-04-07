"""
graph/nodes/task_dispatcher.py — Pop the next task and set routing state fields.

ROLE IN THE GRAPH
-----------------
This node runs immediately after planner (first task) and again after each
sub-task completes when more tasks remain in the queue (the loop edge).

It has one simple job: pop the first task from task_queue and write its
parameters into the flat state fields that all routing functions read.

No LLM calls are made here — this is pure Python dict manipulation.

WHY A SEPARATE NODE?
---------------------
Keeping task dispatch in its own node means:
  - The routing functions (route_after_intent, route_after_cache_lookup, etc.)
    are completely unchanged — they still read intent/job_titles/country from
    the flat state fields, just as before.
  - The loop "planner → task_dispatcher → [pipeline] → task_dispatcher → ..."
    is visible in LangGraph traces, making debugging straightforward.
  - State at each checkpoint is inspectable: you can see exactly which task
    is active and which are still queued.

OUTPUTS written to state
------------------------
  intent           : from the current task
  job_titles       : from the current task (falls back to previous if empty)
  country          : from the current task (falls back to previous if empty)
  focused_topic    : from the current task
  total_posts      : from state (set at invocation) or DEFAULT_TOTAL_POSTS
  task_queue       : the queue with the first element removed
  params_confirmed : reset to False (fresh HITL flag for this task)
  report_confirmed : reset to False
  cache_hit        : reset to False
"""

import logging

from config import DEFAULT_TOTAL_POSTS
from graph.state import JobMarketState

logger = logging.getLogger(__name__)


def task_dispatcher(state: JobMarketState) -> dict:
    """
    Pop the next task from the queue and configure state for routing.

    Args:
        state: Current graph state.  Reads task_queue.

    Returns:
        Updated state fields for the current task + shortened task_queue.
    """
    queue = list(state.get("task_queue") or [])

    if not queue:
        # Should not happen (routing prevents calling this with empty queue)
        # but handle gracefully just in case.
        logger.warning("task_dispatcher: called with empty task_queue — no-op")
        return {}

    current_task = queue.pop(0)
    intent = current_task.get("intent", "general_question")

    # For job titles and country: prefer the task's value; fall back to
    # whatever was in state (so "use 10 posts instead" corrections preserve context)
    job_titles = current_task.get("job_titles") or state.get("job_titles") or []
    raw_country = current_task.get("country") or state.get("country")
    country = raw_country if raw_country and str(raw_country).lower() != "null" else None
    focused_topic = current_task.get("focused_topic")

    logger.info(
        "task_dispatcher: dispatching task — intent=%s titles=%s country=%s topic=%s  (%d remaining in queue)",
        intent, job_titles, country, focused_topic, len(queue),
    )

    return {
        "intent": intent,
        "job_titles": job_titles,
        "country": country,
        "focused_topic": focused_topic,
        "total_posts": current_task.get("total_posts") or state.get("total_posts") or DEFAULT_TOTAL_POSTS,
        "task_queue": queue,
        # Reset per-task HITL flags
        "params_confirmed": False,
        "report_confirmed": False,
        "cache_hit": False,
        # Reset per-task data payloads so a previous task's market data
        # doesn't accidentally feed into the next task's analysis
        "raw_job_postings": None,
        "extracted_requirements": None,
        "market_analysis_markdown": None,
        "skill_gap_markdown": None,
        "html_report_path": None,
        "final_text_response": None,
        "cache_key": None,
    }
