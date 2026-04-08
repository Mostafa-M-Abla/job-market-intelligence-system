"""
graph/nodes/job_collector.py — Fetch real job postings from SerpAPI.

ROLE IN THE GRAPH
-----------------
This node runs after the user has confirmed the search parameters in the HITL
confirm_search_params step.  It is always followed by requirements_extractor
and then market_analyzer (a fixed sequential pipeline).

This node only runs on a cache MISS — if market data already exists in the DB,
the entire job_collector -> requirements_extractor -> market_analyzer sequence
is skipped and the graph goes directly to the appropriate output node.

OUTPUTS written to state
------------------------
  raw_job_postings : list of job dicts, each with title, company, location,
                     description, job_id, apply_links, etc.
"""

import logging

from graph.state import JobMarketState
from tools.google_jobs_tool import GoogleJobsCollectorTool

logger = logging.getLogger(__name__)

# Lazy initialisation: the tool requires SERPAPI_API_KEY to be set.
# Delaying creation until first use means this module can be safely imported
# in environments without that key (e.g. during unit testing).
_tool = None


def _get_tool() -> GoogleJobsCollectorTool:
    """Return the shared GoogleJobsCollectorTool instance, creating it on first call."""
    global _tool
    if _tool is None:
        _tool = GoogleJobsCollectorTool()
    return _tool


def job_collector(state: JobMarketState) -> dict:
    """
    Call GoogleJobsCollectorTool to fetch job postings from SerpAPI.

    The tool handles deduplication, title filtering, and full-description
    fetching internally.  This node simply passes the parameters from state
    and stores the results.

    Args:
        state: Current graph state.  Reads job_titles, country, total_posts.

    Returns:
        {"raw_job_postings": [list of job dicts]}
        Each dict contains: title, company, location, via, job_id,
        apply_links, description, source.
    """
    job_titles = state.get("job_titles") or []
    country = state.get("country") or ""
    total_posts = state.get("total_posts", 30)

    logger.info("job_collector: fetching up to %d postings for %s in %s", total_posts, job_titles, country)
    postings = _get_tool().run({
        "job_titles": job_titles,
        "country": country,
        "limit": total_posts,
    })
    logger.info("job_collector: collected %d postings", len(postings))

    if not postings:
        titles_str = ", ".join(job_titles) if job_titles else "the requested roles"
        country_str = f" in {country}" if country else ""
        no_results_msg = (
            f"No job postings were found for **{titles_str}**{country_str}.\n\n"
            f"This can happen when the search parameters are very specific or "
            f"listings aren't currently available for this combination. "
            f"Please try again with different job titles, a different country, "
            f"or a broader search term."
        )
        existing = list(state.get("accumulated_responses") or [])
        return {
            "raw_job_postings": [],
            "accumulated_responses": existing + [no_results_msg],
        }

    return {"raw_job_postings": postings}
