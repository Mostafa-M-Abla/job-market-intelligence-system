"""
graph/nodes/cache_lookup.py — Check whether market analysis data is already stored.

WHY WE CACHE
------------
Collecting and analysing job postings is expensive:
  - SerpAPI charges per search request.
  - LLM extraction across 30 job postings takes ~30 seconds and costs tokens.

If someone already asked for "AI Engineer jobs in Germany" today, we can reuse
those results for any subsequent request with the same parameters — regardless
of which user session is asking.  Market data for a given role + country does
not change minute-to-minute.

THE CACHE KEY
-------------
We build a SHA-256 hash of:
  { "titles": sorted(job_titles), "country": country, "date": today }

Sorting the titles ensures that ["ML Engineer", "AI Engineer"] and
["AI Engineer", "ML Engineer"] produce the same cache key.
Including today's date means cached data is automatically "expired" the next
day — we never serve data that is more than 24 hours old without a TTL check.
The DB also has an explicit `expires_at` column (7-day TTL by default).

OUTPUTS written to state
------------------------
  cache_hit   : True if matching data was found and is not expired
  cache_key   : the hash (stored so market_analyzer can write to the same key)
  raw_job_postings, extracted_requirements, market_analysis_markdown
              : populated only on a cache hit
"""

import logging

from config import DISABLE_CACHE
from graph.state import JobMarketState
from tools.market_cache_tool import MarketCacheReadTool, build_cache_key

logger = logging.getLogger(__name__)

# Lazy initialisation — the tool is created on first use, not at import time.
# This means the module can be imported without DB credentials being present.
_read_tool = None


def _get_tool() -> MarketCacheReadTool:
    """Return the shared MarketCacheReadTool instance, creating it if needed."""
    global _read_tool
    if _read_tool is None:
        _read_tool = MarketCacheReadTool()
    return _read_tool


def cache_lookup(state: JobMarketState) -> dict:
    """
    Check the market_analysis_cache table for a prior analysis that matches
    the current job_titles + country + today's date.

    Args:
        state: Current graph state. Reads job_titles and country.

    Returns:
        On cache HIT:
            {"cache_hit": True, "cache_key": ...,
             "raw_job_postings": [...], "extracted_requirements": [...],
             "market_analysis_markdown": "..."}

        On cache MISS (or missing params):
            {"cache_hit": False, "cache_key": <key or None>}
            The key is still returned on a miss so market_analyzer can use it
            to write to the cache after it completes the analysis.
    """
    job_titles = state.get("job_titles") or []
    country = state.get("country") or ""

    if DISABLE_CACHE:
        logger.info("cache_lookup: caching disabled — forcing miss")
        return {"cache_hit": False, "cache_key": None}

    if not job_titles or not country:
        # We cannot form a meaningful cache key without both pieces of information.
        # Treat this as a cache miss — the user will be asked to confirm params,
        # which will give us the missing information.
        return {"cache_hit": False, "cache_key": None}

    # Build the deterministic SHA-256 key.
    key = build_cache_key(job_titles, country)

    # Query the database — returns None if not found or if the entry has expired.
    cached = _get_tool().run({"cache_key": key})

    if cached is None:
        logger.info("cache_lookup: MISS for %s in %s (key %s…)", job_titles, country, key[:12])
        return {"cache_hit": False, "cache_key": key}

    logger.info(
        "cache_lookup: HIT for %s in %s — %d postings cached (key %s…)",
        job_titles, country, cached.get("total_posts", 0), key[:12],
    )
    # Cache hit — populate the state with the stored results so all downstream
    # nodes (skill_gap_analyzer, answer_focused, etc.) can use them directly.
    return {
        "cache_hit": True,
        "cache_key": key,
        "raw_job_postings": cached["raw_job_postings"],
        "extracted_requirements": cached["extracted_requirements"],
        "market_analysis_markdown": cached["market_analysis_markdown"],
    }
