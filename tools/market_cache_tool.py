"""
tools/market_cache_tool.py — Persistent cache for market analysis results.

WHY THIS EXISTS
---------------
Collecting and analysing job market data is slow and costs money:
  - Each SerpAPI search costs an API credit.
  - Analysing 30 job descriptions with an LLM takes ~30 seconds.

This module caches the complete results of each market analysis run in SQLite
so that future requests for the same job titles + country can skip the entire
data collection pipeline.

The cache is global (shared across all user sessions) because market data is
not user-specific — an analysis of "AI Engineer roles in Germany" is equally
valid for every user who asks about it.

CACHE KEY DESIGN
----------------
The cache key is a SHA-256 hash of:
    { titles: sorted(job_titles), country: country, date: today }

  - Sorting titles ensures order doesn't matter.
  - Including today's date means the key changes daily, so stale data is never
    served from the previous day's cache without checking the TTL.
  - The DB also stores an `expires_at` timestamp (default 7-day TTL).

TOOL PATTERN
------------
Two separate LangChain tools handle reads and writes:
  - MarketCacheReadTool  : used by cache_lookup node
  - MarketCacheWriteTool : used by market_analyzer node after analysis completes
"""

import hashlib
import json
from datetime import date, datetime, timedelta
from typing import Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from config import CACHE_TTL_DAYS
from db.connection import get_connection


def build_cache_key(job_titles: list[str], country: str) -> str:
    """
    Build a deterministic SHA-256 cache key for a given set of search parameters.

    The key is the same regardless of the order job titles are supplied, and
    changes automatically each day (so we never return yesterday's data without
    an explicit TTL check).

    Args:
        job_titles: List of job role names to include in the key.
        country:    Country name to include in the key.

    Returns:
        A 64-character lowercase hex string (SHA-256 digest).

    Example:
        build_cache_key(["ML Engineer", "AI Engineer"], "Germany")
        # -> SHA256 of '{"country":"germany","date":"2026-04-05","titles":["ai engineer","ml engineer"]}'
    """
    payload = json.dumps({
        # Sort and normalise so ["ML Eng", "AI Eng"] == ["AI Eng", "ML Eng"]
        "titles": sorted(t.lower().strip() for t in job_titles),
        "country": country.lower().strip(),
        "date": date.today().isoformat(),   # Changes the key each new day
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


# ── Input schemas for the two tools ───────────────────────────────────────────

class CacheReadInput(BaseModel):
    """Input for MarketCacheReadTool."""
    cache_key: str = Field(description="SHA-256 cache key to look up in the database")


class CacheWriteInput(BaseModel):
    """Input for MarketCacheWriteTool."""
    cache_key: str = Field(description="SHA-256 key under which to store the analysis")
    job_titles: list[str]
    country: str
    raw_job_postings: list[dict]
    extracted_requirements: list[dict]
    market_analysis_markdown: str
    total_posts: int


# ── Tools ─────────────────────────────────────────────────────────────────────

class MarketCacheReadTool(BaseTool):
    """
    Read a cached market analysis from the database.

    Returns the full cached analysis dict if a valid (non-expired) entry exists,
    or None if no match is found.  Used by the cache_lookup graph node.
    """

    name: str = "market_cache_read"
    description: str = "Look up a cached market analysis by cache key. Returns the cached data or None."
    args_schema: type[BaseModel] = CacheReadInput

    def _run(self, cache_key: str) -> Optional[dict]:
        """
        Query the market_analysis_cache table for a matching, non-expired entry.

        Args:
            cache_key: The SHA-256 key to look up.

        Returns:
            A dict with all cached fields, or None if not found / expired.
        """
        with get_connection() as conn:
            row = conn.execute(
                # Only return the row if it exists AND has not yet expired.
                "SELECT * FROM market_analysis_cache WHERE cache_key = ? AND "
                "(expires_at IS NULL OR expires_at > datetime('now'))",
                (cache_key,),
            ).fetchone()

        if row is None:
            return None

        # Deserialise JSON-stored fields back into Python objects.
        return {
            "cache_key": row["cache_key"],
            "job_titles": json.loads(row["job_titles"]),
            "country": row["country"],
            "analysis_date": row["analysis_date"],
            "raw_job_postings": json.loads(row["raw_job_postings"] or "[]"),
            "extracted_requirements": json.loads(row["extracted_requirements"] or "[]"),
            "market_analysis_markdown": row["market_analysis_markdown"],
            "total_posts": row["total_posts"],
        }

    async def _arun(self, cache_key: str) -> Optional[dict]:
        """Async wrapper — runs the synchronous DB query in a thread pool."""
        import asyncio
        return await asyncio.to_thread(self._run, cache_key)


class MarketCacheWriteTool(BaseTool):
    """
    Persist a completed market analysis to the database.

    Called by the market_analyzer node immediately after it produces a new
    analysis.  Uses INSERT OR REPLACE so that re-running an analysis for the
    same parameters overwrites the previous entry cleanly.
    """

    name: str = "market_cache_write"
    description: str = "Persist a completed market analysis to the cache database."
    args_schema: type[BaseModel] = CacheWriteInput

    def _run(
        self,
        cache_key: str,
        job_titles: list[str],
        country: str,
        raw_job_postings: list[dict],
        extracted_requirements: list[dict],
        market_analysis_markdown: str,
        total_posts: int,
    ) -> str:
        """
        Write the analysis to the market_analysis_cache table.

        Args:
            cache_key:                  Pre-built SHA-256 key from build_cache_key().
            job_titles:                 The job roles that were analysed.
            country:                    The country that was analysed.
            raw_job_postings:           The full list of job dicts from SerpAPI.
            extracted_requirements:     Per-job skill/cert extraction from the LLM.
            market_analysis_markdown:   The aggregated markdown report from the LLM.
            total_posts:                How many postings were collected.

        Returns:
            A short confirmation string for logging.
        """
        # Calculate expiry timestamp from now.
        expires_at = (datetime.now() + timedelta(days=CACHE_TTL_DAYS)).isoformat()

        with get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO market_analysis_cache
                  (cache_key, job_titles, country, analysis_date,
                   raw_job_postings, extracted_requirements,
                   market_analysis_markdown, total_posts, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cache_key,
                    # Store lists/dicts as JSON strings in SQLite.
                    json.dumps(sorted(t.lower().strip() for t in job_titles)),
                    country.lower().strip(),
                    date.today().isoformat(),
                    json.dumps(raw_job_postings),
                    json.dumps(extracted_requirements),
                    market_analysis_markdown,
                    total_posts,
                    expires_at,
                ),
            )
        return f"Cached under key {cache_key[:12]}... (expires {expires_at[:10]})"

    async def _arun(self, **kwargs) -> str:
        """Async wrapper — runs the synchronous DB write in a thread pool."""
        import asyncio
        return await asyncio.to_thread(self._run, **kwargs)
