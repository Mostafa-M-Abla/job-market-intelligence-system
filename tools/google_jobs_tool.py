"""
tools/google_jobs_tool.py — Collects real job postings via SerpAPI's Google Jobs API.

WHAT THIS TOOL DOES
-------------------
Given a list of job titles and a country, this tool:
  1. Searches Google Jobs for each title via SerpAPI.
  2. Filters results to remove irrelevant titles (using token overlap).
  3. Deduplicates results so the same posting doesn't appear twice.
  4. Fetches the full job description for each result (a second API call per job).
  5. Returns a structured list of job posting dicts.

LANGCHAIN TOOL PATTERN
-----------------------
This class inherits from LangChain's `BaseTool`.  This means:
  - It can be given directly to a LangChain agent or called manually via .run().
  - The `args_schema` (JobSearchInput Pydantic model) documents the expected
    inputs and enables automatic validation.
  - `_run()` is the synchronous implementation; `_arun()` wraps it for async use.

RETRIES
-------
SerpAPI occasionally returns transient errors.  We use `tenacity` to
automatically retry failed requests up to 3 times with exponential backoff
(wait 2s, then 4s, then up to 10s).
"""

import asyncio
import os
import re
import time
from typing import Any, Optional

import requests
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential


def _normalize(s: str) -> str:
    """Lowercase and collapse whitespace for consistent string comparison."""
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _title_is_similar(found_title: str, target_titles: list[str]) -> bool:
    """
    Check whether a job posting title is close enough to one of our target titles.

    We use a simple token-overlap approach: split both titles into words and
    measure what fraction of the target title's words appear in the found title.
    A threshold of 60% filters out obviously irrelevant postings while keeping
    variants like "Senior AI Engineer" when searching for "AI Engineer".

    Args:
        found_title:   The title of a job posting returned by SerpAPI.
        target_titles: The list of job titles the user is searching for.

    Returns:
        True if the found title is sufficiently similar to any target title.
    """
    # Tokenise the found title into a set of lowercase words.
    ft = set(re.findall(r"[a-zA-Z]+", _normalize(found_title)))
    if not ft:
        return False

    for tt in target_titles:
        tt_tokens = set(re.findall(r"[a-zA-Z]+", _normalize(tt)))
        if not tt_tokens:
            continue
        # Fraction of target title words that appear in the found title.
        overlap = len(ft.intersection(tt_tokens)) / max(1, len(tt_tokens))
        if overlap >= 0.6:
            return True
    return False


class JobSearchInput(BaseModel):
    """
    Input schema for GoogleJobsCollectorTool.

    Pydantic validates these fields automatically before _run() is called.
    The descriptions are also used by LangChain agents when deciding how to
    call this tool.
    """
    job_titles: list[str] = Field(description="List of job titles to search for, e.g. ['AI Engineer', 'ML Engineer']")
    country: str = Field(description="Country to search in, e.g. 'Germany' or 'United States'")
    limit: int = Field(default=30, description="Maximum total job postings to return across all titles")


class GoogleJobsCollectorTool(BaseTool):
    """
    Collects job postings from Google Jobs via SerpAPI.

    For each job title, the tool:
      1. Calls the Google Jobs search endpoint to get a list of matching postings.
      2. Filters by title similarity to avoid irrelevant results.
      3. Calls the Google Jobs listing endpoint to fetch the full description.
      4. Returns a deduplicated list of structured job dicts.
    """

    name: str = "google_jobs_collector"
    description: str = (
        "Uses SerpAPI Google Jobs to find job postings by title and country. "
        "Returns structured job postings with title, company, location, and description."
    )
    args_schema: type[BaseModel] = JobSearchInput

    # The API key is stored as a private attribute (not a Pydantic field) to
    # prevent it from being serialised or logged.
    _api_key: str = ""

    def model_post_init(self, __context: Any) -> None:
        """Called by Pydantic after __init__.  Loads the API key from the environment."""
        object.__setattr__(self, "_api_key", os.getenv("SERPAPI_API_KEY", ""))
        if not self._api_key:
            raise ValueError("SERPAPI_API_KEY env var is required.")

    def _run(self, job_titles: list[str], country: str, limit: int = 30) -> list[dict]:
        """
        Collect job postings synchronously.

        Iterates over each job title, searches Google Jobs, deduplicates,
        and stops once `limit` postings have been collected.

        Args:
            job_titles: Roles to search for.
            country:    Country to search in.
            limit:      Maximum number of postings to return.

        Returns:
            List of dicts, each with: title, company, location, via, job_id,
            apply_links, description, source.
        """
        collected: list[dict] = []
        # Track (title, company, location) tuples to avoid duplicates across
        # multiple title searches.
        seen_keys: set = set()

        for title in job_titles:
            # Build the search query in the format Google Jobs understands.
            q = f"{title} jobs in {country}"
            jobs = self._search(q)

            for j in jobs:
                job_id = j.get("job_id") or j.get("jobid") or j.get("id")
                found_title = j.get("title") or ""
                company = j.get("company_name") or j.get("company") or ""
                location = j.get("location") or ""

                # Skip results whose title doesn't match what we're looking for.
                if not _title_is_similar(found_title, job_titles):
                    continue

                # Skip exact duplicates (same title + company + location).
                key = (_normalize(found_title), _normalize(company), _normalize(location))
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                # Fetch the full job description via the listing endpoint.
                # This is a second API call per job, but gives much richer data
                # for the LLM to extract skills from.
                details = {}
                if job_id:
                    details = self._fetch_listing(job_id)
                    time.sleep(0.2)  # Polite rate limiting between listing calls

                collected.append({
                    "title": found_title,
                    "company": company,
                    "location": location,
                    "via": j.get("via"),          # Platform where it was posted (LinkedIn, etc.)
                    "job_id": job_id,
                    "apply_links": j.get("apply_options") or [],
                    # Prefer the full description from the listing endpoint; fall back
                    # to the shorter snippet from the search results if listing failed.
                    "description": (
                        (details.get("job_description") or details.get("description") or "").strip()
                        or (j.get("description") or "").strip()
                    ),
                    "source": "serpapi_google_jobs",
                })

                if len(collected) >= limit:
                    return collected  # Stop early once we have enough

            time.sleep(0.2)  # Polite pause between title searches

        return collected

    async def _arun(self, job_titles: list[str], country: str, limit: int = 30) -> list[dict]:
        """
        Async wrapper for _run().

        `asyncio.to_thread` runs the synchronous _run() in a thread pool so it
        doesn't block the async event loop used by the FastAPI server (Phase 2).
        """
        return await asyncio.to_thread(self._run, job_titles, country, limit)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _search(self, q: str) -> list[dict]:
        """
        Call SerpAPI's google_jobs engine to search for job postings.

        The @retry decorator automatically retries up to 3 times on failure,
        waiting 2s → 4s → 8s between attempts (exponential backoff, capped at 10s).

        Args:
            q: The search query string, e.g. "AI Engineer jobs in Germany".

        Returns:
            List of raw job result dicts from SerpAPI.
        """
        r = requests.get(
            "https://serpapi.com/search.json",
            params={"engine": "google_jobs", "q": q, "hl": "en", "api_key": self._api_key},
            timeout=30,
        )
        r.raise_for_status()  # Raises an exception on 4xx / 5xx responses
        return r.json().get("jobs_results", []) or []

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _fetch_listing(self, job_id: str) -> dict:
        """
        Fetch the full job description for a specific job posting.

        SerpAPI's google_jobs_listing engine takes a job_id and returns the
        complete posting including the full description text.

        Args:
            job_id: The unique identifier for the job posting from SerpAPI.

        Returns:
            Dict with the full job listing, including "job_description".
        """
        r = requests.get(
            "https://serpapi.com/search.json",
            params={"engine": "google_jobs_listing", "q": job_id, "hl": "en", "api_key": self._api_key},
            timeout=30,
        )
        r.raise_for_status()
        return r.json() or {}
